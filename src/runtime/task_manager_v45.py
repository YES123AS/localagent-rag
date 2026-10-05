"""Persistent background worker for durable Agent Runs."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from threading import RLock
from typing import Any

from src.observability.tracing import runtime_span
from src.runtime.models import AgentRun, RunStatus, utc_now


class TaskManager:
    """Execute persisted tasks outside the Streamlit request lifecycle."""

    def __init__(
        self,
        runtime: Any,
        max_workers: int = 2,
        *,
        max_worker_recoveries: int = 1,
        auto_recover: bool = True,
    ):
        self.runtime = runtime
        self.worker_id = f"worker-{uuid.uuid4().hex[:12]}"
        self.max_worker_recoveries = max(0, min(int(max_worker_recoveries), 5))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(int(max_workers), 8)),
            thread_name_prefix="localagent-worker",
        )
        self._futures: dict[str, Future[AgentRun]] = {}
        self._lock = RLock()
        if auto_recover:
            self.recover_background_runs()

    @staticmethod
    def _elapsed_ms(started_at: str | None) -> float:
        if not started_at:
            return 0.0
        try:
            return max(
                0.0,
                (datetime.fromisoformat(utc_now()) - datetime.fromisoformat(started_at)).total_seconds()
                * 1000,
            )
        except ValueError:
            return 0.0

    def submit(self, conversation_id: str, goal: str, **kwargs: Any) -> AgentRun:
        run = self.runtime.create_run(conversation_id, goal, **kwargs)
        run.queued_at = utc_now()
        self.runtime.store.save_run(run)
        self.runtime.store.append_event(
            run.run_id, "run_queued", {"worker_id": self.worker_id}
        )
        self._schedule(run.run_id, resume=False)
        return run

    def _schedule(self, run_id: str, *, resume: bool) -> None:
        with self._lock:
            current = self._futures.get(run_id)
            if current and not current.done():
                return
            future = self._executor.submit(self._worker_entry, run_id, resume)
            self._futures[run_id] = future
            future.add_done_callback(
                lambda completed, target=run_id: self._worker_done(target, completed)
            )

    def _worker_entry(self, run_id: str, resume: bool) -> AgentRun:
        run = self.runtime.store.get_run(run_id)
        if run.status in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}:
            return run
        run.worker_id = self.worker_id
        run.queue_time_ms = self._elapsed_ms(run.queued_at)
        if resume:
            run.resume_count += 1
            run.recovery_result = "RESUMING"
        self.runtime.store.save_run(run)
        self.runtime.store.append_event(
            run_id,
            "worker_started",
            {
                "worker_id": self.worker_id,
                "queue_time_ms": run.queue_time_ms,
                "resume_count": run.resume_count,
            },
        )
        with runtime_span(
            "agent.worker",
            worker_id=self.worker_id,
            run_id=run_id,
            queue_time=run.queue_time_ms,
            resume_count=run.resume_count,
        ):
            result = self.runtime.resume_run(run_id) if resume else self.runtime.execute(run_id)
        if resume and result.status == RunStatus.COMPLETED:
            result.recovery_result = "SUCCESS"
            self.runtime.store.save_run(result)
        return result

    def _worker_done(self, run_id: str, future: Future[AgentRun]) -> None:
        try:
            error = future.exception()
        except BaseException as exc:  # pragma: no cover
            error = exc
        if error is None:
            result = future.result()
            self.runtime.store.append_event(
                run_id,
                "worker_finished",
                {"worker_id": self.worker_id, "status": result.status.value},
            )
            return

        run = self.runtime.store.get_run(run_id)
        attempts = int(run.metadata.get("worker_recovery_attempts", 0)) + 1
        run.metadata["worker_recovery_attempts"] = attempts
        run.metadata["recovery_reason"] = (
            f"Worker crashed: {type(error).__name__}: {error}"
        )
        run.recovery_result = "CRASHED"
        if run.status == RunStatus.RUNNING:
            run.transition(RunStatus.PAUSED)
        self.runtime.store.save_checkpoint(run)
        self.runtime.store.append_event(
            run_id,
            "worker_crashed",
            {
                "worker_id": self.worker_id,
                "error_type": type(error).__name__,
                "attempt": attempts,
            },
        )
        if attempts <= self.max_worker_recoveries and run.status == RunStatus.PAUSED:
            run.queued_at = utc_now()
            self.runtime.store.save_run(run)
            self.runtime.store.append_event(
                run_id,
                "recovery_queued",
                {"attempt": attempts, "worker_id": self.worker_id},
            )
            self._schedule(run_id, resume=True)
        else:
            run.recovery_result = "FAILED"
            run.error = f"Worker recovery exhausted after {attempts} attempt(s)"
            run.error_type = type(error).__name__
            if run.status == RunStatus.PAUSED:
                run.transition(RunStatus.FAILED)
            self.runtime.store.save_checkpoint(run)

    def recover_background_runs(self) -> list[str]:
        interrupted = self.runtime.store.recover_interrupted_runs()
        interrupted_ids = {run.run_id for run in interrupted}
        scheduled: list[str] = []
        for run in self.runtime.store.list_runs(limit=500):
            # Specialist child Runs are owned and resumed by their parent
            # Multi-Agent Run, never scheduled as independent UI tasks.
            if run.metadata.get("parent_run_id"):
                continue
            should_resume = run.run_id in interrupted_ids or (
                run.status == RunStatus.PAUSED
                and bool(run.metadata.get("recovery_reason"))
            )
            if run.status == RunStatus.PENDING:
                if not run.queued_at:
                    run.queued_at = utc_now()
                    self.runtime.store.save_run(run)
                self._schedule(run.run_id, resume=False)
                scheduled.append(run.run_id)
            elif should_resume and run.status == RunStatus.PAUSED:
                run.queued_at = utc_now()
                self.runtime.store.save_run(run)
                self.runtime.store.append_event(
                    run.run_id,
                    "recovery_queued",
                    {"worker_id": self.worker_id, "startup": True},
                )
                self._schedule(run.run_id, resume=True)
                scheduled.append(run.run_id)
        return scheduled

    def status(self, run_id: str) -> AgentRun:
        return self.runtime.store.get_run(run_id)

    def events(self, run_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
        return self.runtime.store.list_events(
            run_id, after_sequence=after_sequence
        )

    def pause(self, run_id: str) -> AgentRun:
        return self.runtime.pause_run(run_id)

    def cancel(self, run_id: str) -> AgentRun:
        return self.runtime.cancel_run(run_id)

    def resume(self, run_id: str) -> AgentRun:
        run = self.runtime.store.get_run(run_id)
        if run.status == RunStatus.PAUSED:
            run.queued_at = utc_now()
            self.runtime.store.save_run(run)
            self._schedule(run_id, resume=True)
        return self.runtime.store.get_run(run_id)

    def approve(self, approval_id: str) -> AgentRun:
        decision = self.runtime.store.decide_approval(approval_id, True)
        run_id = str(decision["run_id"])
        run = self.runtime.store.get_run(run_id)
        run.queued_at = utc_now()
        self.runtime.store.save_run(run)
        self._schedule(run_id, resume=True)
        return run

    def reject(self, approval_id: str) -> AgentRun:
        decision = self.runtime.store.decide_approval(approval_id, False)
        run_id = str(decision["run_id"])
        run = self.runtime.store.get_run(run_id)
        run.queued_at = utc_now()
        self.runtime.store.save_run(run)
        self._schedule(run_id, resume=True)
        return run

    def wait(self, run_id: str, timeout: float = 30.0) -> AgentRun:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.status(run_id)
            if run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.WAITING_APPROVAL,
            }:
                with self._lock:
                    future = self._futures.get(run_id)
                if future is None or future.done():
                    return self.status(run_id)
            time.sleep(0.01)
        raise TimeoutError(f"Run {run_id} did not settle within {timeout}s")

    def recoverable_runs(self) -> list[AgentRun]:
        return [
            run
            for run in self.runtime.store.list_runs(limit=500)
            if not run.metadata.get("parent_run_id")
            if run.status
            in {
                RunStatus.PENDING,
                RunStatus.RUNNING,
                RunStatus.PAUSED,
                RunStatus.WAITING_APPROVAL,
            }
        ]

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)
