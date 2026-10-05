"""Durable sequential Agent Runtime layered beside the existing LangGraph graph."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from src.agent.tool_policy import PolicyOutcome, ToolPolicyEngine
from src.memory.context_builder import ContextBuilder
from src.observability.tracing import record_runtime_event, runtime_span
from src.runtime.errors import (
    AgentError,
    BudgetExceededError,
    IdempotencyConflictError,
    PlanningError,
)
from src.runtime.models import AgentRun, RunStatus
from src.runtime.resilience import retry_with_backoff
from src.runtime.runs import RunStore
from src.tools.registry import ToolRegistry


Planner = Callable[[str], dict[str, Any]]
Synthesizer = Callable[[AgentRun, dict[str, Any]], str]
StreamSynthesizer = Callable[[AgentRun, dict[str, Any]], Iterable[str]]


class AgentRuntime:
    def __init__(
        self,
        store: RunStore,
        registry: ToolRegistry,
        planner: Planner,
        synthesizer: Synthesizer,
        *,
        stream_synthesizer: StreamSynthesizer | None = None,
        policy: ToolPolicyEngine | None = None,
        context_builder: ContextBuilder | None = None,
        sleep: Callable[[float], Any] = time.sleep,
    ):
        self.store = store
        self.registry = registry
        self.planner = planner
        self.synthesizer = synthesizer
        self.stream_synthesizer = stream_synthesizer
        self.policy = policy or ToolPolicyEngine()
        self.context_builder = context_builder or ContextBuilder()
        self.sleep = sleep

    def create_run(self, conversation_id: str, goal: str, **kwargs: Any) -> AgentRun:
        return self.store.create_run(conversation_id, goal, **kwargs)

    @staticmethod
    def _signature(tool_name: str, arguments: dict[str, Any]) -> str:
        raw = json.dumps([tool_name, arguments], sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _steps(run: AgentRun) -> list[dict[str, Any]]:
        steps = run.plan.get("steps", []) if isinstance(run.plan, dict) else []
        if not isinstance(steps, list):
            raise PlanningError("Plan steps must be a list")
        return [dict(step) for step in steps]

    def _fresh_run(self, run_id: str) -> AgentRun:
        return self.store.get_run(run_id)

    def _event(self, run_id: str, event_type: str, **payload: Any) -> None:
        self.store.append_event(run_id, event_type, payload)

    @staticmethod
    def _result_evidence(result: Any, tool_name: str, tool_call_id: str) -> list[dict[str, Any]]:
        if isinstance(result, dict) and isinstance(result.get("evidence"), list):
            return [dict(item) for item in result["evidence"]]
        return [{
            "source_type": "tool", "title": tool_name, "content": result,
            "tool_call_id": tool_call_id,
        }]

    def _synthesize(self, run: AgentRun, context: dict[str, Any]) -> str:
        self._event(run.run_id, "synthesis_started", evidence_count=len(run.evidence))
        if self.stream_synthesizer is None:
            result = self.synthesizer(run, context)
        else:
            chunks: list[str] = []
            for token in self.stream_synthesizer(run, context):
                text = str(token)
                if not text:
                    continue
                chunks.append(text)
                self._event(run.run_id, "answer_token", token=text)
            result = "".join(chunks)
            if not result:
                raise ValueError("Streaming synthesis returned no tokens")
        self._event(run.run_id, "final_answer", content=result)
        return result

    def execute(self, run_id: str) -> AgentRun:
        run = self._fresh_run(run_id)
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return run
        if run.status == RunStatus.WAITING_APPROVAL:
            pending = self.store.pending_approval(run_id)
            if pending:
                return run
            run.transition(RunStatus.RUNNING)
        elif run.status in {RunStatus.PENDING, RunStatus.PAUSED}:
            run.transition(RunStatus.RUNNING)
        self.store.save_run(run)
        self._event(run.run_id, "worker_execution_started", status=run.status.value)
        record_runtime_event(
            "agent.run.start", run_id=run.run_id, conversation_id=run.conversation_id,
            status=run.status.value,
        )

        try:
            if not run.plan:
                self._event(run.run_id, "planning_started")
                run.budget.consume("llm")
                run.plan = dict(self.planner(run.goal))
                self._steps(run)
                self.store.save_checkpoint(run)
                self._event(run.run_id, "planning_completed", plan_steps=len(self._steps(run)))
                record_runtime_event(
                    "agent.run.plan", run_id=run.run_id,
                    plan_steps=len(self._steps(run)), checkpoint_count=run.checkpoint_count,
                )

            for index, step in enumerate(self._steps(run)):
                run = self._fresh_run(run_id)
                if run.status != RunStatus.RUNNING:
                    return run
                step_id = int(step.get("id", index + 1))
                if step_id in run.completed_steps:
                    continue
                if str(step.get("action", "tool")) == "synthesize":
                    run.completed_steps.append(step_id)
                    run.current_step = index + 1
                    self.store.save_checkpoint(run)
                    continue

                tool_name = str(step.get("tool") or "")
                tool = self.registry.get(tool_name)
                arguments = dict(step.get("arguments") or {})
                if not arguments:
                    arguments = {"query": str(step.get("query") or run.goal)}
                signature = self._signature(tool_name, arguments)
                if any(call.get("signature") == signature and call.get("status") == "success" for call in run.tool_calls):
                    self._event(run.run_id, "tool_call_deduplicated", step_id=step_id, tool=tool_name)
                    run.completed_steps.append(step_id)
                    run.current_step = index + 1
                    self.store.save_checkpoint(run)
                    continue

                approval = self.store.approval_for_step(run_id, step_id)
                if approval and approval["status"] == "REJECTED":
                    run.tool_calls.append({
                        "tool_call_id": f"call-{uuid.uuid4().hex}", "step_id": step_id,
                        "tool": tool_name, "arguments": arguments, "signature": signature,
                        "status": "rejected", "error": "Human rejected the operation",
                    })
                    run.completed_steps.append(step_id)
                    run.current_step = index + 1
                    self.store.save_checkpoint(run)
                    continue
                approved = bool(approval and approval["status"] == "APPROVED")
                decision = self.policy.evaluate(run, tool.metadata, approved=approved)
                record_runtime_event(
                    "agent.run.policy", run_id=run.run_id, tool=tool_name,
                    step_id=step_id, outcome=decision.outcome.value,
                )
                if decision.outcome == PolicyOutcome.REQUIRES_APPROVAL:
                    self.store.save_checkpoint(run, {**run.to_dict(), "pending_step": step})
                    self.store.request_approval(run, step_id, tool_name, decision.reason)
                    self._event(run.run_id, "approval_required", step_id=step_id, tool=tool_name)
                    return self._fresh_run(run_id)
                if decision.outcome == PolicyOutcome.DENY:
                    run.error = decision.reason
                    run.error_type = "PolicyDenied"
                    break

                idempotency_key = ""
                idempotency_record: dict[str, Any] | None = None
                if tool.metadata.side_effecting:
                    idempotency_key = str(
                        step.get("idempotency_key")
                        or arguments.get("idempotency_key")
                        or hashlib.sha256(
                            f"{run.run_id}:{step_id}:{signature}".encode("utf-8")
                        ).hexdigest()
                    )
                    idempotency_record = self.store.begin_idempotent_call(
                        idempotency_key, run.run_id, tool_name
                    )
                    if (
                        idempotency_record["status"] == "STARTED"
                        and idempotency_record["run_id"] != run.run_id
                    ):
                        raise IdempotencyConflictError(
                            "The same idempotency key is already executing in another Run"
                        )
                    if idempotency_record["status"] == "COMPLETED":
                        cached_result = idempotency_record["output"]
                        cached_call_id = f"call-{uuid.uuid4().hex}"
                        run.tool_calls.append({
                            "tool_call_id": cached_call_id, "step_id": step_id,
                            "tool": tool_name, "arguments": arguments, "signature": signature,
                            "status": "success", "output": cached_result,
                            "idempotency_key": idempotency_key, "idempotency_reused": True,
                        })
                        run.evidence.extend(
                            self._result_evidence(cached_result, tool_name, cached_call_id)
                        )
                        run.completed_steps.append(step_id)
                        run.current_step = index + 1
                        self._event(
                            run.run_id, "tool_call_deduplicated", step_id=step_id,
                            tool=tool_name, idempotency_key=idempotency_key,
                        )
                        self.store.save_checkpoint(run)
                        continue

                run.current_step = index
                self.store.save_checkpoint(run, {**run.to_dict(), "pending_step": step})
                call = {
                    "tool_call_id": f"call-{uuid.uuid4().hex}", "step_id": step_id,
                    "tool": tool_name, "arguments": arguments, "signature": signature,
                    "status": "running", "started_at": time.time(), "retry_count": 0,
                    "idempotency_key": idempotency_key or None,
                }
                run.tool_calls.append(call)
                run.budget.consume("tool")
                if tool.metadata.external:
                    run.budget.consume("web")
                # Persist the in-flight call and consumed budget before invoking an
                # external side effect. Pause/cancel can now observe this boundary.
                self.store.save_run(run)
                self._event(
                    run.run_id, "tool_call_started", step_id=step_id, tool=tool_name,
                    tool_call_id=call["tool_call_id"], idempotency_key=idempotency_key or None,
                )
                produced_evidence: list[dict[str, Any]] = []

                def on_retry(exc: BaseException, attempt: int, delay: float) -> None:
                    call["retry_count"] = attempt
                    call.setdefault("retry_events", []).append(
                        {"error_type": type(exc).__name__, "delay_seconds": delay}
                    )
                    run.retry_count += 1
                    self._event(
                        run.run_id, "retry", tool=tool_name, step_id=step_id,
                        attempt=attempt, delay_seconds=delay, error_type=type(exc).__name__,
                    )

                try:
                    invoke_arguments = dict(arguments)
                    invoke_arguments["_runtime_evidence"] = list(run.evidence)
                    if idempotency_key:
                        invoke_arguments["idempotency_key"] = idempotency_key
                    with runtime_span(
                        "agent.run.tool_call", run_id=run.run_id, tool=tool_name,
                        step_id=step_id, tool_call_id=call["tool_call_id"],
                        idempotency_key=idempotency_key or None,
                    ):
                        result = retry_with_backoff(
                            lambda: tool.invoke(invoke_arguments), tool.metadata.retry_policy,
                            sleep=self.sleep, on_retry=on_retry,
                        )
                    call.update(status="success", output=result, completed_at=time.time())
                    produced_evidence = self._result_evidence(
                        result, tool_name, call["tool_call_id"]
                    )
                    if idempotency_key:
                        self.store.complete_idempotent_call(idempotency_key, result)
                except Exception as exc:
                    call.update(status="failed", error=str(exc), error_type=type(exc).__name__, completed_at=time.time())
                    run.error = str(exc)
                    run.error_type = type(exc).__name__
                    if idempotency_key:
                        self.store.fail_idempotent_call(idempotency_key)
                # Reload lifecycle state so a concurrent pause/cancel is never
                # overwritten by this worker's older RUNNING snapshot.
                latest = self._fresh_run(run_id)
                for call_index, persisted_call in enumerate(latest.tool_calls):
                    if persisted_call.get("tool_call_id") == call["tool_call_id"]:
                        latest.tool_calls[call_index] = call
                        break
                else:
                    latest.tool_calls.append(call)
                latest.retry_count = max(latest.retry_count, run.retry_count)
                latest.evidence.extend(produced_evidence)
                if call["status"] == "failed":
                    latest.error = run.error
                    latest.error_type = run.error_type
                run = latest
                run.completed_steps.append(step_id)
                run.current_step = index + 1
                self.store.save_checkpoint(run)
                self._event(
                    run.run_id, "tool_call_completed", step_id=step_id, tool=tool_name,
                    status=call["status"], error_type=call.get("error_type", ""),
                    idempotency_key=idempotency_key or None,
                )

            run = self._fresh_run(run_id)
            if run.status != RunStatus.RUNNING:
                return run
            memory = list(run.metadata.get("memory", []))
            conversation = list(run.metadata.get("conversation", []))
            with runtime_span("agent.run.context_build", run_id=run.run_id):
                context = self.context_builder.build(
                    query=run.goal, goal=run.goal, plan=run.plan, conversation=conversation,
                    memory=memory, evidence=run.evidence,
                )
            run.context_usage = context.token_usage
            run.budget.consume("tokens", min(context.token_usage["total"], run.budget.remaining("tokens")))
            if run.evidence:
                run.budget.consume("llm")
                with runtime_span(
                    "agent.run.synthesis", run_id=run.run_id,
                    evidence_count=len(run.evidence), token_usage=context.token_usage["total"],
                ):
                    run.result = self._synthesize(run, context.to_dict())
                run.transition(RunStatus.COMPLETED)
            else:
                run.error = run.error or "No evidence was collected within policy and budget limits"
                run.error_type = run.error_type or "NoEvidence"
                run.transition(RunStatus.FAILED)
        except BudgetExceededError as exc:
            run = self._fresh_run(run_id)
            run.error, run.error_type = str(exc), type(exc).__name__
            if run.evidence:
                context = self.context_builder.build(query=run.goal, goal=run.goal, plan=run.plan, evidence=run.evidence)
                run.result = self._synthesize(run, context.to_dict())
                run.transition(RunStatus.COMPLETED)
            else:
                run.transition(RunStatus.FAILED)
        except Exception as exc:
            run = self._fresh_run(run_id)
            run.error, run.error_type = str(exc), type(exc).__name__
            if run.status == RunStatus.RUNNING:
                run.transition(RunStatus.FAILED)
        self.store.save_checkpoint(run)
        self._event(
            run.run_id, "run_completed", status=run.status.value,
            error_type=run.error_type, recovery_result=run.recovery_result,
        )
        record_runtime_event(
            "agent.run.complete", run_id=run.run_id, conversation_id=run.conversation_id,
            status=run.status.value, tool_calls=len(run.tool_calls), retry_count=run.retry_count,
            replan_count=run.replan_count, checkpoint_count=run.checkpoint_count,
            approval_count=run.approval_count, evidence_count=len(run.evidence),
            token_usage=run.budget.tokens, error_type=run.error_type,
            worker_id=run.worker_id, queue_time=run.queue_time_ms,
            resume_count=run.resume_count, recovery_result=run.recovery_result,
        )
        return run

    def pause_run(self, run_id: str) -> AgentRun:
        run = self.store.pause(run_id)
        self.store.save_checkpoint(run)
        return run

    def cancel_run(self, run_id: str) -> AgentRun:
        run = self.store.cancel(run_id)
        self.store.save_checkpoint(run)
        return run

    def resume_run(self, run_id: str) -> AgentRun:
        run = self.store.get_run(run_id)
        if run.status == RunStatus.WAITING_APPROVAL and self.store.pending_approval(run_id):
            return run
        if run.status != RunStatus.PAUSED and run.status != RunStatus.WAITING_APPROVAL:
            raise ValueError(f"Run {run_id} is not resumable from {run.status.value}")
        checkpoint = self.store.load_checkpoint(run_id)
        restored = AgentRun.from_dict(checkpoint)
        # The lifecycle table is authoritative for controls; checkpoint data is
        # authoritative for plan/progress/evidence.
        restored.status = run.status
        restored.updated_at = run.updated_at
        restored.approval_count = max(restored.approval_count, run.approval_count)
        restored.worker_id = run.worker_id
        restored.queued_at = run.queued_at
        restored.queue_time_ms = run.queue_time_ms
        restored.resume_count = run.resume_count
        restored.recovery_result = run.recovery_result
        restored.metadata.update(run.metadata)
        self.store.save_run(restored)
        return self.execute(run_id)

    def approve(self, approval_id: str) -> AgentRun:
        decision = self.store.decide_approval(approval_id, True)
        return self.execute(str(decision["run_id"]))

    def reject(self, approval_id: str) -> AgentRun:
        decision = self.store.decide_approval(approval_id, False)
        return self.execute(str(decision["run_id"]))
