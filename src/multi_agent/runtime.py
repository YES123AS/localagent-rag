"""Bounded multi-agent coordinator reusing durable AgentRuntime for every subtask."""

from __future__ import annotations

import time
from contextvars import copy_context
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import RLock
from typing import Any

from src.agent.runtime import AgentRuntime
from src.agent.tool_policy import ToolPolicyEngine
from src.memory.context_builder import ContextBuilder
from src.observability.tracing import record_runtime_event, runtime_span
from src.runtime.budget import ExecutionBudget
from src.runtime.errors import BudgetExceededError
from src.runtime.models import AgentRun, RunStatus
from src.runtime.runs import RunStore
from src.tools.registry import ToolRegistry

from .models import (
    AgentRole,
    CriticResult,
    Handoff,
    MultiAgentPlan,
    ResearchTask,
    TaskStatus,
)
from .store import SharedEvidenceStore
from .supervisor import Supervisor


Critic = Callable[[str, list[dict[str, Any]], list[ResearchTask]], CriticResult | dict[str, Any]]
Writer = Callable[[AgentRun, list[dict[str, Any]], CriticResult], str]
StreamWriter = Callable[[AgentRun, list[dict[str, Any]], CriticResult], Iterable[str]]


class MultiAgentRuntime:
    """Supervisor -> parallel specialists -> shared evidence -> critic -> writer."""

    def __init__(
        self,
        store: RunStore,
        registry: ToolRegistry,
        *,
        supervisor: Supervisor | None = None,
        evidence_store: SharedEvidenceStore | None = None,
        critic: Critic | None = None,
        writer: Writer | None = None,
        stream_writer: StreamWriter | None = None,
        max_agents: int = 4,
        max_subtasks: int = 4,
        max_research_rounds: int = 2,
        max_critic_rounds: int = 2,
        max_total_tool_calls: int = 8,
        sleep: Callable[[float], Any] = time.sleep,
    ):
        self.store = store
        self.registry = registry
        self.max_agents = max(1, min(int(max_agents), 4))
        self.max_subtasks = max(1, min(int(max_subtasks), 8))
        self.max_research_rounds = max(1, min(int(max_research_rounds), 5))
        self.max_critic_rounds = max(1, min(int(max_critic_rounds), 5))
        self.max_total_tool_calls = max(1, min(int(max_total_tool_calls), 32))
        self.supervisor = supervisor or Supervisor(
            max_agents=self.max_agents, max_subtasks=self.max_subtasks
        )
        self.evidence_store = evidence_store or SharedEvidenceStore(store.path)
        self.critic = critic or self._deterministic_critic
        self.writer = writer or self._deterministic_writer
        self.stream_writer = stream_writer
        self.sleep = sleep
        self._tool_budget_lock = RLock()

    def create_run(self, conversation_id: str, goal: str, **kwargs: Any) -> AgentRun:
        metadata = dict(kwargs.pop("metadata", {}))
        metadata["execution_mode"] = "multi_agent"
        return self.store.create_run(conversation_id, goal, metadata=metadata, **kwargs)

    def _event(self, run_id: str, event_type: str, **payload: Any) -> None:
        self.store.append_event(run_id, event_type, payload)

    def _handoff(
        self,
        parent_run_id: str,
        from_agent: AgentRole,
        to_agent: AgentRole,
        task: str,
        *,
        context: dict[str, Any] | None = None,
        evidence_refs: list[str] | None = None,
        reason: str,
        status: str = "COMPLETED",
    ) -> Handoff:
        handoff = self.evidence_store.save_handoff(
            Handoff(
                parent_run_id=parent_run_id,
                from_agent=from_agent,
                to_agent=to_agent,
                task=task,
                context=context or {},
                evidence_refs=evidence_refs or [],
                reason=reason,
                status=status,
            )
        )
        self._event(
            parent_run_id,
            "agent_handoff",
            handoff_id=handoff.handoff_id,
            from_agent=from_agent.value,
            to_agent=to_agent.value,
            task=task,
            evidence_refs=handoff.evidence_refs,
            reason=reason,
            status=status,
        )
        record_runtime_event(
            "multi_agent.handoff",
            run_id=parent_run_id,
            handoff_id=handoff.handoff_id,
            from_agent=from_agent.value,
            to_agent=to_agent.value,
            evidence_count=len(handoff.evidence_refs),
            status=status,
        )
        return handoff

    @staticmethod
    def _task_dicts(plan: MultiAgentPlan) -> list[dict[str, Any]]:
        return [task.model_dump(mode="json") for task in plan.tasks]

    @staticmethod
    def _plan_from_run(run: AgentRun) -> MultiAgentPlan:
        return MultiAgentPlan.model_validate(
            {"goal": run.plan.get("goal", run.goal), "tasks": run.plan.get("tasks", [])}
        )

    def _save_plan(self, run: AgentRun, plan: MultiAgentPlan) -> None:
        latest = self.store.get_run(run.run_id)
        latest.plan = {
            "type": "multi_agent",
            "goal": plan.goal,
            "tasks": self._task_dicts(plan),
        }
        counts = {status.value.lower(): 0 for status in TaskStatus}
        for task in plan.tasks:
            counts[task.status.value.lower()] += 1
        latest.metadata.update(run.metadata)
        latest.metadata["task_status"] = counts
        latest.replan_count = max(latest.replan_count, run.replan_count)
        latest.budget = run.budget
        self.store.save_checkpoint(latest)
        run.plan = latest.plan
        run.metadata = latest.metadata
        run.status = latest.status
        run.replan_count = latest.replan_count
        run.budget = latest.budget

    def _allowed_tools(self, role: AgentRole) -> tuple[str, ...]:
        if role == AgentRole.LOCAL_RESEARCH:
            return ("knowledge_base",)
        if role == AgentRole.WEB_RESEARCH:
            return tuple(
                name for name in self.registry.names()
                if name == "web_search" or name.startswith("mcp.")
            )
        return ()

    def _reserve_tool_call(self, parent_run_id: str) -> None:
        with self._tool_budget_lock:
            parent = self.store.get_run(parent_run_id)
            used = int(parent.metadata.get("multi_agent_tool_calls", 0))
            if used >= self.max_total_tool_calls:
                raise BudgetExceededError(
                    f"multi-agent tool budget exhausted ({used}/{self.max_total_tool_calls})"
                )
            parent.metadata["multi_agent_tool_calls"] = used + 1
            self.store.save_run(parent)

    def _specialist_runtime(
        self, task: ResearchTask, parent_run_id: str | None = None
    ) -> AgentRuntime:
        allowed = self._allowed_tools(task.assigned_agent)
        if not allowed:
            raise ValueError(f"No tools allowed for {task.assigned_agent.value}")
        tool_name = "knowledge_base" if task.assigned_agent == AgentRole.LOCAL_RESEARCH else "web_search"
        if tool_name not in allowed or tool_name not in self.registry.names():
            raise KeyError(f"Required specialist tool is unavailable: {tool_name}")

        def planner(goal: str) -> dict[str, Any]:
            return {
                "goal": goal,
                "steps": [
                    {"id": 1, "tool": tool_name, "query": goal, "reason": task.title},
                    {"id": 2, "action": "synthesize", "depends_on": [1]},
                ],
            }

        scoped_registry = self.registry.scoped(allowed)
        if parent_run_id:
            guarded_registry = ToolRegistry()
            for name in scoped_registry.names():
                tool = scoped_registry.get(name)

                def guarded(arguments, registered=tool):
                    self._reserve_tool_call(parent_run_id)
                    return registered.invoke(arguments)

                guarded_registry.register(tool.metadata, guarded)
            scoped_registry = guarded_registry
        return AgentRuntime(
            self.store,
            scoped_registry,
            planner,
            lambda run, context: f"{task.title}: collected {len(run.evidence)} evidence item(s)",
            policy=ToolPolicyEngine(),
            context_builder=ContextBuilder(),
            sleep=self.sleep,
        )

    def _prepare_child(self, parent: AgentRun, task: ResearchTask) -> AgentRun:
        if task.run_id:
            return self.store.get_run(task.run_id)
        runtime = self._specialist_runtime(task, parent.run_id)
        child = runtime.create_run(
            parent.conversation_id,
            task.query,
            budget=ExecutionBudget(
                max_llm_calls=3,
                # One normal call plus one bounded crash-resume attempt. The
                # parent MAX_TOTAL_TOOL_CALLS remains the global upper bound.
                max_tool_calls=2,
                max_web_calls=2 if task.assigned_agent == AgentRole.WEB_RESEARCH else 0,
                max_replans=0,
                max_tokens=max(1000, min(parent.budget.max_tokens, 6000)),
                max_duration_seconds=parent.budget.max_duration_seconds,
            ),
            metadata={
                "execution_mode": "specialist",
                "parent_run_id": parent.run_id,
                "task_id": task.task_id,
                "agent_name": task.assigned_agent.value,
            },
        )
        task.run_id = child.run_id
        return child

    def _execute_task(self, parent: AgentRun, task: ResearchTask) -> ResearchTask:
        started = time.monotonic()
        task.status = TaskStatus.RUNNING
        self._handoff(
            parent.run_id,
            AgentRole.SUPERVISOR,
            task.assigned_agent,
            task.title,
            context={"query": task.query, "depends_on": task.depends_on},
            reason="Supervisor assigned a bounded specialist task",
        )
        runtime = self._specialist_runtime(task, parent.run_id)
        child = self._prepare_child(parent, task)
        with runtime_span(
            "multi_agent.specialist",
            run_id=parent.run_id,
            task_id=task.task_id,
            child_run_id=child.run_id,
            agent_name=task.assigned_agent.value,
        ):
            if child.status == RunStatus.PENDING:
                result = runtime.execute(child.run_id)
            elif child.status == RunStatus.PAUSED:
                result = runtime.resume_run(child.run_id)
            elif child.status == RunStatus.RUNNING and parent.resume_count > 0:
                interrupted = self.store.pause(child.run_id)
                self.store.save_checkpoint(interrupted)
                result = runtime.resume_run(child.run_id)
            else:
                result = child
        task.retry_count = result.retry_count
        if result.status == RunStatus.COMPLETED:
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.FAILED
            task.error = result.error or result.status.value

        evidence_refs: list[str] = []
        reused_count = 0
        for item in result.evidence:
            record, reused = self.evidence_store.add(
                parent_run_id=parent.run_id,
                evidence=dict(item),
                agent_id=task.assigned_agent.value,
                task_id=task.task_id,
            )
            evidence_refs.append(record.evidence_id)
            reused_count += int(reused)
        self._handoff(
            parent.run_id,
            task.assigned_agent,
            AgentRole.CRITIC,
            task.title,
            context={
                "child_run_id": child.run_id,
                "status": task.status.value,
                "retry_count": task.retry_count,
            },
            evidence_refs=evidence_refs,
            reason="Specialist submitted evidence for independent verification",
            status=task.status.value,
        )
        self._event(
            parent.run_id,
            "specialist_completed",
            agent_name=task.assigned_agent.value,
            task_id=task.task_id,
            child_run_id=child.run_id,
            status=task.status.value,
            evidence_count=len(evidence_refs),
            evidence_reused=reused_count,
            retry_count=task.retry_count,
            tool_calls=len(result.tool_calls),
            duration=round(time.monotonic() - started, 6),
        )
        return task

    def _run_ready_tasks(self, run: AgentRun, plan: MultiAgentPlan) -> bool:
        completed = {task.task_id for task in plan.tasks if task.status == TaskStatus.COMPLETED}
        ready = [
            task for task in plan.tasks
            if task.status == TaskStatus.PENDING and set(task.depends_on).issubset(completed)
        ]
        if not ready:
            return False
        remaining_calls = self.max_total_tool_calls - sum(
            len(self.store.get_run(task.run_id).tool_calls)
            for task in plan.tasks if task.run_id
        )
        ready = ready[: max(0, min(self.max_agents, remaining_calls))]
        if not ready:
            return False
        # Persist child Run IDs before entering worker threads. A process crash can
        # now resume the same durable child Runs rather than creating duplicates.
        for task in ready:
            self._prepare_child(run, task)
        self._save_plan(run, plan)
        if run.status != RunStatus.RUNNING:
            return False
        self._event(
            run.run_id,
            "parallel_research_started",
            task_ids=[task.task_id for task in ready],
            parallel_count=len(ready),
        )
        with ThreadPoolExecutor(max_workers=min(self.max_agents, len(ready))) as executor:
            future_map = {
                executor.submit(copy_context().run, self._execute_task, run, task): task
                for task in ready
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    updated = future.result()
                    task.status, task.run_id = updated.status, updated.run_id
                    task.retry_count, task.error = updated.retry_count, updated.error
                except Exception as exc:
                    task.status = TaskStatus.FAILED
                    task.error = f"{type(exc).__name__}: {exc}"
                    self._event(
                        run.run_id,
                        "specialist_failed",
                        task_id=task.task_id,
                        agent_name=task.assigned_agent.value,
                        error_type=type(exc).__name__,
                    )
        self._save_plan(run, plan)
        return True

    @staticmethod
    def _citation_valid(item: dict[str, Any]) -> bool:
        if not str(item.get("content") or "").strip():
            return False
        source_type = item.get("source_type")
        citation = item.get("citation") or {}
        if source_type == "web":
            return bool(citation.get("url") or str(item.get("source", "")).startswith("http"))
        if source_type == "knowledge_base":
            return bool(citation.get("document_name") or citation.get("title") or item.get("source"))
        return bool(item.get("source"))

    def _deterministic_critic(
        self,
        goal: str,
        evidence: list[dict[str, Any]],
        tasks: list[ResearchTask],
    ) -> CriticResult:
        by_task = {task.task_id: [] for task in tasks}
        invalid: list[str] = []
        validated: list[str] = []
        for item in evidence:
            by_task.setdefault(str(item.get("task_id")), []).append(item)
            if self._citation_valid(item):
                validated.append(str(item["evidence_id"]))
            else:
                invalid.append(str(item["evidence_id"]))

        unsupported = [
            task.title
            for task in tasks
            if task.status == TaskStatus.COMPLETED and not by_task.get(task.task_id)
        ]
        missing = [
            ("local evidence" if task.assigned_agent == AgentRole.LOCAL_RESEARCH else "web evidence")
            for task in tasks
            if task.status == TaskStatus.FAILED or not by_task.get(task.task_id)
        ]
        claims: dict[str, tuple[str, str]] = {}
        conflicts: list[str] = []
        conflicting_ids: set[str] = set()
        for item in evidence:
            citation = item.get("citation") or {}
            claim = str(citation.get("claim") or "").strip()
            value = str(citation.get("claim_value") or "").strip()
            if claim and value:
                previous = claims.get(claim)
                if previous and previous[0] != value:
                    conflicts.append(
                        f"{claim}: {previous[0]} ({previous[1]}) vs {value} ({item['evidence_id']})"
                    )
                    conflicting_ids.update((previous[1], str(item["evidence_id"])))
                else:
                    claims[claim] = (value, str(item["evidence_id"]))
        validated = [item for item in validated if item not in conflicting_ids]
        valid = bool(validated) and not unsupported and not invalid and not conflicts and not missing
        return CriticResult(
            valid=valid,
            unsupported_claims=unsupported,
            invalid_citations=invalid,
            conflicts=conflicts,
            conflicting_evidence_ids=sorted(conflicting_ids),
            missing_topics=list(dict.fromkeys(missing)),
            need_more_research=bool(missing or unsupported or invalid) and not conflicts,
            validated_evidence_ids=validated,
        )

    @staticmethod
    def _deterministic_writer(
        run: AgentRun,
        evidence: list[dict[str, Any]],
        critic: CriticResult,
    ) -> str:
        if not evidence:
            return "现有证据不足，无法给出有来源支持的结论。"
        lines = ["基于已验证证据，研究结果如下："]
        for index, item in enumerate(evidence, start=1):
            content = str(item.get("content") or "").strip().replace("\n", " ")
            lines.append(f"- {content[:500]} [{index}]")
        if critic.conflicts:
            lines.append("\n证据冲突：" + "；".join(critic.conflicts))
        if not critic.valid:
            lines.append("\n限制：部分主题缺少充分证据，以上仅包含已验证内容。")
        return "\n".join(lines)

    @staticmethod
    def _as_writer_evidence(records: list[Any]) -> list[dict[str, Any]]:
        result = []
        for index, record in enumerate(records, start=1):
            item = record.model_dump(mode="json")
            item["evidence_id"] = record.evidence_id
            item["citation_index"] = index
            item.update(record.citation)
            result.append(item)
        return result

    def _write(self, run: AgentRun, evidence: list[dict[str, Any]], critic: CriticResult) -> str:
        self._handoff(
            run.run_id,
            AgentRole.CRITIC,
            AgentRole.WRITER,
            "Write the final answer",
            context={"critic": critic.model_dump(mode="json")},
            evidence_refs=[str(item["evidence_id"]) for item in evidence],
            reason="Only verified evidence may enter final synthesis",
        )
        self._event(run.run_id, "synthesis_started", evidence_count=len(evidence), agent_name=AgentRole.WRITER.value)
        if self.stream_writer is None:
            result = self.writer(run, evidence, critic)
        else:
            chunks: list[str] = []
            for chunk in self.stream_writer(run, evidence, critic):
                token = str(chunk)
                if token:
                    chunks.append(token)
                    self._event(run.run_id, "answer_token", token=token, agent_name=AgentRole.WRITER.value)
            result = "".join(chunks)
            if not result:
                raise ValueError("Multi-agent writer returned no tokens")
        self._event(run.run_id, "final_answer", content=result, agent_name=AgentRole.WRITER.value)
        return result

    def execute(self, run_id: str) -> AgentRun:
        run = self.store.get_run(run_id)
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return run
        if run.status in {RunStatus.PENDING, RunStatus.PAUSED}:
            run.transition(RunStatus.RUNNING)
        self.store.save_run(run)
        self._event(run_id, "multi_agent_started", agent_name=AgentRole.SUPERVISOR.value)
        started = time.monotonic()
        try:
            if not run.plan:
                with runtime_span("multi_agent.supervisor", run_id=run_id, agent_name=AgentRole.SUPERVISOR.value):
                    plan = self.supervisor.decompose(run.goal)
                run.budget.consume("llm")
                self._save_plan(run, plan)
                self._event(run_id, "task_decomposition_completed", task_count=len(plan.tasks))
            else:
                plan = self._plan_from_run(run)

            critic_result: CriticResult | None = None
            critic_round = int(run.metadata.get("critic_round", 0))
            while critic_round < self.max_critic_rounds:
                while self._run_ready_tasks(run, plan):
                    run = self.store.get_run(run_id)
                    if run.status != RunStatus.RUNNING:
                        return run
                    plan = self._plan_from_run(run)
                run = self.store.get_run(run_id)
                if run.status != RunStatus.RUNNING:
                    return run
                records = self.evidence_store.list(run_id)
                evidence = [record.model_dump(mode="json") for record in records]
                with runtime_span(
                    "multi_agent.critic",
                    run_id=run_id,
                    agent_name=AgentRole.CRITIC.value,
                    evidence_count=len(evidence),
                    critic_round=critic_round + 1,
                ):
                    raw = self.critic(run.goal, evidence, plan.tasks)
                    critic_result = raw if isinstance(raw, CriticResult) else CriticResult.model_validate(raw)
                critic_round += 1
                run = self.store.get_run(run_id)
                run.metadata["critic_round"] = critic_round
                run.metadata["critic_result"] = critic_result.model_dump(mode="json")
                self.evidence_store.set_verification(
                    run_id,
                    critic_result.validated_evidence_ids,
                    critic_result.conflicting_evidence_ids,
                )
                self._event(
                    run_id,
                    "critic_completed",
                    agent_name=AgentRole.CRITIC.value,
                    valid=critic_result.valid,
                    need_more_research=critic_result.need_more_research,
                    unsupported_claims=critic_result.unsupported_claims,
                    conflicts=critic_result.conflicts,
                    missing_topics=critic_result.missing_topics,
                    evidence_count=len(evidence),
                    critic_round=critic_round,
                )
                self.store.save_checkpoint(run)
                research_round = max((task.research_round for task in plan.tasks), default=0)
                if not critic_result.need_more_research:
                    break
                if research_round + 1 >= self.max_research_rounds or len(plan.tasks) >= self.max_subtasks:
                    break
                supplemental = self.supervisor.supplemental_tasks(
                    run.goal,
                    critic_result.missing_topics or critic_result.unsupported_claims,
                    research_round=research_round + 1,
                    remaining=self.max_subtasks - len(plan.tasks),
                )
                if not supplemental:
                    break
                plan.tasks.extend(supplemental)
                run.budget.consume("replan")
                run.replan_count += 1
                self._save_plan(run, plan)
                self._event(
                    run_id,
                    "supplemental_research_planned",
                    task_ids=[task.task_id for task in supplemental],
                    research_round=research_round + 1,
                )

            if critic_result is None:
                raise RuntimeError("Critic did not produce a verification result")
            validated = self._as_writer_evidence(self.evidence_store.list(run_id, validated_only=True))
            run = self.store.get_run(run_id)
            run.evidence = validated
            run.metadata["handoff_count"] = len(self.evidence_store.list_handoffs(run_id))
            run.metadata["agent_count"] = len({task.assigned_agent.value for task in plan.tasks}) + 3
            run.metadata["multi_agent_duration_seconds"] = round(time.monotonic() - started, 6)
            total_tool_calls = int(run.metadata.get("multi_agent_tool_calls", 0))
            run.metadata["total_tool_calls"] = total_tool_calls
            run.budget.tool_calls = min(total_tool_calls, run.budget.max_tool_calls)
            if validated or critic_result.conflicts:
                child_runs = [
                    self.store.get_run(task.run_id)
                    for task in plan.tasks if task.run_id
                ]
                writer_context = ContextBuilder().build(
                    query=run.goal,
                    goal=run.goal,
                    plan=run.plan,
                    evidence=validated,
                )
                run.context_usage = writer_context.token_usage
                run.budget.tokens = min(
                    run.budget.max_tokens,
                    sum(child.budget.tokens for child in child_runs)
                    + int(writer_context.token_usage["total"]),
                )
                run.budget.llm_calls = min(
                    run.budget.max_llm_calls,
                    run.budget.llm_calls + sum(child.budget.llm_calls for child in child_runs),
                )
                run.budget.consume("llm")
                with runtime_span(
                    "multi_agent.writer",
                    run_id=run_id,
                    agent_name=AgentRole.WRITER.value,
                    evidence_count=len(validated),
                ):
                    run.result = self._write(run, validated, critic_result)
                run.transition(RunStatus.COMPLETED)
            else:
                run.error = "No evidence passed Critic verification"
                run.error_type = "NoVerifiedEvidence"
                run.transition(RunStatus.FAILED)
        except Exception as exc:
            run = self.store.get_run(run_id)
            run.error = str(exc)
            run.error_type = type(exc).__name__
            if run.status == RunStatus.RUNNING:
                run.transition(RunStatus.FAILED)
        self.store.save_checkpoint(run)
        self._event(
            run_id,
            "run_completed",
            status=run.status.value,
            agent_name=AgentRole.SUPERVISOR.value,
            evidence_count=len(run.evidence),
            tool_calls=run.metadata.get("total_tool_calls", 0),
            retry_count=sum(task.retry_count for task in self._plan_from_run(run).tasks) if run.plan else 0,
        )
        record_runtime_event(
            "multi_agent.complete",
            run_id=run_id,
            status=run.status.value,
            agent_count=run.metadata.get("agent_count", 0),
            evidence_count=len(run.evidence),
            duration=run.metadata.get("multi_agent_duration_seconds", 0),
            tool_calls=run.metadata.get("total_tool_calls", 0),
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
        if run.status != RunStatus.PAUSED:
            raise ValueError(f"Run {run_id} is not resumable from {run.status.value}")
        checkpoint = self.store.load_checkpoint(run_id)
        restored = AgentRun.from_dict(checkpoint)
        restored.status = run.status
        restored.worker_id = run.worker_id
        restored.queued_at = run.queued_at
        restored.queue_time_ms = run.queue_time_ms
        restored.resume_count = run.resume_count
        restored.recovery_result = run.recovery_result
        restored.metadata.update(run.metadata)
        self.store.save_run(restored)
        return self.execute(run_id)

    def approve(self, approval_id: str) -> AgentRun:
        raise ValueError("Multi-agent parent Runs do not request direct tool approval")

    def reject(self, approval_id: str) -> AgentRun:
        raise ValueError("Multi-agent parent Runs do not request direct tool approval")


class RoutingRuntime:
    """One TaskManager-facing runtime that dispatches persisted Runs by mode."""

    def __init__(self, single_agent: AgentRuntime, multi_agent: MultiAgentRuntime):
        if single_agent.store.path != multi_agent.store.path:
            raise ValueError("Routing runtimes must share one RunStore")
        self.single_agent = single_agent
        self.multi_agent = multi_agent
        self.store = single_agent.store

    def create_run(self, conversation_id: str, goal: str, **kwargs: Any) -> AgentRun:
        return self.store.create_run(conversation_id, goal, **kwargs)

    def _runtime(self, run_id: str):
        run = self.store.get_run(run_id)
        return self.multi_agent if run.metadata.get("execution_mode") == "multi_agent" else self.single_agent

    def execute(self, run_id: str) -> AgentRun:
        return self._runtime(run_id).execute(run_id)

    def resume_run(self, run_id: str) -> AgentRun:
        return self._runtime(run_id).resume_run(run_id)

    def pause_run(self, run_id: str) -> AgentRun:
        return self._runtime(run_id).pause_run(run_id)

    def cancel_run(self, run_id: str) -> AgentRun:
        return self._runtime(run_id).cancel_run(run_id)

    def approve(self, approval_id: str) -> AgentRun:
        return self.single_agent.approve(approval_id)

    def reject(self, approval_id: str) -> AgentRun:
        return self.single_agent.reject(approval_id)
