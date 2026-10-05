# LocalAgent V5 Multi-Agent Research Architecture

## Admission and execution path

V5 does not send every request through multiple agents. `should_use_multi_agent` requires three signals at the same time: local-document evidence, current web evidence, and an explicit combine/compare intent.

```text
Simple / single-source request
  -> existing LangGraph or V4.5 single Agent Runtime

Multi-source research request
  -> Create persisted parent AgentRun -> Background Worker
  -> Supervisor -> bounded ResearchTask DAG
  -> Local Research Agent  --+
  -> Web Research Agent   ----+-> Shared Evidence Store
  -> Critic -> optional bounded supplemental round -> Writer
  -> token stream + final answer
```

`RoutingRuntime` is the only runtime exposed to `TaskManager`. It reads `metadata.execution_mode` and dispatches to the existing `AgentRuntime` or `MultiAgentRuntime`. This preserves the V4.5 queue, polling, cancellation, crash recovery, checkpoint, and event-stream behavior.

## Supervisor and task model

The Supervisor only admits, decomposes, assigns, and tracks work. It does not research. A `ResearchTask` contains `task_id`, query, assigned role, `depends_on`, status, child `run_id`, retry count, error, and research round. Each child is a normal durable `AgentRun`, not an untracked future.

Independent ready tasks execute in a bounded `ThreadPoolExecutor`. A dependent task is scheduled only after every dependency is complete. A specialist failure is recorded on that task and does not raise through the whole research run.

The bounds are:

- `MAX_AGENTS`: maximum concurrent specialist workers;
- `MAX_SUBTASKS`: maximum tasks across all rounds;
- `MAX_RESEARCH_ROUNDS`: initial plus supplemental research rounds;
- `MAX_CRITIC_ROUNDS`: maximum verification passes;
- `MAX_TOTAL_TOOL_CALLS`: shared upper bound across child Runs.

Agent roles are fixed: Supervisor, Local Research, Web Research, Critic, Writer. No agent may create new roles.

## Tool isolation

`ToolRegistry.scoped()` creates an allow-listed registry for each specialist:

| Role | Allowed tools |
| --- | --- |
| Local Research | `knowledge_base` |
| Web Research | `web_search`, configured `mcp.*` |
| Critic | none |
| Writer | none |

The allowed tools continue through the existing Policy Engine, Budget, approval metadata, retry policy, timeout, idempotency, and tracing path. The Writer receives verified evidence objects only and cannot search.

## Handoff contract

Handoffs are structured records in `agent_handoffs`, containing `from_agent`, `to_agent`, task, compact context, `evidence_refs`, reason, status, and parent Run. The runtime passes evidence IDs rather than copying full evidence text between agents.

Expected path:

```text
Supervisor -> Specialist: task/query/dependencies
Specialist -> Critic: child run status + evidence IDs
Critic -> Writer: structured verification + validated evidence IDs
```

Every handoff also emits `agent_handoff` into the persisted Run event stream and an OpenTelemetry span.

## Shared Evidence Store

`shared_evidence` is stored in the same SQLite database as Agent Runs. Each record keeps:

- `evidence_id`, parent Run, source type and source;
- content, producing agent and task;
- citation metadata and score;
- timestamp, fingerprint, validation and conflict flags.

The `(parent_run_id, fingerprint)` constraint deduplicates identical evidence. Writer input is loaded with `validated_only=True`.

## Critic verification

The default Critic is deterministic rather than a second unconstrained answer-generation call. It checks:

- completed task without evidence -> unsupported;
- failed/missing source task -> missing topic;
- web evidence without URL or local evidence without document metadata -> invalid citation;
- explicit same-claim/different-value metadata -> conflict;
- only citation-valid, non-conflicting IDs -> validated evidence.

It returns `CriticResult` with `valid`, `unsupported_claims`, `invalid_citations`, `conflicts`, `missing_topics`, `need_more_research`, and `validated_evidence_ids`. Missing evidence can create supplemental tasks, but only within both research and critic round limits. Conflicts are reported rather than hidden.

## Writer guarantee

The Writer prompt is constructed only from `SharedEvidenceStore.list(validated_only=True)`. It is instructed to cite numbered evidence, add no model-memory facts, and expose conflicts or gaps. It owns no external tools. Its output uses the existing persisted `answer_token` and `final_answer` event types, so debug events remain separate from answer text.

## Observability

The parent trace records `multi_agent.supervisor`, one `multi_agent.specialist` span per child, `multi_agent.critic`, Writer synthesis events, and `multi_agent.complete`. Attributes/events include agent name, parent/child Run ID, task ID, handoff, evidence count/reuse, duration, retry count, tool calls, status, critic round, and research round.

## Current boundary

V5 remains a single-process, SQLite-backed system. Independent child tasks are parallel threads, not distributed jobs. The default Critic performs structural citation/sufficiency/conflict validation, not a formal natural-language entailment proof. The only specialist roles are local and web research; MCP is available only inside the Web Research scope. No Swarm, GraphRAG, browser automation, Kafka, Kubernetes, or dynamic agent creation is included.
