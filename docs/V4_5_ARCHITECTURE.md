# LocalAgent V4.5 Runtime Architecture

## 后台执行

```text
UI request
  -> Create AgentRun(PENDING)
  -> append run_queued
  -> return run_id immediately

Background TaskManager
  -> worker_started(worker_id, queue_time)
  -> AgentRuntime
  -> checkpoints + run_events
  -> COMPLETED / FAILED / CANCELLED / WAITING_APPROVAL

UI fragment (1s)
  -> poll AgentRun
  -> poll run_events after sequence
  -> render Debug Events separately from answer tokens
```

SQLite 保存队列状态，ThreadPoolExecutor 只执行当前进程的工作。页面断开不会取消 Future。新应用进程启动时扫描 PENDING，以及带 `recovery_reason` 的 PAUSED Run。

## Crash / Resume

```text
worker crash
  -> Future reports BaseException
  -> persist PAUSED checkpoint
  -> worker_crashed event
  -> bounded recovery_queued
  -> reload latest checkpoint
  -> skip completed_steps / successful signatures
  -> resume unfinished step
```

`MAX_WORKER_RECOVERIES` 限制恢复循环。成功恢复写入 `resume_count` 与 `recovery_result=SUCCESS`。

## 流式协议

主要事件顺序：

```text
run_created -> run_queued -> worker_started
-> planning_started -> planning_completed
-> tool_call_started -> retry? -> tool_call_completed
-> approval_required?
-> synthesis_started -> answer_token* -> final_answer
-> run_completed -> worker_finished
```

所有事件有 SQLite sequence。Phoenix 同时记录 `worker_id`、`run_id`、`queue_time`、`resume_count`、`stream_event`、`idempotency_key`、`recovery_result`。

## MCP stdio

Transport 管理一个子进程和串行 JSON-RPC 请求：

1. `initialize`；
2. `notifications/initialized`；
3. `tools/list` 自动发现；
4. `tools/call`；
5. timeout 时终止失效子进程并安全失败。

自动注册读取 MCP `annotations.readOnlyHint`：只读 Tool 为低风险；写型 Tool 标记 `side_effecting` 并默认进入 Approval。

## 幂等边界

本地表记录 `STARTED/COMPLETED + output`。完成记录可跨 Run 复用。Retry 和 crash resume 始终向 Tool 传递相同 `idempotency_key`。本地数据库无法单方面保证远端系统事务，因此远端写 Tool 必须实现同 key 去重。
