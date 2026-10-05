# LocalAgent V4 Runtime Architecture

## 设计目标

V4 在原 LangGraph 上增加可靠的单 Agent Runtime，而不是重写 Corrective RAG 或引入 Multi-Agent。Conversation 与 Run 共用 SQLite 文件以简化部署，但使用独立表和生命周期。

## 组件图

```text
                       +---------------- ConversationStore
                       |
User -> UI/API -> Complexity Boundary
                       |
              +--------+--------+
              |                 |
          simple             complex
              |                 |
       Existing LangGraph   AgentRuntime
                                  |
                  +---------------+----------------+
                  |               |                |
                Planner       RunStore         ContextBuilder
                                  |                |
                           Checkpoint/Approval   Token Budget
                                  |
                             Policy Engine
                                  |
                  +---------------+----------------+
                  |               |                |
             Native Tools      MCP Adapter     ExecutionBudget
                  |               |
                  +------ Observation/Evidence ------+
                                      |
                                  Synthesis
```

## Run 生命周期

```text
PENDING -> RUNNING -> COMPLETED
              |  \
              |   +-> FAILED
              |   +-> CANCELLED
              |   +-> PAUSED -> RUNNING
              +-> WAITING_APPROVAL
                        | approve -> RUNNING
                        | reject  -> RUNNING（原操作标记 rejected）
                        + cancel  -> CANCELLED
```

Terminal 状态不可重新打开。Cancel 不删除 Evidence、Tool Call、Trace 或 Checkpoint。

## Checkpoint 与恢复序列

```text
Plan -> save checkpoint
  |
next unfinished step
  |
Policy -> save pending step -> persist in-flight call
  |
Tool -> Observation -> save checkpoint
  |
reload lifecycle status -> continue / pause / cancel
```

恢复读取最新 Checkpoint，并以 Run 主表的 Pause/Approval 状态为控制权威。`completed_steps` 和成功调用 signature 防止恢复后重复执行已完成工具。进程重启时，单实例启动钩子把遗留 `RUNNING` 标记为 `PAUSED`。

## SQLite 表

- `conversations`, `messages`：V3 Conversation Memory；
- `agent_runs`：Run 当前快照；
- `run_checkpoints`：关键状态的追加快照；
- `run_approvals`：PENDING/APPROVED/REJECTED 决策；
- `long_term_memories`：结构化、去重的长期 Memory；
- `runtime_cache`：有 TTL 的 Embedding/Web 结果。

## Policy 和 HITL

Tool Registry 的 Native/MCP 工具都携带风险、审批、权限、超时、调用上限和重试策略。Policy 的拒绝或预算耗尽不会调用工具。批准只对对应 Run Step 生效；每个后续高风险 Step 都需要独立批准。

## 一致性边界

SQLite 状态更新是事务性的。任意外部系统的 exactly-once 无法由本地数据库单方面保证，因此恢复依赖成功 signature 去重；未来写型工具还必须把 `tool_call_id` 作为 idempotency key 传给外部系统。V4 自带 MCP 验证工具只有只读操作。

## 部署边界

Docker Compose 继续运行单个 Streamlit app、Qdrant 和 Phoenix。没有 Redis/Celery/Kafka/Kubernetes。`TaskManager` 提供同进程后台 worker 原语，但当前 Chat UI 仍同步等待最终结果；进程退出后的继续执行需要用户显式 Resume。
