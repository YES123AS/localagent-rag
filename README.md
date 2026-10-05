# LocalAgent V5

一个基于 Streamlit、LangGraph、Qdrant、SQLite 与 DeepSeek 的本地知识库研究助手。项目支持文档问答、联网搜索、安全计算、持久化会话、有界 Agent 执行，以及按需启用的 Multi-Agent Research。

> 隐私提示：文档解析、Embedding、重排和向量存储在本机完成；问题、必要的检索片段及搜索摘要会发送到所配置的模型或搜索 API。请根据自己的数据合规要求使用。

## 主要能力

- 上传 PDF、DOCX、Markdown 和 TXT，写入本地 Qdrant 知识库；
- Corrective RAG：召回、重排、相关性判断、查询改写和引用校验；
- 普通对话、知识库、联网搜索和计算器自动路由；
- 复杂任务使用受预算约束的 Planner、Tool、Evidence 和 Replan 流程；
- 跨本地资料与网络来源的任务可进入 Supervisor、Specialist、Critic、Writer 流程；
- SQLite 保存会话、运行状态、Checkpoint、Memory 和缓存；
- Phoenix / OpenTelemetry 提供本地链路追踪；
- 支持受限只读 Filesystem MCP，以及可选的单个 MCP stdio Server。

架构细节见 [V5 架构](docs/V5_ARCHITECTURE.md)、[V4.5 Runtime](docs/V4_5_ARCHITECTURE.md) 和 [V4 Runtime](docs/V4_ARCHITECTURE.md)。

## 项目结构

```text
.
├── src/
│   ├── app.py              # Streamlit UI 与 LangGraph 集成
│   ├── agent/              # Planner、Evidence、Policy 与执行循环
│   ├── knowledge/          # 文档入库、集合和向量生命周期
│   ├── memory/             # 会话与长期 Memory
│   ├── multi_agent/        # Supervisor、Specialist、Critic、Writer
│   ├── runtime/            # Run、Checkpoint、Budget、Cache 与 Worker
│   └── tools/              # 搜索、计算器和 MCP
├── sample_docs/            # 可公开提交的演示文档
├── runtime-data/           # 本机密钥与运行数据（默认不提交）
├── .env.example            # 环境变量模板
├── docker-compose.yml
└── Dockerfile
```

## 快速开始

要求：Docker Desktop 与 Docker Compose v2。

1. 创建本地配置：

```powershell
Copy-Item .env.example runtime-data/.env
```

Linux / macOS：

```bash
cp .env.example runtime-data/.env
```

2. 编辑 `runtime-data/.env`，至少填写 `DEEPSEEK_API_KEY`。如需联网搜索，再填写 `BOCHA_SEARCH_API_KEY`。

3. 构建并启动：

```bash
docker compose up -d --build
```

4. 打开服务：

| 服务 | 地址 |
| --- | --- |
| LocalAgent | <http://localhost:8501> |
| Phoenix | <http://localhost:6006> |
| Qdrant | <http://localhost:7333> |

查看日志或停止服务：

```bash
docker compose logs -f app
docker compose down
```

## 配置

完整配置和安全占位符见 [.env.example](.env.example)。常用变量如下：

| 变量 | 是否必填 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | 否 | OpenAI-compatible API 地址 |
| `DEEPSEEK_MODEL` | 否 | 对话模型名称 |
| `BOCHA_SEARCH_API_KEY` | 否 | 启用博查联网搜索 |
| `EMBEDDING_MODEL` | 否 | 本地 Hugging Face Embedding 模型 |
| `QDRANT_COLLECTION` | 否 | 默认知识库集合名 |
| `MCP_FILESYSTEM_ROOT` | 否 | 启用受限只读文件系统 MCP |
| `MCP_STDIO_COMMAND_JSON` | 否 | MCP stdio Server 命令数组 |

## 本地数据与安全

所有敏感信息和运行状态统一位于 `runtime-data/`：

```text
runtime-data/
├── .env          # API Key，不提交
├── app/          # SQLite 会话、Run 与缓存
├── qdrant/       # 向量库
├── phoenix/      # Trace 数据
└── model-cache/  # 本地模型缓存
```

`.gitignore` 与 `.dockerignore` 已排除这些内容。提交前建议执行：

```bash
git status --short
git ls-files | grep -E '(^|/)\.env$|runtime-data/'
```

正常情况下只会看到可公开的 `runtime-data/README.md`，不会看到 `.env` 或运行数据。

## 上传到 GitHub

先在 GitHub 创建一个空仓库（不要勾选自动生成 README、License 或 `.gitignore`），然后在项目目录执行：

```bash
git init
git branch -M main
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub 邮箱"
git add .
git commit -m "Initial public release"
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

如果 GitHub 邮箱需要隐藏，请先在 GitHub 的 Email 设置中启用隐私邮箱，再修改本仓库的 `user.email` 后提交。

## 已知边界

- 当前以单实例 SQLite 和进程内 Worker 为核心，不是分布式多租户平台；
- Qdrant 集合维度由 Embedding 模型决定，切换模型后应重建集合；
- Citation 校验保证编号对应 Evidence，但不等同于逐句事实蕴含证明；
- 联网搜索未配置或不可用时会有限重试并明确降级；
- 首次启动需要下载镜像和本地 Embedding 模型，耗时取决于网络环境。

## License

当前仓库尚未附带开源许可证。公开发布前请根据使用目的选择并添加合适的 `LICENSE`。
