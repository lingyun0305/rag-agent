# 部署文档（DEPLOY）

本文档说明如何把本地化 ReAct Agent 系统跑起来，覆盖 Docker 一键与本地手动两种方式，以及 `.env` 配置、生产化建议与运维。

- [环境要求](#环境要求)
- [方式 A：docker-compose（推荐）](#方式-adocker-compose推荐)
- [方式 B：本地手动](#方式-b本地手动)
- [.env 配置项说明](#env-配置项说明)
- [生产化建议](#生产化建议)
- [运维](#运维)

---

## 环境要求

| 组件 | 版本 | 必需 | 说明 |
|---|---|---|---|
| Python | 3.11+ | ✅ | 建议虚拟环境 `venv/` |
| llama.cpp | 最新 | ✅（默认 local） | 提供 OpenAI 兼容接口，监听 `:8080` |
| MySQL | 8.0 | ✅ | 元数据 / 用户 / 工具注册，**无 SQLite 兜底** |
| Redis | 7 | ⬜ 可选 | 缓存；缺失自动降级内存缓存 |
| Docker | 20.10+ | ⬜ 可选 | 方式 A 需要 |

---

## 方式 A：docker-compose（推荐）

`docker-compose.yml` 已编排 `api` + `mysql` + `redis` 三个服务，`api` 构建自仓库根 `Dockerfile`。

```bash
# 1. 配置
cp .env.example .env
# 填 SECRET_KEY 即可；MYSQL_* 已在 docker-compose.yml 的 api.environment 里写死
# （容器内 mysql 用户为 rag_agent / rag_agent_pw，host 为服务名 mysql），无需在 .env 里改

# 2. 一键启动（后台）
docker-compose up -d

# 3. 查看日志
docker-compose logs -f api

# 4. 健康检查
curl http://localhost:8000/health

# 5. 打开前端
#    http://localhost:8000/front/index.html
```

> 注意：`docker-compose.yml` 中 `api` 挂载了 `./backend`、`./front` 等目录（bind mount），本地改代码无需重建镜像；首次或依赖变更时才需要 `docker-compose build api`。

---

## 方式 B：本地手动

### 1. Python 虚拟环境

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt        # Windows
# venv/bin/pip install -r requirements.txt                        # Linux/macOS
```

### 2. 本地大模型（llama.cpp）

项目默认 `DEFAULT_LLM_PROVIDER=local`，LLM 走 `http://localhost:8080/v1`。

```bash
# 假设已编译好 llama.cpp 的 server，模型为 qwen2.5-7b-q5_k_m.gguf
./llama-server -m qwen2.5-7b-q5_k_m.gguf \
    -ngl 99 \                # 尽量把层放到 GPU（无 GPU 可省略）
    --host 0.0.0.0 --port 8080 \
    --cache-prompt \          # 开启前缀缓存：跨请求复用共享前缀的 KV，省去重复 prefill（加速多轮 Agent）。不同构建 flag 名可能不同，用 `llama-server --help` 核对
    --alias qwen2.5-7b

# 验证模型已加载：
curl http://localhost:8080/props
```

> 模型量化建议：`q5_k_m` 在 7B 上兼顾精度与显存。纯 CPU 也能跑，但首 token 延迟较高。
> 若改用云端模型，把 `.env` 的 `DEFAULT_LLM_PROVIDER` 改为 `qwen` / `doubao` / `zhipu` 并填对应 API key 即可，无需 llama.cpp。

### 3. MySQL 与 Redis

```bash
# 方式一：docker
docker run -d --name rag-mysql -e MYSQL_ROOT_PASSWORD=<pwd> -e MYSQL_DATABASE=rag_agent_db -p 3306:3306 mysql:8.0
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine

# 方式二：本机已装则直接启动服务，确保 .env 中 HOST/PORT 对应
```

`create_tables()` 在服务启动事件里自动建表，无需手动迁移。

### 4. 启动后端（重要）

```bash
# ★ 必须从【仓库根目录】运行，module 路径为 backend.main:app
#   （main.py 用相对路径挂载 /front，cd 进 backend/ 会找不到 front/ 而报错）
venv/Scripts/python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5. 灌入知识库

首次使用在浏览器打开 `http://localhost:8000/front/knowledge.html`（或工作台「知识库」），上传 `knowledge_files/` 下的文档，系统会自动切分并写入 Chroma。也可直接把文件放进 `knowledge_files/` 后通过上传流程入库。

---

## .env 配置项说明

复制 `.env.example` 为 `.env` 后，逐项填写：

| 分组 | 变量 | 说明 |
|---|---|---|
| LLM 选择 | `DEFAULT_LLM_PROVIDER` | `local` / `qwen` / `doubao` / `zhipu` |
| 本地 LLM | `LOCAL_MODEL_BASE_URL` / `LOCAL_MODEL_NAME` / `LOCAL_MODEL_API_KEY` | 默认 `http://localhost:8080/v1`、`qwen2.5-7b`、`EMPTY` |
| 云端 LLM | `DASHSCOPE_*` / `ARK_*` / `ZHIPU_*` | 仅当 provider 切到对应值时需要 |
| 向量库 | `VECTOR_STORE_TYPE` / `VECTOR_STORE_PATH` / `CHROMA_COLLECTION_NAME` | 默认 chroma / `./chroma_db` |
| 嵌入 | `HUGGINGFACEEMBEDDINGS_MODEL_NAME` / `_DEVICE` | 默认 bge-large-zh-v1.5 / cpu |
| 重排 | `RERANK_MODEL_NAME` / `_DEVICE` | 默认 bge-reranker-v2-m3 / cpu |
| RAG | `ENABLE_HYDE` / `BASE_RECALL_TOK_K` | 查询增强开关 / 基础召回量 |
| 文件 | `FILE_STORAGE_PATH` / `UPLOAD_STAGING_DIR` / `MAX_*` | 知识库与上传目录、大小限制 |
| MySQL | `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DATABASE` | **必需** |
| JWT | `SECRET_KEY` / `ALGORITHM` / `ACCESS_TOKEN_EXPIRE_MINUTES` | `SECRET_KEY` 用 `python -c "import secrets;print(secrets.token_hex(32))"` 生成 |
| Redis | `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` / `REDIS_URL` | 可选；留空 localhost:6379，连不上自动降级 |
| HF 镜像 | `HF_ENDPOINT` | 国内拉模型用 `https://hf-mirror.com` |

> ⚠️ `.env` 的 `CHUNK_SIZE` / `RETRIEVE_TOP_K` 是**死配置**（无代码读取）；真实切分常量在 `attachment_service.py`（默认 500），真实 top_k 取自 `tool_configs["knowledge_search"]["top_k"]`（默认 5）。

---

## 生产化建议

- **反向代理**：用 Nginx / Caddy 前置，终止 HTTPS、做限流（应用层已有限流中间件 `30/minute`）。
- **资源**：llama.cpp 建议独立 GPU 机器；API 服务无状态，可多副本横向扩展（注意 MySQL / Chroma 为共享状态）。
- **私有化场景**：保持 `local` provider + 本地 Chroma，确保知识库数据不出网；`oa_reimbursement` Mock 需替换为真实 OA 连接器。
- **可观测性**：`/metrics` 暴露缓存统计；日志用 loguru，建议接入集中式日志。
- **故障隔离**：工具调用建议运行在独立沙箱 / 容器；当前 `oa_mock` 为 Mock，真实外部工具接入走 MCP 适配器。

---

## 运维

- **健康检查**：`GET /health` → `{"status":"healthy",...}`，供 Docker / 负载均衡探活。
- **指标**：`GET /metrics` → 缓存命中 / 统计（依赖 Redis，缺失时降级）。
- **API 文档**：`/docs`（Swagger）、`/redoc`。
- **日志**：启动事件会打印 `app_startup_done`；工具初始化失败会有 warning 但不阻断启动。
- **评测**：见根 README「评测体系」，`run_eval.py` 支持断点续跑。
