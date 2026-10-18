# 架构文档（ARCHITECTURE）

本文档说明本地化 ReAct Agent 系统的内部结构与关键设计决策，面向想深入代码的读者。

- [系统总览](#系统总览)
- [请求生命周期](#请求生命周期)
- [多范式编排引擎](#多范式编排引擎)
- [RAG 检索管线](#rag-检索管线)
- [工具层](#工具层)
- [记忆与上下文压缩](#记忆与上下文压缩)
- [护栏（Guard Rails）](#护栏guard-rails)
- [评测方法论](#评测方法论)
- [关键设计决策](#关键设计决策)

---

## 系统总览

系统由四层组成：

1. **接入层**：FastAPI 路由（`chat` / `auth` / `knowledge` / `agent` / `upload`）+ 单文件前端（`/front` 静态挂载）。
2. **编排层（Agent）**：`SmartAgent` 默认编排引擎，按 ReAct 范式在「思考—行动—观察」循环里调度工具；通过 `AGENT_MODE` 可整体替换为原生 FC / LangGraph 实现。
3. **能力层（工具 + RAG）**：工具注册表（`ToolRegistry`）+ 检索服务（向量 / BM25 / 混合 / HyDE + reranker）。
4. **模型与存储**：本地或云端 LLM、bge 嵌入 / 重排、Chroma 向量库、MySQL 元数据、Redis 缓存（可选）。

---

## 请求生命周期

```
用户消息
  → Chat Service（smart_chat_service / agent_chat_service）
  → SmartAgent.run()
       loop (max MAX_REACT_ROUNDS=8):
         1. 组装 System + 历史 + 工具描述，调用 LLM
         2. 解析 LLM 输出：正则抽取 <thought>/<action>/<action_input>
         3. 执行 Action → 工具层 → 观测结果
         4. 护栏检查（心算闸 / 文件验证闸 / 重复调用闸）
         5. 将 Observation 回填，进入下一轮
  → 达到终止条件（Final Answer / 轮数上限）→ 返回答案（SSE 流式）
```

- **流式**：Agent 每轮思考 / 工具调用 / 观测通过 SSE 推送到前端。
- **上下文压缩**：超过阈值时，早期轮次被压缩为摘要（`context/compressor.py`）。
- **记忆分层**：跨会话 = `memory/long_term.py`（MySQL 存文本 + Chroma 向量召回用户事实/偏好）；会话内多轮 = DB 历史，超阈值（>8 轮或 token 超限）由 `context/compressor.py` 压缩为摘要；单轮内 = `AgentState` 基于 `_tool_results` 确定性重算。

---

## 多范式编排引擎

**同一套工具层 + 同一套护栏，之上并排三种编排实现。**

### 三臂与两正交轴

| 范式 | 机制轴 | 循环引擎轴 | 文件 |
|---|---|---|---|
| ReAct | Prompt + 正则解析文本 | 手写 `while` | `agent.py` |
| 原生 Function Calling（FC） | 原生 `tool_calls` | 手写 `while` | `agent_fc.py` |
| LangGraph（LG） | 原生 `tool_calls` | `StateGraph` | `langgraph_agent.py` |

- **ReAct vs FC = 单变量（机制轴）**：引擎同为手写 `while`，唯一区别是「从文本正则抽工具」还是「用原生 `tool_calls`」。
- **FC vs LG = 单变量（引擎轴）**：机制同为原生 FC，唯一区别是「手写 while」还是「LangGraph StateGraph」。
- **ReAct vs LG = 双变量**：两端点，用于验证「文本兜底补齐后，原生 FC 路径也能逼近 ReAct」。

`AGENT_MODE` 在 `run_eval.py` 的 `build_eval_agent()` 中切换返回的类；生产 `chat` 路径默认实例化 `SmartAgent`（ReAct）。

### 为什么不用 LangChain `AgentExecutor`

三者均**未使用** `AgentExecutor`（grep 全空）。LangChain 只作为 LLM 客户端（`ChatOpenAI` + `StructuredTool` / `bind_tools`）。保留自研编排层的原因：

1. **可控性**：手写循环可精确插入护栏、上下文压缩、stop 序列轮换。
2. **可观测性**：每轮 `Thought/Action/Observation` 与 `_tool_calls_detail` 全链路透传。
3. **小模型可靠性**：7B 在原生 FC 下易泄漏工具调用模板，自研文本兜底更稳。
4. **评测驱动**：自研层让「机制 / 引擎」两轴解耦，可做干净对照。

> 若改用 `AgentExecutor`，自研编排层的护栏 / 压缩 / stop 序列轮换这些插入点都会被框架封装掉；且 LangGraph 臂若改 `AgentExecutor` 就不再是 LangGraph。

### 7B 工具调用泄漏与文本兜底

- **现象指纹**：7B 把 chat-template 的序列化**原样回吐**——`<tool_call>{{"name":..,"arguments":{..}}}</tool_call>`，其中 `{{}}` 双花括号是 Jinja2 模板定界符被 regurgitate 的铁证；少量为 `knowledge_search({...})` 形式。
- **根因**：这是**生成阶段失败**——7B 没产出有效 `tool_calls` 槽位，导致 `resp.tool_calls` 为空、模板字面文本落进 `resp.content`，而非反序列化失败。
- **模板归属**：`<tool_call>` 来自 **Qwen2.5 官方 chat-template（GGUF 内置）+ llama.cpp 渲染**，非本项目手写；走 `localhost:8080/v1` OpenAI 兼容接口 + `ChatOpenAI.bind_tools`，无自定义模板。故**无法靠改模板修复**，只能靠：
  1. 文本重建兜底（`agent_fc.py` 的 `_extract_tool_calls_from_text`，FC/LG 共用；含 `{{}}`→`{}` 归一化再 `json.loads`）；
  2. 保留不依赖 `tool_calls` 的 ReAct 路径从根绕过 7B 协议脆弱点（ReAct 工具准确率 100% 之因）。
- **round-2 兜底形态**：① `tool_name({...})` 带圆括号；② `{"name":..,"arguments":{..}}` 模板 JSON（`{{}}`→`{}` 归一化后匹配）。

### 对照实验结论（受控变量）

同机·同日·同代码复测（10-06）：**FC 93.43 > ReAct 92.86 > LG 92.00**，极差 1.43pt ≈ 1 条用例。相对旧基线（ReAct 曾领先 FC 4pt）发生**排名翻转**——这说明差异是7B模型采样噪声而非架构的差异。

- ReAct 的真实残余优势 = **工具准确率 100% 两轮零波动** + 无需兜底补偿（文本形态是其原生输入）。
- 残留弱项（跨臂共性，与编排无关）：`hallucination` 类是全臂噪声热点（单跑互有涨跌）；FC/LG 原生 FC 路径对部分需要检索的问题存在过度拒答倾向，ReAct 路径无此问题。

---

## RAG 检索管线

`knowledge_search` 工具背后的检索服务支持：

- **检索策略**：`vector` / `bm25` / `hybrid`（RRF 融合）/ `hyde`（假设性文档嵌入）。
- **重排**：bge-reranker-v2-m3，对初召回做精排。
- **Query 改写**：与策略正交的增强层（`none` / `hyde` / `expand` / `decompose`），默认关闭。
- **切分**：默认 chunk 500（注意：`.env` 的 `CHUNK_SIZE` 为死配置，真实切分在 `attachment_service.py`）；top_k 默认 5（真实取自 `tool_configs["knowledge_search"]["top_k"]`）。
- **可调旋钮**：chunk 大小 / overlap、top_k、reranker 阈值、query rewrite 策略、是否启用 HyDE / 混合检索（检索质量调的是这些接入点，不是向量相似度公式本身）。

---

## 工具层

- **`registry.py`**：内存工具注册表，`init_default_tools()` 在启动时注册内置工具。
- **`base.py`**：工具基类，`to_langchain_tool()` 返回 LangChain `StructuredTool`（供任意编排复用）。
- **内置工具**：
  - `knowledge_search_tool`：检索知识库
  - `calculator_tool`：数值计算（支持 `^`→`**`）
  - `file_reader_tool`：读本地文件（白名单目录）
  - `oa_mock_tool`：`oa_reimbursement` Mock 连接器（真实环境需替换）
- **`mcp_adapter.py`**：通用 MCP 适配器——遍历 `ToolRegistry`，按 `parameters_schema` 动态构建函数签名并注册到同一 FastMCP server（stdio），从而通过 MCP 协议接入外部工具。`mcp_client_demo.py` 演示完整 initialize → tools/list → tools/call 流程。
  - 适配层三个坑（已解决）：① Windows 子进程 stdout GBK → 须 `PYTHONUTF8=1`；② loguru 打 stdout 污染 JSONRPC → server main 移除 loguru 挂 stderr；③ FastMCP 对非 str 注解预解析 JSON → `params` 须裸 `str`。

---

## 记忆与上下文压缩

- **跨会话长期记忆**：`memory/long_term.py`——MySQL 存记忆文本/类型/重要度/时间戳，Chroma 向量索引做语义召回（`memorize` 双写 / `recall` 向量召回后回 MySQL 取详情 / `forget` 清理过期低重要度记忆）。
- **会话内多轮历史**：DB 持久化对话历史；轮次或 token 超阈值时由 `context/compressor.py` 压缩为摘要。
- **单轮内状态**：`AgentState`（`agent_state.py`）基于 `_tool_results` **确定性重算**，刻意不同于 LangGraph reducer——保证可复现、可观测。
- **上下文压缩**：`context/compressor.py` 按 provider 配置（如 qwen：max_context_tokens 32768、keep_full_steps 2），达阈值时把早期轨迹压缩为摘要。

---

## 护栏（Guard Rails）

编排层内置多道兜底闸，防止幻觉与工具误用：

- **心算闸**：要求数值类问题必须调用 `calculator`，否则拦截 Final Answer 重跑一轮。
  - 判定逻辑（`agent.py` `_should_enforce_calculator`）：正则从答案抽数字，若答案含数值/百分比但本轮 calculator 未成功返回过（`_calculator_returned_number` 为假），则拦截；calculator 调用失败仍触发重跑。
  - 残留风险：重跑次数耗尽后强制总结（`force_final`）时闸会放行，极端情况下仍可能输出未校验数值，是已知边界。
- **文件验证闸**：`file_reader` 返回后校验内容一致性。
- **重复调用闸**：检测并阻断对相同参数的重复工具调用。
- **`_normalize_answer`**：答案归一化，去除复述、内部字段泄漏。

---

## 评测方法论

- **端到端** `run_eval.py`：70 条用例 / 9 类，输出综合分、答案匹配率、工具正确率、幻觉率；支持按类型过滤、断点续跑（checkpoint 按 `AGENT_MODE` + 压缩策略区分）。
- **检索层** `run_rag_eval.py`：多策略 / 多阈值对比，输出 P@k / R@k / F1 / MRR / NDCG（可选 Ragas LLM-judge）。
- **口径**：工具正确率按「期望 ⊆ 实际」判定；答案匹配对 `no_answer` / `hallucination` 类须扩完整拒答词集，避免 "any" 误判。
- **受控对照**：同机·同日·同代码只变 `AGENT_MODE`，确保分数差可归因于编排而非环境噪声。

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 编排框架 | 自研 ReAct 循环（非 AgentExecutor） | 可控 / 可观测 / 对小模型稳 / 可做干净对照 |
| 多范式 | ReAct + FC + LangGraph 并存 | 同一评测集 A/B 对照，tradeoff 清晰可量化 |
| LLM provider | 默认 `local`（llama.cpp） | 私有化、数据不出网、零 API 成本 |
| MCP | 自写通用适配器 | 协议标准化、工具解耦、面向 MCP 生态 |
| 可观测性 | 全链路 `_tool_calls_detail` + 评测体系 | 每轮 Thought/Action/Observation 与工具结果可追溯、可回放 |
| 评测 | 自建而非接 LangSmith | 与私有化部署目标一致，避免评测数据外发云端 |
