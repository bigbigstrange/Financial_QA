# 金融文档问答系统（DocIR + 双引擎检索 + Document Agent）

本项目是你当前升级后的主版本，核心目标是：
- 统一文档解析中间层（DocIR）
- 双引擎检索（文本 + 图像）并融合召回
- 文档智能体处理复杂问题
- 输出可追溯证据与可观测指标

## 1. 当前架构

```text
文档输入
  -> preprocess.py
      -> Parsers（PlainText / Word / PDF+表格+Tesseract 回退 / Image）
      -> DocIR（.docir.json + .md）
      -> knowledge_base.json + combined_knowledge_base.md + parser_report.json

knowledge_base.json
  -> FinancialQASystem（src/system）
      -> TextRetriever（BGE + FAISS）
      -> VisionPageRetriever（ColPali 或 CLIP，失败回落文本代理）
      -> FusionRetriever（RRF） + 可选 Neo4jGraphRetriever
      -> HybridRetriever（稠密 + 图谱）
      -> DocumentAgent（QueryPlanner + TableTool + Rerank + Generate）
      -> output/qa_results.json + output/qa_results.md（运行 main.py）

一键批量
  -> main.py（加载 vector_cache，不强制重建索引）
```

## 1.1 企业平台层（AI 全栈落地）

在原有 RAG/Agent 能力之上，补齐可上线工程能力：

| 能力 | 说明 |
|------|------|
| 鉴权与会话 | JWT 登录、会话/消息落库、答案反馈 |
| 内存防护 | warn/soft/hard 水位、推理并发闸门、生成模型空闲卸载 |
| 模型迭代 | `configs/model_registry.json` profile、晋级/回滚审计 |
| 持续评测 | EvalRun 落库 + `scripts/eval_gate.py` 门禁（CI 可用） |
| 前端 | Vue3 问答台 + 运维看板（`frontend/`） |
| 部署 | `docker compose up financial-qa-api`（含 mem_limit） |

### 启动企业平台

```bash
pip install -r requirements.txt
# 终端 1：API
python -m app.main
# 终端 2：前端（开发）
cd frontend && npm install && npm run dev
```

- API/Docs: http://127.0.0.1:8000/docs  
- 前端: http://127.0.0.1:5173  
- 默认管理员: `admin` / `admin123`

### 模型迭代门禁（上线前）

```bash
# 对候选 profile 跑黄金集，失败则 exit 2
python scripts/eval_gate.py --profile candidate-v2 --max-samples 20
# 通过后晋级
python scripts/eval_gate.py --profile candidate-v2 --max-samples 20 --promote
```

内存与 OOM 策略要点：
1. 演示模式默认截断知识块、跳过图像塔（`FINANCE_WEB_DEMO=1`）
2. `MemoryGuard`：系统内存 ≥ soft 拒加载新模型，≥ hard 拒推理
3. `MAX_CONCURRENT_INFER=1` 防止请求叠载爆内存
4. 生成模型空闲 `GEN_IDLE_UNLOAD_SEC` 后自动卸载
5. Compose `mem_limit` 与应用水位双保险

## 2. 快速开始

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 文档预处理（生成知识库 + 解析报告）

```bash
python scripts/preprocess.py --input ./data/documents --output ./output
```

输出关键文件：
- `output/knowledge_base.json`
- `output/combined_knowledge_base.md`
- `output/parser_report.json`

### 2.3 构建索引

```bash
python scripts/build_index.py --mmd-path ./output/knowledge_base.json
```

（默认使用已有 `vector_cache`；仅当知识库或嵌入模型变更后需要加 `--force-rebuild`。）

### 2.3.1 离线构建知识图谱三元组（新增）

```bash
python scripts/build_kg.py --kb-path ./output/knowledge_base.json --output-dir ./output
```

### 2.4 批量生成全部问题答案（主入口）

将问题表放在 `test/merged.xlsx`（列：`question_id`、`question`），并确保已完成索引构建、`vector_cache` 与当前知识库一致后：

```bash
python main.py
```

生成 `output/qa_results.json` 与 `output/qa_results.md`。

环境变量示例见 `code/.env.example`（可选，用于 docker-compose 中的图像后端等）。

### 2.4.1 Web 前端提问（交互演示）

需已有 `output/knowledge_base.json`。在 `code/` 目录：

```bash
python web_app.py
```

或 PowerShell：`.\scripts\run_web.ps1`

浏览器打开 **http://127.0.0.1:7860**。

**默认「演示精简模式」**（解决本机内存不足）：只加载与示例题相关的文档子集（约数千块）、跳过图像检索塔、关闭图谱，分批写入 FAISS，避免一次性分配约 1.8GB+ 向量矩阵。右上角显示「就绪·演示库」后再提问。

全量索引（需较大内存）：

```powershell
$env:FINANCE_WEB_DEMO='0'
python web_app.py
```

可选环境变量：`FINANCE_MAX_BLOCKS`、`FINANCE_DEMO_KEYWORDS`、`FINANCE_SKIP_IMAGE`、`FINANCE_EMBED_BATCH_SIZE`。  
接口：`GET /api/health`、`POST /api/ask`。

### 2.5 Docker Compose（一次性批量）

在 `code/` 目录下准备好 `output/knowledge_base.json`、`test/merged.xlsx` 及向量缓存卷后：

```bash
cd code
docker compose up --build
```

容器执行 `python main.py` 后退出；`./output` 可写以保存 `qa_results.json`。

### 2.6 运行回归评测（可选）

```bash
python scripts/benchmark_regression.py \
  --kb-path ./output/knowledge_base.json \
  --test-set ./data/复赛测试集.xlsx \
  --output ./output/benchmark_report.json \
  --parser-report ./output/parser_report.json \
  --final-output ./output/final_report.json
```

## 3. 核心输出说明

- `knowledge_base.json`：统一知识库（DocIR 聚合）
- `parser_report.json`：解析器命中率/回退链路/兼容率
- `benchmark_report.json`：问答时延、成功率、证据覆盖
- `final_report.json`：解析+问答闭环总报告
- `qa_results.json` / `qa_results.md`：运行 `main.py` 后的批量问答结果
- `kg_entities.json`：实体清单（离线抽取）
- `kg_triples.json`：关系三元组（离线抽取）
- `kg_index.json`：文档/页面实体倒排索引

## 4. 模块职责（按运行逻辑）

### A. 入口与编排

- `main.py`  
  唯一主入口：加载已有向量索引（`force_rebuild=False`），读取 `test/merged.xlsx`，写出 `output/qa_results.json`。

- `src/system/__init__.py`  
  对外导出 `FinancialQASystem` 及 CLI 辅助函数；实现见 `src/system/qa_engine.py`。

### B. 预处理与 DocIR

- `scripts/preprocess.py`  
  预处理主脚本。遍历文档、按解析器链路处理，输出知识库与解析报告（兼容率、回退链）。

- `src/pipeline/docir.py`  
  定义 DocIR 数据结构：`DocIR / DocPage / DocBlock`，并负责保存 `.docir.json` 与 `.md`。

- `src/pipeline/parsers.py`  
  解析器（`build_default_parsers`）：
  - `PlainTextParser`
  - `WordDocumentParser`
  - `PDFDocumentParser`（PyMuPDF 文本块 + 表格 + 无文本页 Tesseract OCR）
  - `ImageDocumentParser`

- `src/pipeline/__init__.py`  
  对外导出 DocIR 与解析器接口。

### C. 检索、推理、观测

- `src/system/qa_engine.py`  
  问答主引擎：加载知识库、构建检索栈、调用智能体、输出结果。

- `src/system/retrieval.py`  
  检索层：
  - `KnowledgeStore`
  - `TextRetriever`（向量检索）
  - `VisionPageRetriever`（页级图像向量：ColPali 或 CLIP，失败回落 BGE 文本代理）
  - `FusionRetriever`（RRF 融合）
  - `Neo4jGraphRetriever`（图检索，Neo4j 或本地三元组）
  - `HybridRetriever`（向量+图谱混合检索）

- `src/system/agent.py`  
  Document Agent：
  - `QueryPlanner`（拆子问题）
  - `TableTool`（表格证据处理）
  - `EvidenceReranker`（重排序）
  - `GenerativeAnswerer`（生成式回答）
  - `AnswerSynthesizer`（结构化答案与置信度）

- `src/system/kg.py`  
  离线 KG 抽取：实体/关系/属性抽取与 JSON 三元组落库。

- `src/system/observability.py`  
  可观测：结构化日志、Prometheus 指标（query 次数、时延）。

- `src/system/__init__.py`  
  system 模块统一导出。

### D. 服务与评测

- `scripts/build_index.py`  
  离线构建索引/清理缓存/查看缓存。

- `scripts/benchmark_regression.py`  
  回归评测脚本，输出 `benchmark_report.json` 与 `final_report.json`。

### E. 辅助转换工具（数据准备）

- `src/word转pdf.py`  
  Word 转 PDF 工具。

- `src/excel转pdf.py`  
  Excel 转 PDF 工具。

- `src/ppt转pdf.py`  
  PPT 转 PDF 工具。

### F. 部署文件

- `Dockerfile`  
  Ubuntu 22.04 + Python 3.10 + Tesseract（中英文）+ 应用代码。

- `docker-compose.yml`  
  一键启动 API，挂载 `output/`、持久化 `vector_cache`。

- `.dockerignore`  
  减小构建上下文，避免把本地向量缓存或完整输出打进镜像层。

- `.env.example`  
  `MMD_PATH`、`CACHE_DIR`、`IMAGE_BACKEND`、`ENABLE_GRAPH`、Neo4j 等环境变量示例。

- `scripts/build_index.sh`  
  容器/服务器上索引构建脚本。

- `scripts/run_service.sh`  
  容器/服务器上服务启动脚本。

### G. 依赖文件

- `requirements.txt`  
  CPU 可安装的核心依赖（FAISS、Transformers、pandas 等）。可选 GPU 包（vLLM、flash-attn、xformers）见文件内注释，需与 CUDA 版本自行对齐。

## 5. 历史清理说明

以下文件已在早期重构中移除，主链路不再引用：

- `src/zstq.py`、`src/deepseek_ocr.py`、`src/config.py`
- `scripts/evaluate.py`、`scripts/download_models.sh`
- `src/result2.py`（逻辑合并至 `src/system`，入口统一从 `src.system` 导入）
