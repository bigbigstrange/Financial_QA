# 阶段 3：可测混合检索运行手册

> 状态：稀疏/结构化通道、统一 Evidence 分数与重排分数隔离已完成；全量评测与 Qdrant 迁移待做
> 日期：2026-09-20

## 1. 本阶段目标

在阶段 2 的 DocIR v2 和知识快照基础上建立可测的混合检索：

- 稠密（FAISS）+ 稀疏（BM25）+ 结构化（机构/日期/来源/章节）三通道召回。
- 每路分数独立保存，RRF 排名融合，重排分数不再覆盖融合分数。
- 最终 Evidence 携带 `dense_score/sparse_score/structured_score/fusion_rank/fusion_score/rerank_score/retrieval_channels`。
- 保留 FAISS 基线；Qdrant 只在真实评测证明增益后再迁移。

## 2. 新增模块

### 稀疏通道

- `src/system/sparse.py`：确定性 token 化 + BM25（k1=1.5，b=0.75）。
- `SparseRetriever.search()` 返回 `sparse_score` 与 `retrieval_channel="sparse"`。

### 结构化通道

- `src/system/structured.py`：按机构别名、日期、来源、章节、文档版本和块类型召回。
- 返回 `structured_score` 与 `matched_filters`。

### 融合

- `HybridRetriever` 现在同时接收 dense、sparse、structured、graph 四路。
- 按 block 去重；RRF 生成 `fusion_rank/fusion_score`；保留每路 RRF 原始值。
- 删除版本仍在返回前通过 `visibility` 过滤。

### 重排

- `EvidenceReranker` 不再把 `rerank_score` 写回 `fusion_score`，两种尺度保持独立。

### Evidence 契约

- `src/contracts.py` 的 `evidence_from_legacy` 与 `evidence_from_v2` 都会保留各通道分数。

## 3. 验证

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_hybrid_retrieval -v
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_index_fingerprint_io tests.unit.test_retrieval_metrics -v
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' -m compileall -q app src scripts tests
```

当前结果：120 个测试全部通过，编译无错误。

## 4. 实测 A/B 结果（暂定基线）

评测语料：从 532MB 知识库流式抽取 996 块（标准证据 11/15 + 背景 985），标准证据 ID 暂取阶段 0 基线报告，来源标记为 `baseline_auto_provisional`，上线前必须人工校验。

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_phase3_eval_corpus.py `
  --kb output/knowledge_base.json `
  --baseline output/baseline/qa_report.json `
  --output-dir output/phase3 `
  --background-blocks 985

$env:HF_HOME='C:\code\个人资料\code\Agentic_RAG\model_cache\huggingface'
$env:HF_HUB_OFFLINE='1'
& '.\.venv\Scripts\python.exe' scripts/eval_retrieval_phase3.py `
  --corpus output/phase3/eval_corpus.jsonl `
  --gold output/phase3/golden_retrieval.json `
  --output output/phase3/retrieval_report.json
```

| 策略 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| dense（BGE-M3 基线） | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sparse（BM25） | 0.2000 | 0.2000 | 0.3333 | 0.3333 | 0.1913 |
| structured | 0.2000 | 0.6000 | 0.6667 | 0.3333 | 0.4152 |
| hybrid（RRF 三通道） | 0.2667 | 0.5333 | 0.6667 | 0.4167 | 0.4224 |
| hybrid + 启发式重排 | 0.3333 | 0.6667 | 0.6667 | 0.4444 | 0.5248 |

3000 块扩展语料结果（背景块 2985 + 标准证据 11）：

| 策略 | R@5 | R@10 | R@20 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| dense（BGE-M3 基线） | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| sparse（BM25） | 0.0000 | 0.0000 | 0.0667 | 0.0167 | 0.0000 |
| structured | 0.1333 | 0.3333 | 0.3333 | 0.1667 | 0.2415 |
| hybrid（RRF 三通道） | 0.1333 | 0.1333 | 0.1333 | 0.3333 | 0.1844 |
| hybrid + 启发式重排 | 0.0667 | 0.2000 | 0.3333 | 0.3333 | 0.1828 |

门禁状态：996 块暂定集通过当前门禁；3000 块扩展集未通过（R@10/MRR/nDCG 均低于暂定阈值）。这是诚实结果，不能被当作已达标。

结论：混合检索和重排方向仍然正确，但当前自动派生证据质量、背景采样和检索配置都还不足以支撑晋级。下一步必须先把 `output/phase3/gold_review.csv` 人工审核完，再在全量快照上重跑，不能调整阈值强行通过。

## 5. 缓存指纹

`IndexFingerprint` 已实际接入文本 FAISS 缓存：

- v2 快照构建时，`build_retrieval_stack` 从块里读取 `snapshot_id` 并传给 `TextRetriever`。
- `TextRetriever` 的缓存文件名使用 `fingerprint.cache_key()`；加载前读取 `*.fingerprint.json`，校验快照、模型、维度、归一化、距离、parser/chunker、演示子集和 block ID 摘要。
- 任意字段或块集合不匹配都会拒绝复用并重建，旧缓存不会被误用。

## 6. 尚未完成

- 人工审核 `output/phase3/gold_review.csv` 并扩充标准证据集（这是当前最阻塞项）。
- 在全量快照上重跑 A/B；BM25 参数、结构化阈值、API 重排候选各自做消融。
- 人工审核通过前，3000 块扩展集门禁保持失败关闭。
- Qdrant 候选索引：只有双索引切换、删除/更新一致性和回归门禁通过后才切换别名。
- 产生可复现的“索引切换/回滚”报告。
