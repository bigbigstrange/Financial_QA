# 阶段 2：文档解析与证据链路设计

> 状态：待用户评审  
> 日期：2026-09-19  
> 适用仓库：`finance-qa-platform`  
> 上位设计：`2026-09-19-financial-agentic-rag-integration-design.md`

## 1. 目标与成功标准

阶段 2 把“文件被解析成一批文本”升级为“每条答案证据都能稳定定位到某个不可变文档版本中的原始结构位置”。它只修正文档与证据基础，不同时更换检索算法、Embedding、向量库或 Agent 编排。

本阶段完成后必须满足：

1. Word 中段落与表格保持原始先后顺序，不再编造页码。
2. PDF 使用真实页码和区域坐标；无法确认阅读顺序时明确标记质量风险。
3. 表格保留表头、行、单位、标题、父表和原始位置，大表格可以按行组检索。
4. `document_id` 表示逻辑文档，`document_version` 表示一次不可变内容与解析配置组合，`block_id` 在该版本内唯一且可复现。
5. 新版本经过解析和完整性检查后才可激活；重复内容不重复索引。
6. 删除或停用文档后，下一次检索立即过滤该版本，不依赖向量索引异步清理完成。
7. 每次运行固定知识快照；历史 Run 中保存的 Evidence 仍可读取其原版本，新的检索不会再返回已停用版本。
8. 缓存键覆盖所有影响向量含义和块边界的配置，不会误用旧索引。

## 2. 当前实现与可验证问题

当前代码已有 `DocIR`、Word/PDF/Excel/PPT 解析器、预处理报告、FAISS 缓存和 Evidence 契约，可以渐进改造。但存在四类基础问题。

### 2.1 结构顺序不可靠

`WordDocumentParser` 先遍历 `document.paragraphs`，再遍历 `document.tables`。原文中穿插在段落之间的表格会全部移动到文末。Word、Excel 和图片等没有可靠物理页信息的格式统一写成 `page=1`，用户看到的“第 1 页”并不是真实来源位置。

### 2.2 身份不稳定且不唯一

文档 ID 和块 ID 主要由文件名生成，例如 `<stem>_p1_b1`。同名文件、目录中的重复文件或重新上传会产生冲突。阶段 0 的真实知识库审计结果是：

| 指标 | 数值 |
|---|---:|
| 文档数 | 1,146 |
| 页数 | 18,070 |
| 块数 | 481,276 |
| 重复 `block_id` 种类 | 240,637 |
| 表格块 | 236 |
| OCR 块 | 570 |

因此当前 `block_id` 不能作为证据主键，也不能安全用于去重、缓存、更新或删除。

### 2.3 表格只剩展示文本

现有解析器多数把首行默认当表头并生成 Markdown，缺少可靠的 `table_headers`、`table_rows`、单位、标题、父表 ID、行范围和原始定位。检索加载时又只读取 `block.text`，`text=""` 的表格块会被直接跳过，表格虽然出现在 DocIR 中，却可能根本进不了文本索引。

### 2.4 缺少版本与权威可见性

运行时直接加载一个 `knowledge_base.json`，FAISS 缓存键主要是文件哈希、维度和通道。系统没有逻辑文档、候选版本、激活版本或删除状态；删除后也没有查询时的权威过滤。因此仅删除源文件或异步清理向量都不能保证内容立即不可检索。

## 3. 方案比较与决策

### 方案 A：只修 Word 顺序和块 ID

优点是改动小。缺点是没有解决版本激活、删除可见性、缓存失效和历史证据读取，无法通过阶段 2 验收。此方案不采用。

### 方案 B：一次替换为 Docling、Qdrant、对象存储和异步任务

优点是目标形态完整。缺点是同时改变解析器、索引、存储和运行方式，出现答案波动时无法定位原因；本机资源与部署成本也会明显上升。Docling 的实际增益尚未用项目文档评测证明。此方案留到阶段 7 对照实验，不在阶段 2 采用。

### 方案 C：原生解析器修正 + DocIR v2 + 文档目录 + 不可变知识快照

这是本阶段采用的方案。继续使用 `python-docx`、PyMuPDF、Camelot/openpyxl 和现有 FAISS，把身份、结构、质量和生命周期先做正确。SQLAlchemy 文档目录是可见性的权威来源；文件系统保存不可变 DocIR 和知识快照；FAISS 仍是可重建的检索加速层。

这一路径改动范围可控，能与阶段 0/1 基线逐项比较，也为阶段 3 混合检索和阶段 4 Agent 主控提供可靠 Evidence。

## 4. 总体链路

```text
接收文件
→ 校验扩展名、MIME、文件签名、大小和内容哈希
→ 创建或匹配逻辑 Document
→ 创建 staged DocumentVersion
→ 选择原生解析器生成 DocIR v2
→ 结构化切块并生成稳定 block_id
→ 生成 ParseQualityReport
→ 质量门禁
→ 生成候选 KnowledgeSnapshot 与 FAISS 缓存
→ 完整性核对
→ 数据库事务原子激活版本和快照
→ Run 开始时固定 knowledge_snapshot_id
→ 检索候选
→ 按数据库可见性和运行快照过滤
→ 返回 Evidence
→ 通过 document_version + block_id 读取原始 DocIR 定位
```

解析产物、快照和向量索引都是可重建派生物。数据库中的文档状态、激活版本和访问范围是权威业务状态。

## 5. DocIR v2 数据契约

### 5.1 文档级字段

```text
schema_version          固定为 2
document_id             服务端生成的 UUID，逻辑文档身份
document_version        不可变版本 ID
source_name             用户看到的原文件名，仅用于展示
source_format           标准化扩展名
source_sha256           原始字节 SHA-256
source_size_bytes       原始文件大小
parser_name             实际成功的解析器
parser_version          解析器实现版本
parser_config_hash      影响解析结果的配置摘要
chunker_version         切块器版本
created_at              UTC 时间
language                语言或 unknown
access_scope            服务端写入的访问范围
effective_from/to       可选业务有效期
pages/elements          按原始文档顺序排列的结构元素
```

`document_id` 不从文件名推导。首次导入由服务端生成，后续“更新此文档”沿用它；“另存为新文档”即使文件名相同也得到新的 ID。

`document_version` 使用以下规范输入生成：

```text
dv_<sha256(document_id | source_sha256 | schema_version |
           parser_name | parser_version | parser_config_hash |
           chunker_version)[:32]>
```

这样同一逻辑文档、同一内容和同一解析配置会得到相同版本；解析器或切块规则改变时会生成新版本，避免把旧块误当新块。

### 5.2 元素与块字段

每个可检索块至少包含：

```text
block_id、parent_id、block_type
text、table_headers、table_rows、table_markdown
caption、unit、section_path
element_index、structural_locator
page、bbox、source_region_id
ocr_used、quality_flags
```

`page` 改为可空整数。只有 PDF、PPT 幻灯片或能够可靠获得页号的格式才赋值。Word 没有版面渲染结果时使用 `page=null`，依靠结构位置定位。

`structural_locator` 是可读、可复现的规范路径：

| 格式 | 示例 |
|---|---|
| Word 段落 | `body/000012/paragraph` |
| Word 表格行组 | `body/000013/table/rows/000020-000039` |
| PDF 文本区 | `page/000007/text/000004/bbox/72.0,108.2,510.4,166.8` |
| PDF 表格行组 | `page/000007/table/000001/rows/000000-000019` |
| Excel | `sheet/资产负债表/rows/000002-000021` |
| PPT | `slide/000003/shape/000007` |

`block_id` 使用：

```text
blk_<sha256(document_version | structural_locator | block_type)[:32]>
```

同一版本中重复文本位于不同位置时仍有不同 ID；相同版本重跑解析时 ID 保持一致。ID 不依赖文件路径、机器路径或当前时间。

### 5.3 兼容规则

旧 `knowledge_base.json` 只能通过显式的 `LegacyDocIRAdapter` 读取。适配器生成的证据标记 `quality_flags=["legacy_identity"]`，不能进入阶段 2 新快照的质量通过状态。新代码不再静默补造缺失 `block_id` 或页码。

## 6. 各格式解析设计

### 6.1 Word

使用 `document.element.body.iterchildren()` 遍历 OOXML body，根据 `CT_P` 和 `CT_Tbl` 在同一个循环中解析段落和表格，保存原始 `element_index`。标题样式维护 `section_path` 栈；普通段落、列表和表格都继承当时的标题路径。

Word 的 `page` 默认是 `null`。只有以后引入可验证的版面渲染器并保存页映射时才写页码。定位显示为“章节路径 + 第 N 个元素”，例如“风险管理 / 流动性风险，第 13 个元素”。

空段落不创建检索块，但其 body 下标仍保留在后续元素的 `structural_locator` 中，防止删除空段落造成未变化元素被无意义重编号。嵌套表格在单元格内递归解析，并将父单元格位置写入 locator。

### 6.2 PDF

PyMuPDF 继续负责文本块、真实页码和 bbox；Camelot 继续作为表格候选解析器。每页保留原始块序号与坐标，规范化 bbox 时统一保留一位小数。

PDF 阅读顺序只做确定性排序：优先使用解析器块序；当同页出现明显多栏、重叠或异常跳跃时不猜测唯一顺序，写入 `uncertain_reading_order`。表格区域与文本区域按 bbox/文本重合检测，避免表格内容同时以正文和表格重复入库。

OCR 仅在页面无文本或文本覆盖低于配置阈值时启用。OCR 块写入 `ocr_used=true`、语言、引擎版本和 `ocr_low_confidence` 等质量标记。OCR 失败是页面级失败，进入质量报告；不能把空页面记为成功。

### 6.3 Excel 与 PPT

Excel 不使用伪页码。每个工作表以名称和原始行范围定位，保留公式值/展示值的选择策略、表头、合并单元格信息和行组范围。阶段 2 保持当前最大行限制，但质量报告必须记录截断，截断版本不能无提示激活。

PPT 使用真实幻灯片号，shape 按文档顺序和 shape id 定位；文本与表格分别生成块。旧 `.ppt/.pps` 的 COM 回退必须记录实际解析器和版本。

## 7. 表格规范化与切块

表格先形成父表，再生成可检索行组。父表保存完整结构或其不可变引用；行组通过 `parent_id` 指向父表。

### 7.1 表头与单位

1. 优先使用解析器明确识别的表头。
2. 没有明确语义时可把首个非空行作为候选，但必须标记 `inferred_header`。
3. 多级表头保留原始行，并生成用于检索的扁平列名，例如 `2025年/金额`。
4. 单位从表格标题、表头或邻近段落提取；不能确认时为 `null`，不自行猜测。
5. 合并单元格保留 span 元数据；用于检索的规范表格可以前向填充表头，原始单元格值仍保留。

### 7.2 行组策略

默认每组最多 20 个数据行，同时受 token 上限约束。每个行组都重复：章节路径、表格标题、表头、单位和行范围。小表格在 token 上限内保持一个行组。跨页表格只有在列数、规范表头和标题关系都一致时合并，否则保留独立父表并标记 `possible_continuation`。

用于 Embedding 的 `text` 是确定性序列化结果，不能只保存空字符串：

```text
[章节] 流动性风险
[表格] 期限结构
[单位] 万元
[列] 期限 | 2024年 | 2025年
[行 20] 1年以内 | 100 | 120
...
```

`table_rows` 仍保存结构化数据，Markdown 只用于展示，不作为唯一事实来源。

## 8. 解析质量报告与门禁

每个 `DocumentVersion` 都生成 `ParseQualityReport`，知识快照再生成聚合报告。报告至少包含：

```text
document_id、document_version、parser/config versions
page_or_element_count、block_count、block_type_counts
table_count、table_row_group_count、ocr_page_count
empty_block_count、duplicate_block_id_count
missing_locator_count、missing_parent_count
failed_page_or_element_count、truncated_count
uncertain_reading_order_count、quality_flags
status、failure_reasons
```

以下任一条件触发硬失败，版本保持 `failed` 或 `staged`，不得激活：

- 整篇无有效块；
- `block_id` 重复；
- 块缺少 `document_version` 或 `structural_locator`；
- 表格行组缺少父表、表头或实际数据；
- 解析器报告未处理异常或静默截断；
- DocIR 序列化后无法按 schema v2 重新读取；
- 候选快照引用不存在的 DocIR、块或版本。

`uncertain_reading_order`、`inferred_header`、`ocr_low_confidence` 属于软警告。它们允许激活，但必须进入报告和 Evidence，便于回答校验与人工抽查。

## 9. 文档目录与版本状态机

### 9.1 数据模型

新增 SQLAlchemy 模型：

**DocumentRecord**

```text
id UUID PK
display_name
owner_id / access_scope_json
active_version_id nullable
status active | deleted
created_at、updated_at、deleted_at
```

**DocumentVersionRecord**

```text
id document_version PK
document_id FK
source_sha256、source_size_bytes、source_format
parser_name、parser_version、parser_config_hash、chunker_version
status staged | validating | ready | active | superseded | failed | deleted
docir_path、quality_report_path
created_at、activated_at、deleted_at
failure_code、failure_message
UNIQUE(document_id, source_sha256, parser_config_hash, chunker_version)
```

**KnowledgeSnapshotRecord**

```text
id snapshot_id PK
manifest_path
status candidate | active | retired | failed
document_versions_json
parser_schema_version、chunker_version
block_count、manifest_sha256
created_at、activated_at
```

`RunRecord` 增加不可变的 `knowledge_snapshot_id`。Evidence 已包含 `document_id`、`document_version` 和 `block_id`；阶段 2 补全其 locator 字段但不改变现有 API 的基本形状。

### 9.2 生命周期

```text
上传/更新
→ staged
→ validating
→ ready
→ 构建 candidate snapshot/index
→ 原子激活 active
→ 旧版本 superseded
```

相同逻辑文档的 `source_sha256 + parser_config_hash + chunker_version` 已存在时返回现有版本，记录重复操作，不重新解析和索引。

激活在一个数据库事务中更新：新版本状态、`DocumentRecord.active_version_id`、新快照状态和旧快照状态。文件重命名只影响 `display_name`，不会制造新版本。

### 9.3 删除与回滚

删除先在数据库事务中把 Document/Version 标记为不可见，并生成不含该版本的新 active snapshot。检索结果在返回前必须查询或使用请求开始时固定的权威可见性集合，因此事务提交后的下一次新 Run 立即看不到该内容。向量、DocIR 和页图随后清理；清理延迟不影响可见性。

历史 Run 固定其 snapshot，并可以在审计权限下读取历史 Evidence。普通新查询不得使用 superseded/deleted 版本。回滚通过重新激活一个已通过质量门禁的旧版本并生成新 snapshot 完成，不修改旧快照内容。

## 10. 知识快照与运行时过滤

知识快照是不可变 manifest，包含排序后的 active `document_version`、每个版本的 DocIR 路径、块数量和配置版本。`snapshot_id` 由规范化 manifest 的 SHA-256 生成。

`KnowledgeStore` 从 manifest 加载块，不再把单个 `knowledge_base.json` 文件本身当权威状态。新快照中的每个运行时块必须透传：

```text
snapshot_id、document_id、document_version、block_id
source_name、page、bbox、section_path、structural_locator
parent_id、block_type、content、quality_flags
```

`RunService` 在推理开始前固定 active `knowledge_snapshot_id`，与 ModelSnapshot 一样先持久化再调用模型。检索器接收该 snapshot；返回候选后再经过 `DocumentVisibilityRepository` 校验。双重约束用于防止旧 FAISS 元数据、进程内缓存或异步清理返回已删除版本。

阶段 2 不要求为每个快照复制一份完整向量。可以复用物理索引，但所有候选必须携带版本 ID 并执行可见性过滤；若候选不足，检索器扩大候选窗口一次，不允许用被过滤的结果补足。

## 11. 缓存版本规则

文本和视觉索引缓存键统一由 `IndexFingerprint` 生成：

```text
snapshot_id
embedding_model_provider/name/revision
embedding_dimension
normalization_method
distance_metric
docir_schema_version
parser_versions_hash
chunker_version
retrieval_channel
demo_subset_hash（仅基线/开发模式）
```

缓存元数据文件保存完整 fingerprint、块数量、block ID 摘要和创建时间。加载时逐字段校验，任何不一致都拒绝复用并重建；不能只凭文件存在或向量维度相同就加载。

旧 `<kb_hash>_<dim>_text.faiss` 缓存只供 legacy 模式使用。阶段 2 新快照不覆盖旧缓存，迁移失败时可以切回阶段 1 基线。

## 12. Evidence 定位与读取

检索结果转换为 Evidence 时保留：

```text
evidence_id = ev_<sha256(snapshot_id | document_version | block_id)[:32]>
document_id、document_version、block_id
source_name、page、bbox
section_path、structural_locator
text/table 摘要、quality_flags
```

新增只读 `EvidenceReader` 服务：根据 Evidence 身份读取对应不可变 DocIR，校验 block 确实属于该版本，再返回原文、表格结构、页码/章节位置和页图引用。API 层提供已鉴权的 `GET /api/evidence/{evidence_id}`；调用者只能读取自己当前有权访问的内容，审计角色可以按历史 Run 读取已删除版本。

定位显示规则：

- PDF：文档名 + 页码 + bbox，可选页图高亮。
- Word：文档名 + 章节路径 + 元素序号；不显示虚假页码。
- Excel：文档名 + 工作表 + 行范围。
- PPT：文档名 + 幻灯片号 + shape。

## 13. 安全与资源边界

阶段 2 的文件入口执行：扩展名、MIME 和 magic bytes 三方校验；文件名不参与存储路径；使用随机内部键；限制文件大小、页数、表格行数、解压后总大小和文件数；OOXML 宏、外部链接和嵌入脚本不执行。失败文件进入隔离状态并保存安全错误码，不把本机绝对路径或解析器堆栈返回给普通用户。

解析配置设定页数、OCR 页数、单表行数和总块数上限。达到上限时明确失败或进入人工检查，不能静默截断后激活。文档 ACL 只能来自服务端身份和目录记录，模型或文档正文无权修改。

## 14. 迁移策略

现有约 48 万块的知识库不原地修改：

1. 保留阶段 1 的 `knowledge_base.json` 和旧 FAISS 缓存作为只读基线。
2. 用同一批原始文件在独立目录生成 DocIR v2、质量报告和候选 snapshot。
3. 先用小型固定夹具验证顺序、表格和身份，再对全量输出运行审计。
4. 全量门禁通过后构建新 FAISS，执行阶段 0 的固定问题回归。
5. 只切换 active snapshot 指针，不覆盖旧产物。
6. 发现严重回归时切回旧运行 profile；修复后重新生成新的候选 snapshot。

旧数据中 240,637 种重复块 ID 不做就地映射。它们由原始文件重解析生成新身份；无法找到原文件的内容只能保留在 legacy profile，不能伪装成阶段 2 合格数据。

## 15. 实施边界

阶段 2 包含：DocIR v2、稳定身份工具、Word/PDF/Excel/PPT 针对性修复、表格行组、质量门禁、文档目录、知识快照、可见性过滤、缓存 fingerprint、EvidenceReader、迁移脚本和测试。

阶段 2 不包含：

- BM25/RRF、专用 reranker 或结构化召回，它们属于阶段 3；
- LangGraph 主控、补检索和答案校验，它们属于阶段 4；
- SSE、断线续传、异步 Worker 和人工审批，它们属于阶段 5；
- Qdrant、Docling 默认替换或图谱扩展，它们必须先通过后续对照评测。

为避免本阶段膨胀，管理端上传界面可以后置；本阶段只要求服务层、最小文档管理 API 和可测试的 CLI 能完成新增、更新、删除、激活与回滚。

## 16. 测试设计

### 16.1 确定性单元测试

- 构造“段落—表格—段落”的 DOCX，断言输出顺序和 locator 完全一致，`page=null`。
- 同名不同目录文件产生不同 `document_id` 和 block ID。
- 同一版本重复解析产生完全相同的 block ID。
- 内容或解析器/切块版本改变时生成新 `document_version`。
- 大表格按行组拆分，每组重复表头、单位和父表 ID。
- PDF 块保留真实页码和 bbox；无文本页只在阈值触发时 OCR。
- 表格块的确定性文本能进入 `KnowledgeStore`，不会因原始 `text` 为空被跳过。
- fingerprint 任一字段变化都拒绝旧缓存。

### 16.2 生命周期集成测试

- 相同哈希重复上传不重建版本或索引。
- 新版本在 ready 前不参与检索；激活后新 Run 固定新 snapshot。
- 删除事务提交后，下一次检索立即返回 0 个该版本候选，即使旧 FAISS 仍存在。
- 历史 Run 的 Evidence 可按审计权限定位，普通新查询不能检索旧版本。
- 激活失败不改变旧 active snapshot；回滚能恢复旧版本。
- 两个并发更新通过唯一约束和事务只激活一个确定版本。

### 16.3 全量与回归测试

- 新知识快照 `duplicate_block_id_count=0`、`missing_locator_count=0`、`empty_block_count=0`。
- 每个表格行组都有父表、表头和数据；每个块都能由 EvidenceReader 读回。
- 用阶段 0 固定问题对比答案、证据和延迟。允许因正确结构修复而变化，但必须逐项解释；不能把无法解释的下降作为通过。
- 保留阶段 1 profile 做一键回退验证。

## 17. 阶段验收门槛

只有以下条件全部满足，阶段 2 才算完成：

1. Word 混排夹具顺序 100% 正确，所有无真实页信息的块 `page=null`。
2. 新快照重复 block ID、缺失 locator、空内容块均为 0。
3. 抽样 Word、PDF、Excel、PPT 和表格 Evidence 均可定位回 DocIR 原文；PDF 页码和 bbox 可核对。
4. 表格行组保留表头、单位、行范围和父表关系，能实际进入文本索引。
5. 重复上传不重复索引；更新只在质量门禁后原子切换。
6. 删除提交后的下一次新 Run 不再返回被删版本，无需等待 FAISS 物理清理。
7. 运行快照保存 `knowledge_snapshot_id`，历史 Evidence 身份不随后续更新漂移。
8. 缓存 fingerprint 不匹配时稳定重建，旧缓存不会被误加载。
9. 阶段 0 固定问题回归无未解释的答案或证据退化。
10. 全部单元、集成和契约测试通过，旧 `/chat` 与 `/runs` 兼容接口继续工作。

## 18. 已确定的工程决策

1. 先保证证据身份和生命周期正确，再升级检索算法。
2. Word 没有可靠页映射时使用结构定位，不制造页码。
3. 数据库控制可见性，向量索引只做加速；物理删除可以延迟，逻辑过滤必须立即生效。
4. DocIR、知识快照和缓存均不可变，通过创建新版本切换，不覆盖旧产物。
5. 表格结构数据是事实来源，Markdown 是展示格式，Embedding 文本是确定性派生格式。
6. 阶段 2 保留现有原生解析技术；Docling 只在后续真实样本评测证明收益后采用。
7. 新格式严格执行 schema v2；旧格式通过显式适配器隔离，不允许静默补造身份或位置。
