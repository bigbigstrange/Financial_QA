# 纯图片证据召回 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前遗漏的 144 张无文字切块页图纳入可追溯视觉召回，并能在鉴权后查看原图，不伪造文字。

**Architecture:** 从快照 DocIR 建立独立视觉页清单，Chinese-CLIP 对全部有效物理页编码，有切块时返回切块、无切块时返回页图证据。复用 EvidenceRecord 的 JSON 元数据和鉴权链路，新增受控图片响应及前端预览，不修改原始解析产物或文本索引。

**Tech Stack:** 现有 Python 3.11、unittest、Pillow、NumPy、Chinese-CLIP、FastAPI、SQLAlchemy、Vue 3、TypeScript；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-09-23-image-only-evidence-design.md`（用户已确认）。上位语料规范：`specs/001-complete-local-corpus/spec.md`。

## Global Constraints

- 本次不恢复付费服务、不新增依赖或调度器、不绕过 Office 安全阻止、不自动激活候选库。
- 原 91 份在线库及其历史快照保持不变。
- 纯图片证据的 `chunk_id` 为空、`block_ids` 为空、`text` 为空。
- 无物理页码的 TXT 不创建虚构页面。
- 不将本机绝对路径作为前端图片 URL，不把访问令牌放在 URL 中。
- 保留旧视觉缓存，新索引使用不同的页身份／缓存格式版本。
- 所有实际运行设置 `FINANCE_DISABLE_PAID_API=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。
- 项目根：`C:\code\个人资料\code\Agentic_RAG\finance-qa-platform`；Python 使用项目 `.venv/Scripts/python.exe`。
- 当前工作区有大量未提交且未追踪的既有成果。执行前按 using-git-worktrees 规则确认执行位置；不能用一个缺少这些成果的 HEAD 工作树冒充完整基线，不 stash/reset/搬走原目录，不重跑现存 GPU 作业。
- 每项完成后只提交可归属本任务的变更；若与既有未提交成果无法分离，保留工作区并记录原因，不把用户改动夹带提交。

## Review Focus

1. 多页切块命中第二页时，图片定位仍须指向第二页（任务 2、3）。
2. 精简加载或文档筛选时，视觉索引不能偷偷扩展到未选文档；独立图片文档在未筛选全量模式下不能被排除（任务 1、2）。
3. 用户切换会话时图片请求尚未完成，返回值不得污染新会话或泄漏 Blob URL（任务 4）。
4. 快照哈希一致但图片文件被替换，不能继续返回旧身份下的新内容（任务 3）。
5. 历史文字证据未带新字段仍可访问；普通用户不能借图片接口读取其他用户证据（任务 3、5）。

## File Structure

| 文件 | 职责 |
|---|---|
| 新建 `src/system/visual_pages.py` | 页身份、页清单、受限产物路径解析；不加载模型 |
| 修改 `src/system/vision.py`、`src/system/retrieval.py` | 视觉页输入、图片专属证据、缓存隔离与检索装配 |
| 修改 `src/contracts.py`、`src/agent/nodes.py`、`src/agent/prompts.py`、`app/services/qa_runtime.py` | 元数据保真与纯图片回答边界；必要时调整现有 legacy 回答路径 |
| 修改 `app/services/evidence_reader.py`、`app/api/evidence.py`、`app/schemas/__init__.py` | 统一授权、页图身份验证、图片响应 |
| 修改 `frontend/src/api/client.ts`、`frontend/src/views/ChatView.vue` | 鉴权 Blob 请求、原图预览与生命周期 |
| 修改 `scripts/validate_visual_channel.py` | 参数化候选目录、全量视觉编码与真实查询报告 |
| 新建／扩展对应 unit、integration 测试 | 使用真实小图片、manifest、数据库和 HTTP，模型编码仅在单测中替代 |

## 执行前基线

- [ ] 阅读设计、运行记录、相关目录 AGENTS.md（如有），核对实际进程与用户已有修改。

```powershell
$env:FINANCE_DISABLE_PAID_API='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -c "from scripts.pipeline_liveness import live_jobs; print(live_jobs())"
.venv/Scripts/python.exe -m unittest discover -s tests -q
```

已知基线为 272 项、268 通过、4 跳过；执行时以新输出为准。只读记录 manifest、chunks.jsonl 和活动快照身份，作为最终不变性对照。

### Task 1: 可核对的视觉页清单

**Files:** Create `src/system/visual_pages.py`; Test `tests/unit/test_visual_pages.py`。

**Interfaces:**
- `make_visual_page_id(snapshot_id: str, document_id: str, document_version: str, page_no: int) -> str`
- `resolve_artifact_path(reference: str, base_dir: Path, allowed_roots: tuple[Path, ...]) -> Path`
- `build_visual_pages(manifest_path: Path, blocks: list[dict], *, allowed_roots: tuple[Path, ...], selected_versions: set[str] | None = None) -> list[dict]`
- 返回项固定键：`page_id, snapshot_id, document_id, document_version, source_doc, page, image_ref, image_sha256, chunks, quality_flags`。`chunks` 为既有运行时切块列表；未关联页为 `[]`。清单由验证脚本写出，生产加载不偷偷写文件。

- [ ] 编写真实临时 manifest/DocIR/PNG 用例：两个同名文档不同版本各 1 页，其中一页无块；再加无物理页 TXT。使用已有 `DocIR.save`、`SnapshotManifest.build`、`write_manifest_atomic`，Pillow `Image.new('RGB', (8, 8)).save(path)` 生成图片。

```python
def test_page_identity_separates_version_and_page(self):
    from src.system.visual_pages import make_visual_page_id
    a = make_visual_page_id('s', 'd', 'v1', 1)
    self.assertTrue(a.startswith('vp_'))
    self.assertEqual(a, make_visual_page_id('s', 'd', 'v1', 1))
    self.assertNotEqual(a, make_visual_page_id('s', 'd', 'v2', 1))
    self.assertNotEqual(a, make_visual_page_id('s', 'd', 'v1', 2))
```

- [ ] 运行 `.venv/Scripts/python.exe -m unittest tests.unit.test_visual_pages -v`，确认新增行为失败。
- [ ] 实现规范 JSON SHA-256 页身份、manifest/DocIR 摘要核验、版本与页码匹配；用 `(version, page)` 关联切块，枚举 DocIR 而非仅枚举切块。

```python
identity = [snapshot_id, document_id, document_version, page_no]
page_id = 'vp_' + hashlib.sha256(json.dumps(identity, ensure_ascii=False,
    separators=(',', ':')).encode('utf-8')).hexdigest()[:32]
resolved = (base_dir / reference).resolve()
if not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots):
    raise ValueError('image_path_outside_artifact_roots')
```

- [ ] 对缺失引用／文件保留页记录并加 `missing_page_image`，不伪造摘要；无物理页码跳过。空、重复或不一致身份立即拒绝。受限路径只允许 KB 派生产物和本项目 `output/kb_v3/parsed`，不允许整个磁盘。
- [ ] 加入显式 `selected_versions` 用例，确保子集不扩展；`None` 表示完整 manifest，包括没有文字切块的文档。运行该测试文件，确认通过并记录变更。

### Task 2: 视觉召回与纯图片结果语义

**Files:** Modify `src/system/vision.py`, `src/system/retrieval.py`, `src/contracts.py`, `src/agent/nodes.py`, `src/agent/prompts.py`, `app/services/qa_runtime.py`; Test `tests/unit/test_chinese_vision.py`, `tests/unit/test_local_mode.py`, `tests/unit/test_visual_score_isolation.py`。

**Interfaces:** 扩展 `ChineseVisionRetriever.build(blocks, page_images, force_rebuild=False, *, visual_pages=None)`，无新参数时保持旧调用兼容。有切块页保持既有结果；无切块页产生 `block_id=page_id, evidence_kind='page_image', chunk_id=None, block_ids=[], content=''`。

- [ ] 添加无文字页命中、不可见版本过滤、多页切块命中第二页和 `top_k=0` 用例。构造 `retriever.pages` 的真实页记录，不启动模型：

```python
hits = retriever.resolve_hits(np.array([0.9]), top_k=1)
self.assertEqual(hits[0]['evidence_kind'], 'page_image')
self.assertEqual(hits[0]['content'], '')
self.assertIsNone(hits[0]['chunk_id'])
self.assertEqual(hits[0]['block_ids'], [])
self.assertEqual(hits[0]['vision_page'], 1)
```

- [ ] 运行上述三个 unittest 模块并观察新增测试失败，然后实现分支：

```python
if page['chunks']:
    candidates = page['chunks']
else:
    candidates = [{**{k: page[k] for k in ('snapshot_id', 'document_id',
        'document_version', 'source_doc', 'quality_flags')},
        'block_id': page['page_id'], 'page_id': page['page_id'],
        'evidence_kind': 'page_image', 'chunk_id': None, 'block_ids': [],
        'content': '', 'page': page['page'], 'image_sha256': page['image_sha256']}]
```

- [ ] 所有结果保留真实 `vision_page`、图片摘要和图像分数；纯图片不被强制赋值为伪 `chunk_id`。`evidence_from_v2` 显式保留 `evidence_kind/page_id/image_sha256`，已有文本证据字段不破坏。
- [ ] `build_retrieval_stack` 仅为 Chinese-CLIP 装配独立视觉页，不改变 TextRetriever／SparseRetriever 输入。子集按已选版本限制；全量按完整 manifest。无文字而有视觉页时，不因文本为空直接终止整个检索栈，文本通道采用既有空通道语义。
- [ ] 提升新视觉缓存 schema，指纹覆盖模型、处理参数、页面身份和图片摘要。首版允许一次全量本地编码，不设计新缓存服务；旧缓存不删除。
- [ ] 运行真实 Agent 本地流程测试：纯图片结果保留为证据且显示“仅图像证据，未经文字解析／答案验证”；空文本不变成推测描述，校验状态不因有图片而成为已验证。混合结果继续使用真实文本。检查 legacy 回答入口，避免其把空内容过滤后误报“没有任何证据”。
- [ ] 运行三个目标测试文件及 Agent 测试，确认通过，记录变更。

### Task 3: 鉴权原图回查

**Files:** Modify `app/services/evidence_reader.py`, `app/api/evidence.py`, `app/schemas/__init__.py`; Test `tests/integration/test_evidence_api.py`。

**Interfaces:** 保留 `EvidenceReader.read(db, evidence_id, requesting_user) -> dict`；增加 `EvidenceReader.read_image(db, evidence_id, requesting_user) -> tuple[bytes, str]`。详情增加 `evidence_kind: str = 'text', vision_page: int | None = None, image_available: bool = False`；HTTP 新增 `GET /api/evidence/{evidence_id}/image`。

- [ ] 在现有 SQLite/TestClient 夹具中增加一份 2 页 DocIR 和两张不同 PNG，保存页图证据、切块证据。使用生产页身份算法创建记录，但图片响应期望值直接取测试原图字节，不用生产读取函数生成期望。

```python
response = self.client.get('/api/evidence/ev-page/image')
self.assertEqual(response.status_code, 200)
self.assertEqual(response.content, expected_png_bytes)
self.assertEqual(response.headers['content-type'], 'image/png')
detail = self.client.get('/api/evidence/ev-page').json()
self.assertEqual(detail['text'], '')
self.assertEqual(detail['evidence_kind'], 'page_image')
```

- [ ] 运行 `.venv/Scripts/python.exe -m unittest tests.integration.test_evidence_api -v`，记录新路由及页证据失败。
- [ ] 抽取现有所有者／管理员授权为两个入口共用的内部逻辑；先授权，再查文件。页图分支按保存的快照读取 manifest，校验 DocIR hash、文档 ID／版本、页 ID 和真实页码。切块图像优先用验证过的 `vision_page`，而非 `page_start`；旧文字证据无图像元数据仍按现有逻辑可读。
- [ ] 从受信任 DocIR 解析图片路径，不使用请求参数或保存 JSON 的任意路径直接读取。验证允许目录、PNG/JPEG 类型和图片摘要。最多读取 64 MiB，超限拒绝；读取后对同一份字节做摘要与格式检查，并返回该字节，避免“校验后路径被替换”的二次读取。

```python
data, media_type = EvidenceReader.read_image(db, evidence_id, user)
return Response(content=data, media_type=media_type,
    headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})
```

- [ ] 添加并执行：跨用户 404、未认证拒绝、缺图 404、图像替换／DocIR 篡改／页 ID 篡改 409、越界路径拒绝、非图片拒绝、历史文字证据不回归、多页切块第二页字节匹配。新读取异常映射为已定义错误，不泄漏磁盘路径。
- [ ] 确认所有测试通过并记录变更。候选库注册／活动状态测试只使用隔离数据库。

### Task 4: 前端原图查看

**Files:** Modify `frontend/src/api/client.ts`, `frontend/src/views/ChatView.vue`。

**Interfaces:** `apiBlob(path: string, init: RequestInit = {}): Promise<Blob>` 与现有 `api<T>` 共享同一鉴权／错误处理；证据详情使用 `image_available/evidence_kind/vision_page`。

- [ ] 先用 Task 3 的隔离服务和已知测试证据确认旧页面没有图片入口，记录浏览器验收基线。不为前端引入新测试依赖。
- [ ] 将 client 中 fetch/鉴权/错误解析抽成同文件内部 `request(path, init): Promise<Response>`，保持既有 JSON 返回行为：

```typescript
export async function apiBlob(path: string, init: RequestInit = {}): Promise<Blob> {
  return (await request(path, init)).blob()
}
```

- [ ] 证据展开区按 `image_available` 显示按钮，用 `apiBlob` 和 AbortController 请求图片：

```typescript
const blob = await apiBlob(`/api/evidence/${encodeURIComponent(id)}/image`,
  { signal: controller.signal })
if (controller.signal.aborted) return
const url = URL.createObjectURL(blob)
```

- [ ] 为每个证据记录 controller 和 URL；关闭、会话切换、卸载时 abort 并 revoke。请求完成时核对会话代次，旧响应不能写入新会话；失败提示不展示旧图。
- [ ] 显示原图来源、版本、真实命中页；纯图提示与空文本分开显示。图片提供有意义 alt 和可键盘操作的按钮，不使用磁盘路径或令牌查询参数。
- [ ] 在 frontend 运行 `npm run build`。浏览器检查 owner 查看、跨用户失败、加载失败重试、连续开关、加载中切换会话、旧文字全文查看。网络检查确认 Bearer 在请求头、图片接口无公共缓存。记录结果后再标完成。

### Task 5: 全量候选的离线验收与报告

**Files:** Modify `scripts/validate_visual_channel.py`; Create `tests/unit/test_visual_validation_cli.py`; Update `docs/baseline/2026-09-22-local-completion-runbook.md`; outputs only under `output/kb_full`。

**Interfaces:** `validate_visual_channel.py --kb-dir PATH --cache-dir PATH --model PATH --device cuda|cpu --report PATH --audit-only`。`--audit-only` 只生成页清单／来源审计，不导入模型；正式运行编码并执行真实查询。页清单写 `visual_pages.json`，最终报告含每页状态、模型与快照指纹。

- [ ] 编写 CLI 临时夹具测试：候选目录与活动目录不同，audit-only 对含 1 张图的 manifest 输出 1 页；活动目录和 chunks 文件摘要不变；缺图明确报告失败，不记录视觉就绪。运行 unittest 文件观察失败后加 argparse 和 Task 1 复用代码。
- [ ] 核对真实进程、活动 91 份快照和原产物摘要；运行目标测试、全套 unittest 和 frontend 构建。失败先修复，不启动大规模编码掩盖失败。
- [ ] 在禁用付费及离线模型环境运行：

```powershell
.venv/Scripts/python.exe scripts/validate_visual_channel.py --kb-dir output/kb_full --cache-dir output/kb_full/vector_cache --model ../model_cache/chinese-clip-vit-base-patch16 --device cuda --report output/kb_full/visual_validation_v2.json
```

- [ ] 逐页核对 17,957 页：每项成功编码或明确失败，144 个历史缺口逐项有身份映射；检查向量有限性／归一化和缓存顺序。再跑原有 3 个查询及“银行业金融机构资产规模增长”“理财产品走势图”，保存实际排名和结果，不用预期答案替代观测。
- [ ] 真实命中经过证据转换与内存数据库，用 HTTP 回查原图字节。对自然查询未命中的指定图表，额外做确定性页身份回查以验证链路，并明确其不代表自然召回成功。
- [ ] 复核 chunks 仍 30,406、DocIR/manifest 不变、活动快照仍 91 份、付费开关仍 1。报告分别列覆盖、功能、权限、浏览器、相关性观察，更新运行记录，不覆盖旧报告。
- [ ] 进行独立只读代码审查，修复重要问题后复跑相关测试。仅当上述证据齐全才宣称纯图片证据链路完成；不宣称总体准确率达标。

## 本计划后的剩余事项

以下事项不靠本计划冒充完成，也不阻塞上述本地开发：

- 3 份 Office 异常：本轮原始数据目录检索仅找到原件，未见同名替代件；等待可信 PDF 或可安全打开的新副本，不关闭 Office 保护、不重复盲重试。
- 视觉相关性：先收集新全量结果及人工判定，再单独确定过滤／重排修改；不能为了三个样例硬编码排名。
- 人工金标准：可准备候选证据和标注表，参考答案与证据必须经过人工审核后才能计算正式质量门禁。
- 付费图谱、生成和模型校验：用户明确恢复前不调用，已完成的 6 份图谱文档保留。
- 全量库发布：隔离联调通过后单独确认切换与回退，不自动替换当前活动库。

## 计划自检

- 设计的页身份／无文本语义对应任务 1、2；安全接口对应任务 3；前端生命周期对应任务 4；完整性、真实查询、保留在线库对应任务 5。
- Review Focus 五项均已落实到具体任务检查；没有把单测构造得分当作自然语言召回评测。
- 不新增 OCR、调度器、模型服务、数据库迁移或缓存平台；派生页清单与现有索引职责分开。
- 计划执行状态：任务 1–5 的实现与本地候选验收完成（2026-09-23）；原逐步清单保留为计划文本，实际命令、RED→GREEN、审查修复和偏差见 `.superpowers/sdd/2026-09-23-image-only-evidence/progress.md` 及 baseline 运行记录。实际浏览器覆盖原图显示、关闭重开、新会话；负面访问及异步竞态通过 HTTP／Node 自动测试，未宣称全浏览器人工验收。
- 最终报告：`output/kb_full/visual_validation_v2_final.json`。303 项 Python 测试中 299 通过、4 跳过；前端构建及生命周期检查通过；17,957 页编码，144 页历史缺口覆盖。原 91 份活动库保持不变，相关性问题与其他后续事项没有被算作完成。
