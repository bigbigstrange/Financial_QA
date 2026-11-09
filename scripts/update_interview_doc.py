# -*- coding: utf-8 -*-
"""Update interview docx Project B content based on finance-qa-platform reality."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = Path(r"g:\面试\Agentic_RAG\AI应用全栈工程师-张艳面试完整口述稿.docx")
    if not src.exists():
        raise FileNotFoundError(f"源文档不存在: {src}")

    # 优先输出到项目目录，避免 G 盘原文件被 Word 占用导致无法写入
    candidates = [
        root / "AI应用全栈工程师-张艳面试完整口述稿-已更新.docx",
        Path(r"g:\面试\Agentic_RAG\AI应用全栈工程师-张艳面试完整口述稿-已更新.docx"),
    ]
    doc = Document(str(src))

    replacements = {
        "第二，垂直领域多模态RAG工程落地，面向金融文档场景搭建多格式知识库问答平台。支持PDF、Word、扫描图片等混合文档解析，采用BGE文本向量+ColPali图像向量多路检索，通过RRF算法融合多路召回结果，搭配拆分子问题、机构业务过滤、精排模型提升答案准确性；针对本地硬件资源不足、内存溢出、跨机构信息混淆等工程问题设计分批建索引、演示精简模式、业务硬过滤等落地方案，完成从知识库构建到可交互Web演示系统的全流程开发。": (
            "第二，垂直领域多模态RAG工程落地，面向金融文档场景搭建可上线知识库问答平台。支持PDF、Word、扫描图片等混合文档解析，"
            "自研DocIR统一页码与来源溯源；检索侧采用BGE文本向量+ColPali/CLIP页级图像向量双路召回，RRF融合后再叠加可选本地三元组/Neo4j图谱通道，"
            "由HybridRetriever做向量+图谱混合检索；DocumentAgent负责拆分子问题、机构业务过滤、heuristic+API混合精排与生成式回答；"
            "平台层用FastAPI封装JWT鉴权、会话落库与答案反馈，配套MemoryGuard内存水位防护、推理并发闸门、模型profile注册与eval_gate持续评测门禁，"
            "前端用Vue3问答台+运维看板交付；针对OOM、Windows路径兼容、跨机构错答等问题，设计分批建索引、演示精简模式、多级生成降级，"
            "完成从知识库到企业可观测问答服务的全链路开发。"
        ),
        "金融内部手册文档格式混杂，包含PDF、Word、扫描截图表格等多类型文件；基础文本检索容易出现跨机构信息混淆、客服号码匹配错误、多意图问题回答不全，且金融场景要求答案必须附带原文页码、文档来源可追溯，不能输出无依据内容。我搭建一套多模态RAG应用平台，实现混合文档解析、图文多路检索、业务规则过滤、答案溯源，同时适配低配本地电脑运行，支持批量评测与交互式Web演示。": (
            "金融内部手册文档格式混杂，包含PDF、Word、扫描截图表格等多类型文件；基础文本检索容易出现跨机构信息混淆、客服号码匹配错误、"
            "多意图问题回答不全，且金融场景要求答案必须附带原文页码、文档来源可追溯，不能输出无依据内容。我搭建一套多模态RAG应用平台，"
            "实现混合文档解析、图文多路检索、业务规则过滤、答案溯源；在竞赛/演示版本之上，进一步补齐JWT鉴权与会话落库、MemoryGuard内存防护、"
            "模型profile迭代与eval_gate评测门禁，形成可观测、可降级、可持续迭代的企业问答服务。同时适配低配本地电脑运行，"
            "支持批量评测、Vue3交互问答与运维看板。"
        ),
        "多格式金融原始文档 → 自研DocIR结构化解析层（统一提取文档、页码、文本块、图片资源，保留完整溯源信息）\n→ 双路检索：BGE文本向量检索 + ColPali页级图像检索\n→ RRF算法融合多路检索得分\n→ DocumentAgent子问题拆分模块，多意图问题拆解为独立子查询并行检索\n→ 机构业务过滤 + CrossEncoder精排过滤低相关内容\n→ 大模型生成答案，附带检索证据、置信度、原文页码来源\n→ 双运行模式：①全量批量评测模式 ②轻量化Web演示模式（适配低内存设备）\n→ 异常降级：模型生成失败自动切换抽取式模板答案，保证页面不空白": (
            "多格式金融原始文档 → 自研DocIR结构化解析层（PlainText/Word/PDF+表格+Tesseract OCR/Image，统一提取文档、页码、文本块、图片资源）\n"
            "→ 离线构建知识图谱三元组（kg_triples.json，可选Neo4j）\n"
            "→ 双路向量检索：BGE+FAISS文本检索 + ColPali/CLIP/VisRAG页级图像检索（失败回落文本代理）\n"
            "→ RRF算法融合文本/图像得分 → HybridRetriever叠加图谱通道做混合召回\n"
            "→ DocumentAgent：QueryPlanner拆分子问题 → 机构业务过滤/电话专属通道 → EvidenceReranker混合精排（简单题heuristic机构加权，难例走DeepSeek API；已移除本地CrossEncoder防OOM）\n"
            "→ GenerativeAnswerer生成答案（DeepSeek API优先，本地HF次之，抽取模板兜底），AnswerSynthesizer输出证据、置信度、页码来源\n"
            "→ 三形态交付：①main.py批量评测 ②web_app.py轻量化演示 ③app.main企业平台（JWT会话+Vue3问答台+运维看板）\n"
            "→ 运行时防护：MemoryGuard三档水位 + MAX_CONCURRENT_INFER并发闸门 + 生成模型空闲自动卸载\n"
            "→ 模型迭代：configs/model_registry.json profile切换 + scripts/eval_gate.py门禁晋级\n"
            "→ 异常降级：生成失败/超时/乱码自动切换抽取式模板，保证接口不空白"
        ),
        "这个金融多模态RAG项目主打AI垂直应用工程落地，自研DocIR结构保证金融场景答案可溯源，同时搭建文本+图像两路检索，用RRF融合结果、CrossEncoder精排提升检索精度。落地时重点解决了本地内存溢出、Windows中文路径兼容、跨机构信息混淆三大工程难题，通过分批建索引、轻量化演示模式、业务硬过滤完成优化，同时设计生成降级机制保障系统可用性，完整走完知识库搭建、多模态检索、后端接口、交互演示整套AI应用交付流程。": (
            "这个金融多模态RAG项目主打AI垂直应用工程落地，自研DocIR保证答案可溯源，文本+图像双路检索经RRF融合，再叠加可选图谱通道与heuristic+API混合精排。"
            "平台侧用FastAPI+JWT+Vue3问答台+运维看板交付，MemoryGuard和eval_gate解决OOM与模型迭代失控。"
            "落地重点攻克内存溢出、Windows中文路径、跨机构错答、并发叠载四类工程问题，通过分批建索引、演示精简模式、机构硬过滤、多级生成降级完成优化，"
            "完整走完知识库—检索—Agent—企业API—前端演示全链路。"
        ),
        "双塔适合粗召回（BGE/ColPali）；CrossEncoder适合精排过滤低相关块。我在金融项目中「召回用双塔、精排用CrossEncoder」。": (
            "双塔适合粗召回（BGE/ColPali/图谱倒排）；CrossEncoder/query-doc交互精排精度高但本地易OOM。"
            "我在金融项目中「召回用双塔+RRF+可选图谱，精排用heuristic机构加权+难例DeepSeek API rerank」，已移除本地CrossEncoder，用hybrid精排兼顾精度与资源。"
        ),
        "检索层过滤噪声+精排；Agent层grounded校验并同证据重试；业务层机构过滤与拒答；证据不足抽取模板/拒答兜底。": (
            "检索层RRF融合+机构过滤+hybrid精排；生成层DeepSeek API/本地模型+抽取模板多级降级；"
            "业务层电话专属通道与低置信度提示；平台层MemoryGuard防叠载、eval_gate防劣化上线。"
        ),
        "能独立完成FastAPI异步接口、SSE、基础HTML展示与联调。短板是复杂Vue/React工程化，不影响AI服务交付主链路。": (
            "能独立完成FastAPI REST接口、JWT鉴权、Vue3问答台与运维看板联调；Agentic项目用SSE做步骤可视化。"
            "复杂前端组件库与大型React工程化仍在持续加强，不影响AI服务交付主链路。"
        ),
        "①对齐红线（幻觉/溯源/外网/纠错）；②知识工程DocIR；③MVP检索+生成+溯源；④加打分与幻觉门控；⑤FastAPI+流式演示+降级；⑥黄金集持续迭代。封闭库默认关外网。": (
            "①对齐红线（幻觉/溯源/外网/纠错）；②知识工程DocIR+离线KG；③MVP检索+生成+溯源；④加机构过滤与hybrid精排；"
            "⑤FastAPI+JWT会话+Vue3演示+MemoryGuard；⑥eval_gate黄金集门禁持续迭代。封闭库默认关外网。"
        ),
        "金融：内存溢出+跨机构错答→分批索引/精简模式+机构过滤与关键词通道。Agentic：LangChain破窗+门控过激导致过度外网→Prompt本地化、锁依赖、可配置阈值与路径监控。": (
            "金融：全量向量化OOM+并发叠载→分批FAISS/MemoryGuard/并发闸门/空闲卸载；CrossEncoder OOM→hybrid精排；"
            "跨机构错答→机构硬过滤+电话通道。Agentic：LangChain破窗+门控过激→Prompt本地化、锁依赖、可配置阈值与路径监控。"
        ),
        "金融RAG核心：分批索引抗OOM；RRF融合；机构过滤防串行；多级生成降级；演示精简模式。": (
            "金融RAG核心：DocIR溯源；BGE+ColPali/CLIP双塔+RRF+可选KG；hybrid精排；MemoryGuard+并发闸门；JWT+Vue3+eval_gate；分批索引与演示精简模式。"
        ),
        "配套评测脚本统计成功率、时延、证据召回；竞赛奖项背书。": (
            "配套评测脚本统计成功率、时延、证据召回；企业平台EvalRun落库+eval_gate门禁（成功率≥75%、证据覆盖≥70%等阈值）；竞赛奖项背书。"
        ),
    }

    prefix_replacements = {
        "RAG全称检索增强生成，核心是先检索外部私有知识，再约束大模型基于真实资料生成，缓解幻觉。普通线性RAG只有「检索—生成」；我的金融多模态RAG增加图文多路检索、业务过滤、答案溯源；Agentic RAG增加路由、文档打分、双重校验与自动重试闭环，形成带质量管控的智能体RAG。": (
            "RAG全称检索增强生成，核心是先检索外部私有知识，再约束大模型基于真实资料生成，缓解幻觉。普通线性RAG只有「检索—生成」；"
            "我的金融多模态RAG增加DocIR溯源、图文双塔+RRF+可选图谱、机构过滤、hybrid精排、MemoryGuard与eval_gate等企业工程层；"
            "Agentic RAG增加路由、文档打分、双重校验与自动重试闭环，形成带质量管控的智能体RAG。"
        ),
        "检索相关性/证据覆盖；生成幻觉率、完整度、意图匹配、业务错误；服务时延、降级比例、可用性；Agent再加路由占比、重试次数、循环超限。用黄金集+pytest批量回归。": (
            "检索相关性/证据覆盖/图谱命中；生成幻觉率、完整度、意图匹配、业务错答；服务时延、MemoryGuard触发率、降级比例；"
            "模型迭代看eval_gate门禁通过率。Agent项目再加路由占比、重试次数、循环超限。用黄金集+pytest+eval_gate批量回归。"
        ),
    }

    for p in doc.paragraphs:
        if p.text in replacements:
            p.text = replacements[p.text]
        for prefix, new_text in prefix_replacements.items():
            if p.text.startswith(prefix[:30]):
                p.text = new_text

    new_diffs = [
        "难点6：多请求并发推理导致模型叠载、再次触发OOM",
        "现象：第二个用户提问时进程内存陡增，偶发被系统Kill。",
        "解决：QARuntime推理Semaphore（MAX_CONCURRENT_INFER=1）；MemoryGuard warn/soft/hard三档水位，soft拒加载新模型、hard拒推理；生成模型空闲GEN_IDLE_UNLOAD_SEC后自动卸载；Docker mem_limit与应用水位双保险。",
        "收获：AI服务上线必须把内存当一等公民，容器限内存+应用内防护缺一不可。",
        "难点7：本地CrossEncoder精排加载即OOM，精排环节成为瓶颈",
        "现象：全量检索后加载CrossEncoder，Windows本机直接MemoryError。",
        "解决：移除本地CrossEncoder，改为EvidenceReranker：简单题heuristic机构/关键词加权，多意图/长问/排查类难例走DeepSeek API rerank；无Key时全程heuristic。",
        "收获：精排不必绑死CrossEncoder；按题型分流API精排，是资源受限下的务实方案。",
        "难点8：模型/检索配置迭代缺少回归门禁，改参数易 silently 劣化",
        "解决：configs/model_registry.json维护stable/candidate profile；scripts/eval_gate.py跑黄金集，校验成功率/时延/证据覆盖/置信度，失败exit 2禁止晋级，通过可--promote切换生产。",
        "收获：RAG上线后迭代必须评测门禁化，和CI/CD一样可自动化阻断。",
    ]

    inserted = False
    already_has_new = any("难点6：多请求并发推理" in p.text for p in doc.paragraphs)
    if not already_has_new:
        for p in doc.paragraphs:
            if p.text.strip() == "收获：线上必须有多级降级链路。" and not inserted:
                parent = p._element.getparent()
                ref = p._element
                for line in new_diffs:
                    new_p = OxmlElement("w:p")
                    parent.insert(parent.index(ref) + 1, new_p)
                    ref = new_p
                    para = Paragraph(new_p, p._parent)
                    para.text = line
                inserted = True
                break

    if len(doc.tables) >= 2:
        t = doc.tables[1]
        already_has_kg = any("图谱检索" in row.cells[0].text for row in t.rows)
        table_updates = {
            "精排": [
                "精排",
                "heuristic+API混合精排",
                "简单题本地heuristic机构加权，难例DeepSeek API rerank；已移除本地CrossEncoder防OOM",
                "本地CrossEncoder：精度高但加载即OOM，不适合低配单机",
            ],
            "生成兜底": [
                "生成兜底",
                "DeepSeek API+本地小模型+抽取模板",
                "有Key优先云端生成省内存；断网/限流回落本地HF；再不行抽取模板",
                "只靠本地小模型：输出不稳；只靠云端：断网即不可用",
            ],
            "Web适配": [
                "Web/前端",
                "Vue3问答台+运维看板+演示精简开关",
                "JWT会话/反馈/内存看板/模型profile/评测门禁；演示模式关图像塔、缩块数",
                "仅静态HTML：缺鉴权与运维；全量加载：低配OOM",
            ],
        }
        for row in t.rows:
            key = row.cells[0].text.strip()
            for k, vals in table_updates.items():
                if key.startswith(k) or key == k:
                    for i, v in enumerate(vals):
                        row.cells[i].text = v

        if not already_has_kg:
            for vals in [
                [
                    "图谱检索（可选）",
                    "离线KG+Neo4jGraphRetriever",
                    "实体关系补召回，Neo4j不可用时读kg_triples.json",
                    "强依赖在线图数据库：部署重，演示离线差",
                ],
                [
                    "企业服务层",
                    "FastAPI+JWT+SQLite",
                    "鉴权、会话/消息落库、答案反馈、Ops API",
                    "无鉴权裸接口：无法多用户与审计",
                ],
                [
                    "内存与并发防护",
                    "MemoryGuard+QARuntime",
                    "三档水位+推理Semaphore+生成模型空闲卸载",
                    "无限并发：多请求叠载必OOM",
                ],
                [
                    "模型迭代",
                    "model_registry+eval_gate",
                    "profile候选评测，门禁通过才晋级",
                    "手工改配置：无回归，线上 silently 劣化",
                ],
            ]:
                new_row = t.add_row()
                for i, v in enumerate(vals):
                    new_row.cells[i].text = v

    saved = None
    last_err: Exception | None = None
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(path))
            saved = path
            break
        except PermissionError as e:
            last_err = e
            continue
    if saved is None:
        raise PermissionError(f"无法写入 Word 文档，请关闭已打开的 docx 后重试。最后错误: {last_err}")
    print(f"Updated: {saved}")


if __name__ == "__main__":
    main()
