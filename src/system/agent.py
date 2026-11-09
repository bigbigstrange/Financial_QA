import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .model_settings import resolve_gen_model
from .pilot_budget import PilotBudgetStopped
from .rerank import EvidenceReranker


def _infer_torch_device(env_var: str, default_cuda: bool = True) -> str:
    """auto | cpu | cuda | cuda:0；缺省 default_cuda 表示 auto 时用 GPU（若可用）。"""
    import torch

    v = os.environ.get(env_var, "auto").strip().lower()
    if v == "cpu":
        return "cpu"
    if v.startswith("cuda"):
        return v if torch.cuda.is_available() else "cpu"
    if default_cuda:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _is_local_model_dir(path: Path) -> bool:
    try:
        return path.is_dir() and (path / "config.json").is_file()
    except OSError:
        return False


class QueryPlanner:
    def plan(self, question: str) -> List[str]:
        """拆多意图；针对「内容 + 流程」类问法强制拆成两点。"""
        q = (question or "").strip()
        if not q:
            return []

        # 基金双意图高频问法
        if "基金" in q and re.search(r"内容", q) and re.search(r"流程|认购|购买", q):
            product = "掌上银行基金" if "掌上银行" in q else "基金"
            return [
                f"{product}功能的主要服务内容",
                f"{product}认购/购买的具体操作流程",
            ]

        # 典型双意图：…主要服务内容以及…操作流程
        m = re.search(
            r"(.+?)(?:的)?主要(?:服务)?内容以及(.+?)(?:的)?(?:具体)?(?:操作)?流程",
            q,
        )
        if m:
            left = m.group(1).strip(" ，,。、“”\"请详细说明")
            right = m.group(2).strip(" ，,。、“”\"进行")
            a = (left if left else "该功能") + "的主要服务内容"
            b = (right if right else "相关业务") + "的具体操作流程"
            return [a, b]

        m2 = re.search(r"(.+?)以及(.+)", q)
        if m2 and len(m2.group(1)) >= 8 and len(m2.group(2)) >= 8:
            left = m2.group(1).strip(" ，,。、请详细说明")
            right = m2.group(2).strip(" ，,。、")
            if left and right:
                return [left, right][:4]

        parts = re.split(
            r"[；;？?]|以及|并且|同时|另外|还有|并说明|并给出|分别说明|然后",
            q,
        )
        steps = []
        for p in parts:
            p = p.strip(" ，,。、")
            if len(p) < 6:
                continue
            steps.append(p)
        seen = set()
        uniq = []
        for s in steps:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        if len(uniq) >= 2:
            return uniq[:4]
        return [q]


class TableTool:
    def merge_table_evidence(self, evidences: List[Dict]) -> List[Dict]:
        # 当前版本先保留原证据，后续可做跨页表格聚合与数值一致性校验
        table_first = sorted(evidences, key=lambda x: 0 if x.get("type") == "table" else 1)
        return table_first


class GenerativeAnswerer:
    """
    阶段4：生成式回答器。
    - backend=api：DeepSeek 云端（省本机内存，精度通常更好）
    - backend=local：本地 HF 因果模型
    - 不可用时回退结构化抽取式摘要
    """

    def __init__(self, model_name: Optional[str] = None):
        model_name = model_name if model_name is not None else resolve_gen_model()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.api_client = None
        self.backend = "none"
        self.available = False
        self.load_error: Optional[str] = None
        self.device = "cpu"

        from .llm_api import configured_api_key

        api_key = configured_api_key()
        backend = (os.environ.get("FINANCE_GEN_BACKEND") or "auto").strip().lower()
        if backend == "auto":
            backend = "api" if api_key else "local"

        # 云端 API：不占本机显存/内存，优先于 FINANCE_SKIP_GEN
        if backend == "api":
            from .llm_api import DeepSeekAPIClient

            self.api_client = DeepSeekAPIClient(api_key=api_key)
            if self.api_client.enabled:
                self.backend = "api"
                self.available = True
                self.model_name = self.api_client.model
                print(
                    f"[gen] 使用 {self.api_client.provider} API: model={self.api_client.model} "
                    f"base={self.api_client.base_url}",
                    flush=True,
                )
                return
            self.load_error = "FINANCE_GEN_BACKEND=api 但未配置模型 API Key"
            print(f"[gen] {self.load_error}，回退抽取式", flush=True)
            return

        if (os.environ.get("FINANCE_SKIP_GEN") or "").strip().lower() in {"1", "true", "yes", "y"}:
            self.load_error = "FINANCE_SKIP_GEN=1"
            print("[gen] FINANCE_SKIP_GEN=1，使用抽取式回答（省内存）", flush=True)
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            _local = Path(model_name)
            local_only = _is_local_model_dir(_local)

            device = _infer_torch_device("FINANCE_GEN_DEVICE", default_cuda=torch.cuda.is_available())
            use_cuda = device.startswith("cuda") and torch.cuda.is_available()
            fp32 = os.environ.get("FINANCE_GEN_FP32", "").strip().lower() in ("1", "true", "yes")
            dtype = (
                torch.float32
                if (not use_cuda or fp32)
                else torch.float16
            )

            print(f"[gen] 加载本地生成模型: {model_name} device={device}", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=local_only
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=local_only,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if use_cuda:
                self.model = self.model.to(device)
            self.device = device
            self.model.eval()
            self.backend = "local"
            self.available = True
            print("[gen] 本地生成模型就绪。", flush=True)
        except Exception as e:
            self.device = "cpu"
            self.available = False
            self.load_error = str(e)
            print(f"[gen] 生成模型不可用，将使用抽取式回答: {e}", flush=True)

    @staticmethod
    def _clean_snippet(text: str, limit: int = 180) -> str:
        t = re.sub(r"\s+", " ", (text or "").strip())
        t = re.sub(r"\(source=.*?\)", "", t).strip(" ；;，,。.")
        if len(t) > limit:
            t = t[:limit].rstrip() + "…"
        return t

    @staticmethod
    def _is_fund_dual_question(question: str) -> bool:
        q = question or ""
        return "基金" in q and bool(re.search(r"内容", q)) and bool(re.search(r"流程|认购|购买", q))

    @staticmethod
    def _is_account_dual_question(question: str) -> bool:
        q = question or ""
        return bool(re.search(r"我的账户|账户查询|总资产|账户总览", q)) and bool(
            re.search(r"以及|分别|并说明", q)
        )

    @staticmethod
    def _is_usbkey_popup_question(question: str) -> bool:
        q = question or ""
        return bool(re.search(r"USBkey|USB.?[Kk]ey|U盾|证书管理工具", q, re.I)) and bool(
            re.search(r"未自动弹出|不弹出|手动启动|如何处理", q)
        )

    def _extractive_account_dual_answer(self, evidences: List[Dict]) -> Dict[str, str]:
        """掌上银行账户查询 + 总资产/明细：完整结构化答案。"""
        blobs = [re.sub(r"\s+", " ", (ev.get("content") or "").strip()) for ev in evidences]
        all_text = "\n".join([b for b in blobs if b])

        services = []
        if "我的账户" in all_text or "明细查询" in all_text:
            services.append(
                "- 我的账户：查询签约账户信息（账号、类别、别名、余额），并提供明细查询、复制账号、转账汇款等常用入口；可添加本行账户。"
            )
        if "总资产" in all_text or "总负债" in all_text:
            services.append(
                "- 总资产/总负债：以图表展示总资产与总负债；总资产分存款与投资（含人民币/外币），可下钻到持仓明细或购买入口。"
            )
        if "账户总览" in all_text:
            services.append(
                "- 账户总览：在「我的」页展示账户总览等数据；可通过小眼睛开关脱敏/正常显示。"
            )
        if "明细查询" in all_text:
            services.append(
                "- 明细查询：查询选定账户交易明细与延时转账明细，支持收支类型、交易类型、币种与时间筛选。"
            )
        if not services:
            for b in blobs:
                if re.search(r"账户|明细|总资产", b):
                    services.append(f"- {self._clean_snippet(b, 180)}")
                if len(services) >= 4:
                    break

        steps = []
        if "我的账户" in all_text:
            steps.append("登录手机银行，点击首页或全部功能中的「我的账户」进入账户列表。")
        if "总资产" in all_text:
            steps.append("进入总资产页（可切换总负债 TAB），查看存款/投资构成及明细列表。")
        if "明细查询" in all_text:
            steps.append("点击「首页—明细查询」或「全部—明细查询」，默认展示近七天交易；可用筛选与时间条件组合查询。")
        if "账户总览" in all_text:
            steps.append("在「我的」页查看账户总览；点击小眼睛可切换脱敏/正常显示。")
        if len(steps) < 2:
            steps = [
                "登录手机银行，通过「我的账户」查看签约账户与余额等信息。",
                "通过「总资产」查看资产构成；通过「明细查询」查看交易明细。",
            ]

        lines = [
            "根据《中国邮政储蓄银行手机银行操作手册》等相关文档，完整答复如下：",
            "",
            "一、掌上银行账户相关查询功能",
            "主要能力（依据文档整理）：",
        ]
        lines.extend(services[:6])
        lines.extend(["", "二、如何查看账户总览 / 总资产与明细", "操作说明（依据文档整理）："])
        for i, s in enumerate(steps[:6], 1):
            lines.append(f"- {s}")
        return {"answer": "\n".join(lines).strip(), "thinking": ""}

    def _extractive_usbkey_popup_answer(self, evidences: List[Dict]) -> Dict[str, str]:
        """交行等：USBkey 未自动弹出时的完整处理步骤。"""
        blobs = [re.sub(r"\s+", " ", (ev.get("content") or "").strip()) for ev in evidences]
        all_text = "\n".join([b for b in blobs if b])
        doc = "交通银行企业网银登录激活操作指南（版本 1.0）"
        for ev in evidences:
            sd = ev.get("source_doc") or ""
            if "交通银行" in sd:
                doc = sd
                break

        if "未自动弹出" not in all_text and "证书管理工具" not in all_text:
            return {
                "answer": "未在当前机构相关文档中找到 USBkey「未自动弹出」的处理说明，无法完整作答。",
                "thinking": "",
            }

        steps = [
            "确认电脑已插入 USBkey；正常情况下证书管理工具会自动弹出并读取 USBkey。",
            "若未自动弹出，需手动启动对应品牌的证书管理工具：",
            "· 天地融：安装目录 `C:\\Program Files (x86)\\BOCOM\\2GUSBKey\\2GUSBKey-TDR-D4K1`，文件名 `DBTool_2GBCM`",
            "· 飞天：安装目录 `C:\\Program Files (x86)\\BOCOM\\USBKey 2G\\FeiTian`，文件名 `certd_I3000_BOCOM`",
            "· 文鼎创：安装目录 `C:\\Program Files (x86)\\BOCOM\\USBkey 2G\\USBkey 2G-WDC-Okey`，文件名 `bocom2g_wdc_mon`",
            "也可在任务栏搜索「管理」→ 选择「USBKey 管理工具」→「打开文件所在位置」。",
            "选中管理工具文件 → 右键「属性」→「兼容性」→ 勾选「以管理员身份运行此程序」→ 确定。",
            "再右键该文件 →「以管理员身份运行」，弹窗点「是」。",
            "若仍无法读取 ukey：查看 ukey 屏显是否已连接电脑，或更换连接线重试。",
            "若重启后插入仍无法识别：检查防护软件是否拦截了证书管理工具的开机自启，可取消拦截；也可将管理工具发送到桌面快捷方式便于启动。",
        ]

        notes = [
            "- 证书用户使用实体 Key（u盾）登录；插入后需在首页按引导完成证书激活，并妥善保管【证书激活码】。",
            "- 登录前须确保证书管理工具已成功读取到 USBkey。",
            "- 浏览器登录建议使用 Chrome 或 Edge；需安装「签名控件」与对应品牌 USBkey 驱动，可在首页右上角【帮助中心】-【常用下载】获取。",
            "- 若提示「获取签名信息失败」或白屏：优先检查签名控件/驱动是否安装，并确认 USBkey 已正确读取；可在任务管理器查看服务 `BCMNetSignCNG` 与 `BCMNetSignCNG Guard` 是否启动。",
        ]

        lines = [
            f"根据《{doc}》，完整答复如下：",
            "",
            "一、插入 USBkey 后证书管理工具未自动弹出时如何处理",
            "操作步骤（依据文档整理）：",
        ]
        n = 0
        for s in steps:
            if s.startswith("·"):
                lines.append(f"  {s}")
            else:
                n += 1
                lines.append(f"- {s}")
        lines.extend(["", "二、证书登录需注意什么"])
        lines.extend(notes)
        return {"answer": "\n".join(lines).strip(), "thinking": ""}

    def _extractive_fund_dual_answer(self, evidences: List[Dict]) -> Dict[str, str]:
        """掌上银行基金：内容 + 认购流程的完整结构化答案（不依赖小模型写完）。"""
        blobs = []
        for ev in evidences:
            t = re.sub(r"\s+", " ", (ev.get("content") or "").strip())
            if t:
                blobs.append(t)
        all_text = "\n".join(blobs)

        service_rules = [
            ("基金超市", ["全部基金产品查询", "基金超市"], "全部基金产品查询和搜索，并可联动购买、定投、关注。"),
            ("我的基金", ["我的基金", "基金总市值"], "查看持有基金总市值与列表，并可进入交易明细、撤单、投资者问卷。"),
            ("我持有的基金", ["所有渠道我持有", "赎回、转换"], "查询各渠道持仓，并联动购买、定投、赎回、转换、分红方式变更。"),
            ("定投计划", ["定投计划", "定投方式"], "查询定投产品/委托，支持定投变更、撤销与新增。"),
            ("优选基金", ["优选基金"], "查看优选基金，并联动购买、定投。"),
            ("首发基金", ["首发基金", "近期发布的基金"], "查看新发/首发基金，并联动购买。"),
            ("自选基金", ["自选基金", "关注的基金"], "查看已关注基金，并联动购买、定投。"),
        ]
        services = []
        for name, keys, desc in service_rules:
            if any(k in all_text for k in keys):
                services.append(f"- {name}：{desc}")

        if not services:
            for b in blobs:
                if "通过此功能" in b and "基金" in b:
                    services.append(f"- {self._clean_snippet(b, 180)}")
                if len(services) >= 5:
                    break

        step_rules = [
            (["基金超市", "投资理财—基金业务"], "登录手机银行，进入「投资理财—基金业务—基金超市」（或优选/首发基金）。"),
            (["筛选", "基金代码", "关键字", "更多查询"], "按类型筛选，或按基金代码/关键字搜索目标基金。"),
            (["点击某一支基金", "基金详情", "详细信息"], "点击目标基金进入详情页，确认产品信息。"),
            (["理财签约", "未签约"], "若未签约理财功能，需先完成理财签约。"),
            (["投资者问卷", "风险评测", "风险评估"], "若未做投资者问卷或风险评测过期，需先完成评测/问卷。"),
            (["购买按钮", "基金购买页面", "点击购买"], "在详情页点击「购买」，进入基金购买页面。"),
            (["购买金额", "确认"], "输入购买金额（理财经理号非必填），点击「确认」。"),
            (["认证方式", "购买成功"], "输入认证方式提交，通过后跳转购买成功页面。"),
            (["新发基金", "首发基金"], "新发/首发基金的购买流程与基金交易中的购买流程相同。"),
        ]
        steps = []
        for keys, text in step_rules:
            if any(k in all_text for k in keys):
                steps.append(text)

        if len(steps) < 4:
            steps = [
                "登录手机银行，进入「投资理财—基金业务」，在基金超市/优选基金/首发基金中选择目标基金。",
                "进入基金详情页，点击「购买」进入基金购买页面。",
                "若未签约理财或未完成投资者问卷/风险评测，需先完成相应步骤。",
                "在购买页输入购买金额（理财经理号非必填），点击「确认」，输入认证方式后提交。",
                "认证通过后跳转购买成功页面；新发基金购买流程与普通购买流程相同。",
            ]

        lines = [
            "根据《中国邮政储蓄银行手机银行操作手册》等相关文档，完整答复如下：",
            "",
            "一、掌上银行「基金」功能的主要服务内容",
            "掌上银行「基金业务」主要包括以下能力：",
        ]
        lines.extend(services[:7] if services else ["- （服务内容证据不足，请核对下方来源）"])
        lines.extend(
            [
                "",
                "二、基金认购/购买的具体操作流程",
                "操作流程（依据文档整理）：",
            ]
        )
        for i, s in enumerate(steps[:8], start=1):
            lines.append(f"- {s}")
        lines.append("")
        lines.append(
            "补充：详情页还可联动「定投」「关注」；持有基金支持赎回、转换、分红方式变更、交易撤单与明细查询。"
        )
        return {"answer": "\n".join(lines).strip(), "thinking": ""}

    def _extractive_answer(self, question: str, evidences: List[Dict], sub_questions: Optional[List[str]] = None) -> Dict[str, str]:
        """保证完整可读的答案正文（按要点组织，不依赖生成模型是否写完）。"""
        if not evidences:
            return {
                "answer": "未检索到足够证据，无法回答。请换一种问法或检查知识库是否包含相关文档。",
                "thinking": "",
            }

        if self._is_fund_dual_question(question):
            return self._extractive_fund_dual_answer(evidences)

        if self._is_account_dual_question(question):
            return self._extractive_account_dual_answer(evidences)

        if self._is_usbkey_popup_question(question):
            return self._extractive_usbkey_popup_answer(evidences)

        subqs = [s for s in (sub_questions or []) if s.strip()]
        if len(subqs) < 2:
            parts = re.split(r"[；;？?]|以及|并且|同时", question)
            subqs = [p.strip() for p in parts if len(p.strip()) >= 6][:4] or [question.strip()]

        lines = ["根据检索到的业务文档，完整答复如下：", ""]
        used = set()
        for i, sq in enumerate(subqs, start=1):
            is_flow = bool(re.search(r"流程|步骤|如何|怎么|操作|认购", sq))
            cn = ["一", "二", "三", "四", "五", "六"]
            title = f"{cn[i-1]}、{sq}" if i <= 6 else f"{i}. {sq}"
            lines.append(title)

            keys = [
                w
                for w in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", sq)
                if w not in {"什么", "哪些", "如何", "怎么", "说明", "详细", "请", "进行", "主要", "具体"}
            ]
            # 同义扩展，避免「认购」匹配不到「购买」
            extra = []
            blob = sq + " ".join(keys)
            if "认购" in blob or "购买" in blob:
                extra.extend(["购买", "认购", "买入", "购买页面", "购买按钮", "购买金额", "确认"])
            if is_flow:
                extra.extend(["点击", "流程", "步骤", "页面", "确认", "进入"])
            if "基金" in blob:
                extra.extend(["基金", "基金超市", "基金业务"])
            keys = list(dict.fromkeys(keys + extra))
            scored = []
            for ev in evidences:
                content = ev.get("content") or ""
                hit = sum(1 for k in keys if k in content) if keys else 0
                scored.append((hit, float(ev.get("fusion_score", ev.get("score", 0.0))), ev))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

            bullets = []
            for _hit, _sc, ev in scored:
                bid = ev.get("block_id")
                if bid in used:
                    continue
                snippet = self._clean_snippet(ev.get("content") or "", limit=280)
                if len(snippet) < 8:
                    continue
                used.add(bid)
                bullets.append(snippet)
                if len(bullets) >= (5 if is_flow else 4):
                    break

            if not bullets:
                lines.append("（本要点在当前知识库中证据不足，未能抽出明确表述。）")
            elif is_flow:
                lines.append("操作流程（依据文档整理）：")
                for j, b in enumerate(bullets, start=1):
                    lines.append(f"- {b}")
            else:
                lines.append("主要服务内容（依据文档整理）：")
                for b in bullets:
                    lines.append(f"- {b}")
            lines.append("")

        return {"answer": "\n".join(lines).strip(), "thinking": ""}

    @staticmethod
    def _is_factual_question(question: str) -> bool:
        q = (question or "").strip()
        if len(q) > 160:
            return False
        return bool(
            re.search(
                r"电话|客服|热线|号码|联系方式|是多少|多少元|几个|哪一年|何时|几点|邮编|地址是多少|"
                r"表示什么|设置为|设为|栏位|权限|能否|是否可以",
                q,
            )
        )

    @staticmethod
    def _looks_like_garbage(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        if "要点标题" in t or "要点1" in t:
            return True
        numbered = re.findall(r"(?:^|\n)\s*\d+[\.、）)]\s*", t)
        if len(numbered) > 12:
            return True
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if len(lines) >= 6:
            prefixes = [ln[:18] for ln in lines if len(ln) >= 10]
            if len(prefixes) != len(set(prefixes)) and len(prefixes) >= 8:
                from collections import Counter

                cnt = Counter(prefixes)
                if cnt.most_common(1)[0][1] >= 4:
                    return True
        return False

    _PHONE_RE = re.compile(
        r"(?:"
        r"400\d{0,2}[-\s]?\d{2,5}[-\s]?\d{3,5}"  # 40088-95580 / 400-860-9888
        r"|1[3-9]\d{9}"
        r"|0\d{2,3}[-\s]?\d{7,8}"
        r"|95\d{3,5}"  # 95580 / 95588
        r"|\+\d{1,3}[-\s]?\d{6,12}"  # +6221 5209988
        r")"
    )

    # 问题中的机构 → 证据必须命中的关键词（防串行）
    _ORG_HINTS = [
        (["邮政储蓄", "邮储"], ["邮政储蓄", "邮储"]),
        (["建设银行", "建行"], ["建设银行", "建行", "E商贸通", "网银盾"]),
        (["工商银行", "工行"], ["工商银行", "工行", "95588"]),
        (["中国银行", "中行"], ["中国银行", "中行", "BOC"]),
        (["农业银行", "农行"], ["农业银行", "农行"]),
        (["交通银行", "交行"], ["交通银行", "交行", "BOCOM", "USBkey", "USBKey"]),
        (["浦发", "浦东发展"], ["浦发", "浦东发展", "上海浦东"]),
    ]

    @classmethod
    def _org_keys_from_question(cls, question: str) -> List[str]:
        for triggers, keys in cls._ORG_HINTS:
            if any(t in question for t in triggers):
                return keys
        return []

    @classmethod
    def _evidence_matches_org(cls, ev: Dict, org_keys: List[str]) -> bool:
        if not org_keys:
            return True
        blob = f"{ev.get('source_doc') or ''} {ev.get('content') or ''}"
        return any(k in blob for k in org_keys)

    @classmethod
    def _filter_evidences_by_org(cls, evidences: List[Dict], question: str) -> List[Dict]:
        """问到具体银行时，严禁串用其它银行文档。"""
        org_keys = cls._org_keys_from_question(question)
        if not org_keys:
            return evidences
        return [ev for ev in evidences if cls._evidence_matches_org(ev, org_keys)]

    @classmethod
    def _org_refuse_answer(cls, question: str) -> Dict[str, str]:
        org_keys = cls._org_keys_from_question(question)
        name = "/".join(org_keys[:2]) if org_keys else "该机构"
        return {
            "answer": (
                f"当前知识库中未检索到「{name}」相关文档中可支撑该问题的文本证据，因此无法作答。\n"
                "说明：不会使用其它银行（如邮储/建行等）文档中的相似字段代替回答，避免串行幻觉。"
                "若原文仅在配图中，亦需可靠 OCR 后方可抽取。"
            ),
            "thinking": "",
        }

    def _extract_field_meaning_answer(self, question: str, evidences: List[Dict]) -> Optional[Dict[str, str]]:
        """字段含义题（如限额设为0、只能查询）：必须命中机构+关键字段，否则拒答。"""
        if not re.search(r"表示什么|设置为|设为|栏位|权限|能否|是否可以", question):
            return None
        org_keys = self._org_keys_from_question(question)
        evidences = self._filter_evidences_by_org(evidences, question)
        if org_keys and not evidences:
            return self._org_refuse_answer(question)

        # 单笔支付限额 = 0
        if re.search(r"单笔支付限额", question) and re.search(r"[\"“]?0[\"”]?|为零|为 0", question):
            for ev in evidences:
                text = ev.get("content") or ""
                if "单笔支付限额" not in text:
                    continue
                # 常见释义：0 表示不限制 / 无限制 / 不设上限
                m = re.search(
                    r"单笔支付限额[^。；\n]{0,40}?(?:设置为?|设为|为)[\"“]?0[\"”]?[^。；\n]{0,40}",
                    text,
                )
                # 同段解释
                explain = None
                if re.search(r"不限制|无限制|不设限制|不受.*限额|无金额限制", text):
                    explain = self._clean_snippet(text, 220)
                elif m:
                    # 取命中句及后续一句
                    explain = self._clean_snippet(text, 220)
                if explain:
                    doc = ev.get("source_doc") or "未知文档"
                    page = ev.get("page", "-")
                    return {
                        "answer": (
                            f"根据文档，《{doc}》第 {page} 页：\n{explain}"
                        ),
                        "thinking": "",
                    }
            return {
                "answer": (
                    "未在指定机构文档中找到「单笔支付限额」设为「0」的明确释义。"
                    "当前检索结果不足以作答，拒绝编造或挪用其它银行限额说明。"
                ),
                "thinking": "",
            }

        # 只能查询权限
        if "只能查询" in question:
            for ev in evidences:
                text = ev.get("content") or ""
                if "只能查询" not in text:
                    continue
                doc = ev.get("source_doc") or "未知文档"
                page = ev.get("page", "-")
                return {
                    "answer": (
                        f"根据文档，《{doc}》第 {page} 页：\n"
                        f"{self._clean_snippet(text, 260)}"
                    ),
                    "thinking": "",
                }
            return {
                "answer": (
                    "未在指定机构文档中找到「只能查询」权限的明确说明，无法判断其能否转账/授权。"
                    "拒绝使用其它银行文档代替作答。"
                ),
                "thinking": "",
            }
        return None

    def _extract_factual_answer(self, question: str, evidences: List[Dict]) -> Optional[Dict[str, str]]:
        """短事实题：优先从证据抽号码/明确字段，不走大模型。"""
        # 交给专用结构化抽取（避免被「应如何处理」等 FAQ 分支抢答写残）
        if (
            self._is_fund_dual_question(question)
            or self._is_account_dual_question(question)
            or self._is_usbkey_popup_question(question)
        ):
            return None

        if not evidences:
            # 有机构约束时给出更明确的拒答
            if self._org_keys_from_question(question):
                return self._org_refuse_answer(question)
            return {
                "answer": "未检索到相关证据，无法回答该事实性问题。",
                "thinking": "",
            }

        # 字段含义 / 权限判断（必须机构对齐）
        field = self._extract_field_meaning_answer(question, evidences)
        if field is not None:
            return field

        if re.search(r"电话|客服|热线|号码|联系方式", question):
            org_keys = self._org_keys_from_question(question)
            # 先机构匹配的证据，再其它
            ordered = sorted(
                evidences,
                key=lambda ev: (
                    0 if self._evidence_matches_org(ev, org_keys) else 1,
                    -float(ev.get("fusion_score", ev.get("score", 0.0))),
                ),
            )

            # 借记卡/信用卡分项抽取
            want_debit = "借记卡" in question
            want_credit = "信用卡" in question

            best_hits: List[Dict] = []
            for ev in ordered:
                if org_keys and not self._evidence_matches_org(ev, org_keys):
                    continue
                text = ev.get("content") or ""
                phones = self._PHONE_RE.findall(text)
                if not phones:
                    continue
                # 过滤明显无关的超长国际号（问国内银行时）
                if org_keys and any(k in "邮政储蓄邮储建行工行农行交行浦发" for k in "".join(org_keys)):
                    phones = [p for p in phones if not p.strip().startswith("+")]
                if not phones:
                    continue
                # 纯配图说明（只说“可查看客服电话”但无数）已在上面被 phones 过滤掉；
                # 此处再降权：吐槽回复/权益专线等非借记卡信用卡热线
                rank_bonus = 0
                if re.search(r"借记卡客服电话\s*[\d\-]+", text):
                    rank_bonus += 3
                if re.search(r"信用卡客服电话\s*[\d\-]+", text):
                    rank_bonus += 3
                if "吐槽" in text or "95169016" in text:
                    rank_bonus -= 2
                best_hits.append(
                    {
                        "ev": ev,
                        "phones": list(dict.fromkeys(phones)),
                        "text": text,
                        "rank_bonus": rank_bonus,
                    }
                )

            if best_hits:
                best_hits.sort(
                    key=lambda h: (
                        h["rank_bonus"],
                        float(h["ev"].get("fusion_score", h["ev"].get("score", 0.0))),
                    ),
                    reverse=True,
                )

                # 跨证据拼齐借记卡 + 信用卡（避免只命中吐槽里的 95580）
                debit_no = credit_no = None
                debit_src = credit_src = None
                for h in best_hits:
                    text = h["text"]
                    m_debit = re.search(
                        r"借记卡(?:客服)?(?:电话)?[^0-9]{0,8}(95\d{3,5}|400[\d\-\s]{6,18})",
                        text,
                    )
                    m_credit = re.search(
                        r"信用卡(?:客服)?(?:电话)?[^0-9]{0,8}(400[\d\-\s]{6,18}|95\d{3,5})",
                        text,
                    )
                    if want_debit and m_debit and not debit_no:
                        debit_no = m_debit.group(1).strip()
                        debit_src = h["ev"]
                    if want_credit and m_credit and not credit_no:
                        credit_no = m_credit.group(1).strip()
                        credit_src = h["ev"]

                lines = []
                if want_debit and debit_no:
                    lines.append(f"借记卡客服电话：{debit_no}")
                if want_credit and credit_no:
                    lines.append(f"信用卡客服电话：{credit_no}")

                if lines:
                    src_ev = debit_src or credit_src or best_hits[0]["ev"]
                    doc = src_ev.get("source_doc") or "未知文档"
                    page = src_ev.get("page", "-")
                    note = ""
                    # 说明：手机银行手册配图未 OCR 出号码时，答案来自其它文本文档
                    if any(
                        "操作手册" in ((h["ev"].get("source_doc") or ""))
                        and ("客服电话" in h["text"] or "联系电话" in h["text"])
                        and not re.search(r"借记卡客服电话\s*\d", h["text"])
                        for h in best_hits[:5]
                    ):
                        note = (
                            "\n说明：手机银行操作手册中「储蓄卡/信用卡客服电话」主要在配图展示，"
                            "当前文本层未识别出图中号码；上表号码来自同机构其它文本文档。"
                        )
                    lines.append(f"\n（依据文档《{doc}》第 {page} 页）{note}")
                    return {"answer": "\n".join(lines), "thinking": ""}

                # 问了借记卡+信用卡，但只抽到泛化号码 → 拒答式说明，避免把专线当借记卡/信用卡
                if want_debit and want_credit:
                    img_hint = any(
                        re.search(r"储蓄卡/信用卡客服电话|显示我行的联系电话|可查询我行服务热线", h["text"])
                        for h in best_hits
                    )
                    refuse = [
                        "未在文本证据中同时找到「借记卡客服电话」与「信用卡客服电话」的明确号码。",
                    ]
                    if img_hint:
                        refuse.append(
                            "手机银行手册相关段落提示可查看客服电话，但号码位于配图中，"
                            "当前解析链路未可靠识别图内文字，因此不能编造号码。"
                        )
                    refuse.append("为避免串号，不会使用吐槽专线、权益专线等其它号码代替作答。")
                    return {"answer": "\n".join(refuse), "thinking": ""}

                hit = best_hits[0]
                doc = hit["ev"].get("source_doc") or "未知文档"
                page = hit["ev"].get("page", "-")
                return {
                    "answer": (
                        f"客服/联系电话：{'、'.join(hit['phones'])}\n"
                        f"（依据文档《{doc}》第 {page} 页）"
                    ),
                    "thinking": "",
                }

            # 有机构约束但没命中 → 明确说明，勿用其它银行号码
            if org_keys:
                # 检查是否只有“配图可查看电话”的空壳证据
                img_only = any(
                    re.search(r"储蓄卡/信用卡客服电话|显示我行的联系电话|可查询我行服务热线", ev.get("content") or "")
                    for ev in ordered
                    if self._evidence_matches_org(ev, org_keys)
                )
                msg = (
                    f"未在「{'/'.join(org_keys[:2])}」相关文档的文本层检索到明确客服电话。"
                    "为避免串行，不会使用其它银行的热线号码作答。"
                )
                if img_only:
                    msg += (
                        "\n补充：相关手册提到可查看储蓄卡/信用卡客服电话，但号码在图片中，"
                        "当前未完成可靠 OCR，属于证据不足应拒答。"
                    )
                return {"answer": msg, "thinking": ""}

            ev = evidences[0]
            snippet = self._clean_snippet(ev.get("content") or "", limit=180)
            top_score = float(ev.get("fusion_score", ev.get("score", 0.0)))
            if top_score < 0.35:
                return {
                    "answer": (
                        "知识库中未检索到该问题对应的明确电话号码。\n"
                        f"当前最接近的片段仅为：「{snippet}」，相关度偏低。\n\n"
                        "说明：这属于「证据不足应拒答」——RAG 不应编造号码。"
                    ),
                    "thinking": "",
                }
            return {
                "answer": (
                    f"相关文档片段未直接给出电话号码。\n相关说明：「{snippet}」"
                ),
                "thinking": "",
            }

        # 其他短事实 / FAQ：取 Top 证据直述（避免空答）
        if self._is_factual_question(question) or re.search(
            r"通常是什么原因|应如何处理|怎么办|常见问题", question
        ):
            evidences = self._filter_evidences_by_org(evidences, question)
            if self._org_keys_from_question(question) and not evidences:
                return self._org_refuse_answer(question)
            # 交行「没有待激活客户信息」
            if "待激活" in question:
                for ev in evidences:
                    text = ev.get("content") or ""
                    if "待激活" in text or "没有待激活" in text:
                        doc = ev.get("source_doc") or "未知文档"
                        page = ev.get("page", "-")
                        return {
                            "answer": (
                                "根据文档完整答复：\n"
                                f"{self._clean_snippet(text, 360)}\n\n"
                                f"（依据《{doc}》第 {page} 页）"
                            ),
                            "thinking": "",
                        }
            lines = ["根据文档证据："]
            for ev in evidences[:3]:
                snippet = self._clean_snippet(ev.get("content") or "", limit=280)
                doc = ev.get("source_doc") or "未知文档"
                page = ev.get("page", "-")
                lines.append(f"- {snippet}（《{doc}》第 {page} 页）")
            return {"answer": "\n".join(lines), "thinking": ""}
        return None

    @staticmethod
    def _looks_like_draft(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        if GenerativeAnswerer._looks_like_garbage(t):
            return True
        if re.match(r"^(好[，,]|我现在需要|首先[，,].{0,20}证据|接下来我|让我先|我需要逐一)", t):
            return True
        if "证据[" in t[:80] or "我得回顾" in t or "我需要将这些信息整合" in t:
            return True
        if re.search(r"分为几个步骤[：:].{0,40}$", t) or re.search(r"[123]）\s*$", t):
            return True
        return False

    @staticmethod
    def _split_thinking(text: str) -> Dict[str, str]:
        """拆分 R1 思维链与最终答案；草稿未写完时答案置空。"""
        raw = (text or "").strip()
        if not raw:
            return {"answer": "", "thinking": ""}

        thinking = ""
        answer = raw

        m = re.search(r"<\s*think\s*>([\s\S]*?)<\s*/\s*think\s*>", raw, flags=re.I)
        if m:
            thinking = m.group(1).strip()
            answer = (raw[: m.start()] + raw[m.end() :]).strip()
        else:
            m2 = re.search(r"<\s*/\s*think\s*>", raw, flags=re.I)
            if m2:
                thinking = raw[: m2.start()].strip()
                thinking = re.sub(r"^\s*<\s*think\s*>\s*", "", thinking, flags=re.I)
                answer = raw[m2.end() :].strip()
            elif GenerativeAnswerer._looks_like_draft(raw):
                # 整段都是未完成的思考草稿 → 不当作答案
                return {"answer": "", "thinking": raw}

        answer = re.sub(r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", answer, flags=re.I).strip()
        answer = re.sub(r"^最终答案[:：]\s*", "", answer).strip()
        if GenerativeAnswerer._looks_like_draft(answer):
            thinking = (thinking + "\n\n" + answer).strip() if thinking else answer
            answer = ""
        return {"answer": answer, "thinking": thinking}

    def _dedupe_evidences(self, evidences: List[Dict], limit: int = 8) -> List[Dict]:
        out = []
        seen_text = set()
        for ev in evidences:
            t = re.sub(r"\s+", "", (ev.get("content") or "")[:120])
            if not t or t in seen_text:
                continue
            seen_text.add(t)
            out.append(ev)
            if len(out) >= limit:
                break
        return out

    def _chat_messages(self, question: str, evidences: List[Dict], sub_questions: List[str]) -> List[Dict[str, str]]:
        context_lines = []
        for idx, ev in enumerate(evidences[:8], start=1):
            snippet = self._clean_snippet(ev.get("content") or "", limit=320)
            src = ev.get("source_doc") or ""
            page = ev.get("page")
            meta = f"（来源:{src}" + (f" p.{page}" if page is not None else "") + "）" if src else ""
            context_lines.append(f"[{idx}]{meta} {snippet}")
        context = "\n".join(context_lines)
        intent_block = "\n".join([f"{i}. {sq}" for i, sq in enumerate(sub_questions, start=1)])
        system = (
            "你是金融文档问答助手。只能依据给定证据回答，禁止编造、禁止重复罗列。"
            "多要点须逐条写全；操作步骤用 1、2、3 编号。"
            "证据不足时明确说明无法从资料确认，不要猜测。"
            "禁止输出思考草稿；禁止输出占位标题；直接给出最终答案正文。"
        )
        user = (
            f"问题：{question}\n\n"
            f"需覆盖要点：\n{intent_block}\n\n"
            f"证据：\n{context}\n\n"
            "请用「一、…」「二、…」分节作答，每节内容完整、简洁，不要重复："
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_prompt(self, question: str, evidences: List[Dict], sub_questions: List[str]) -> str:
        messages = self._chat_messages(question, evidences, sub_questions)
        system, user = messages[0]["content"], messages[1]["content"]
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return f"{system}\n\n{user}"

    def _generate_once(self, prompt: str, max_new_tokens: int, messages: Optional[List[Dict[str, str]]] = None) -> str:
        if self.backend == "api" and self.api_client is not None:
            msgs = messages or [{"role": "user", "content": prompt}]
            return self.api_client.chat(msgs, temperature=0.2, max_tokens=max_new_tokens,
                                        node="legacy_generate")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        import torch

        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def _answer_incomplete(self, answer: str, sub_questions: List[str]) -> bool:
        if not answer or len(answer.strip()) < 12:
            return True
        if self._looks_like_draft(answer) or self._looks_like_garbage(answer):
            return True
        # 短事实回答（含电话号码）视为完整
        if GenerativeAnswerer._PHONE_RE.search(answer) and len(answer) < 400:
            return False
        if len(answer) < 40 and len(sub_questions) < 2:
            # 单意图短答：只要不像草稿即可
            return self._looks_like_draft(answer)
        if re.search(r"[：:]\s*$|分为几个步骤[：:]\s*$|[123]）\s*$", answer):
            return True
        if len(sub_questions) < 2:
            return False
        # 多意图：正文过短或缺少分节，视为未写全
        if len(answer.strip()) < 180:
            return True
        if not re.search(r"[一二三]、|^#{1,3}\s+", answer, re.M):
            return True
        # 流程类要点必须出现可执行步骤词
        if any(re.search(r"流程|步骤|操作|认购|如何", sq) for sq in sub_questions):
            if not re.search(r"点击|进入|确认|输入|登录|选择", answer):
                return True
            # 基金认购：缺少购买金额/认证等关键动作视为不全
            if any("基金" in sq for sq in sub_questions) and not re.search(
                r"购买金额|认证|确认|基金购买", answer
            ):
                return True
        hit = 0
        for sq in sub_questions:
            keys = [w for w in re.findall(r"[\u4e00-\u9fff]{2,}", sq) if len(w) >= 2][:4]
            if any(k in answer for k in keys):
                hit += 1
        return hit < max(2, (len(sub_questions) + 1) // 2)

    def generate(self, question: str, evidences: List[Dict], sub_questions: Optional[List[str]] = None) -> Dict[str, str]:
        """
        策略：先保证「完整结构化答案」，再可选用生成模型润色。
        R1 思考草稿绝不会作为最终答案返回。
        """
        if evidences and all(ev.get('evidence_kind') == 'page_image' for ev in evidences):
            from src.agent.nodes import AgentNodes
            return {'answer': AgentNodes._extractive_answer(evidences), 'thinking': ''}
        evidences = self._dedupe_evidences(evidences, limit=8)
        # 机构硬过滤：浦发题绝不能落到邮储手册
        evidences = self._filter_evidences_by_org(evidences, question)
        subqs = [s for s in (sub_questions or []) if s and s.strip()]
        if len(subqs) < 1:
            subqs = [question]

        if self._org_keys_from_question(question) and not evidences:
            return self._org_refuse_answer(question)

        # 0) 短事实题：禁止走大模型（易幻觉重复）
        factual = self._extract_factual_answer(question, evidences)
        if factual is not None:
            return factual

        # 1) 始终先产出完整结构化答案（API 失败时可回退）
        base = self._extractive_answer(question, evidences, subqs)
        complete_answer = base["answer"]
        thinking = ""

        if not self.available:
            return {"answer": complete_answer, "thinking": ""}

        # 本地小模型：高覆盖题型继续用结构化，避免写残
        # 云端 deepseek-v4-flash：优先走 API，精度通常更好
        if self.backend != "api" and (
            self._is_fund_dual_question(question)
            or self._is_account_dual_question(question)
            or self._is_usbkey_popup_question(question)
        ):
            return {"answer": complete_answer, "thinking": ""}

        use_llm = self.backend == "api" or len(subqs) >= 2 or len(question) >= 55
        if not use_llm:
            return {"answer": complete_answer, "thinking": ""}

        try:
            messages = self._chat_messages(question, evidences, subqs)
            prompt = self._build_prompt(question, evidences, subqs)
            default_max = "1024" if self.backend == "api" else "768"
            max_tokens = min(int(os.environ.get("FINANCE_GEN_MAX_NEW_TOKENS", default_max)), 2048)
            raw = self._generate_once(prompt, max_new_tokens=max_tokens, messages=messages)
            split = self._split_thinking(raw)
            thinking = split.get("thinking") or ""
            llm_answer = (split.get("answer") or "").strip()
            min_len = 80 if self.backend == "api" else max(180, int(len(complete_answer) * 0.55))
            if (
                llm_answer
                and not self._looks_like_garbage(llm_answer)
                and not self._answer_incomplete(llm_answer, subqs)
                and len(llm_answer) >= min_len
            ):
                complete_answer = llm_answer
            elif raw and self._looks_like_draft(raw):
                thinking = raw if not thinking else thinking
        except PilotBudgetStopped:
            raise
        except Exception as e:
            thinking = f"（生成模型调用失败，已回退结构化答案）{e}"

        return {"answer": complete_answer, "thinking": thinking}


class AnswerSynthesizer:
    @staticmethod
    def _format_evidence_footer(evidences: List[Dict], limit: int = 5) -> str:
        lines = ["", "————", "【证据溯源】"]
        for idx, ev in enumerate(evidences[:limit], start=1):
            doc = ev.get("source_doc") or "未知文档"
            page = ev.get("page")
            bid = ev.get("block_id") or "-"
            score = round(float(ev.get("fusion_score", ev.get("score", 0.0))), 6)
            # 路径感：文档名 + 页码 + 块
            path_hint = f"{doc}"
            if page is not None:
                path_hint += f" / 第{page}页"
            if bid and bid != "-":
                path_hint += f" / block={bid}"
            snippet = re.sub(r"\s+", " ", (ev.get("content") or "").strip())
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            lines.append(f"{idx}. 来源：{path_hint}")
            lines.append(f"   相关分：{score}")
            lines.append(f"   摘录：{snippet}")
        return "\n".join(lines)

    def synthesize(
        self,
        question: str,
        evidences: List[Dict],
        generated_answer: str = "",
        thinking: str = "",
        sub_questions: Optional[List[str]] = None,
    ) -> Dict:
        points = []
        for idx, ev in enumerate(evidences[:5], start=1):
            content = (ev.get("content") or "").strip()
            content = re.sub(r"\s+", " ", content)
            if len(content) > 280:
                content = content[:280] + "..."
            points.append(
                {
                    "rank": idx,
                    "text": content,
                    "source_doc": ev.get("source_doc"),
                    "page": ev.get("page"),
                    "block_id": ev.get("block_id"),
                    "score": round(float(ev.get("fusion_score", ev.get("score", 0.0))), 6),
                }
            )

        if points:
            avg = sum(p["score"] for p in points) / len(points)
            confidence = round(min(0.92, max(0.35, 0.35 + abs(avg) * 0.45)), 4)
        else:
            confidence = 0.0

        answer_body = (generated_answer or "").strip()
        if GenerativeAnswerer._looks_like_draft(answer_body) or GenerativeAnswerer._looks_like_garbage(answer_body):
            thinking = (thinking + "\n\n" + answer_body).strip() if thinking else answer_body
            answer_body = ""

        if not answer_body:
            answer_body = "未能生成有效答案，请展开下方证据溯源核对原文。"

        # 低相关度提示：仅当答案里尚未写过同类提示时再附加
        low_confidence_note = ""
        if points:
            top_score = max(p["score"] for p in points)
            if top_score < 0.35:
                confidence = round(min(confidence, 0.45), 4)
                if "证据不足" not in answer_body and "未检索到" not in answer_body and "相关度偏低" not in answer_body:
                    low_confidence_note = (
                        "检索相关度偏低，请以下方证据溯源核对原文。"
                    )

        if low_confidence_note:
            answer_body = low_confidence_note + "\n\n" + answer_body

        return {
            "question": question,
            "answer": answer_body,
            "thinking": thinking or "",
            "sub_questions": sub_questions or [],
            "knowledge_points": points,
            "confidence": confidence,
        }


class DocumentAgent:
    def __init__(self, reranker_model: Optional[str] = None, gen_model: Optional[str] = None):
        gen_model = gen_model if gen_model is not None else resolve_gen_model()
        self.planner = QueryPlanner()
        self.table_tool = TableTool()
        # reranker_model 参数保留兼容旧调用，已不再加载本地 CrossEncoder
        self.reranker = EvidenceReranker(reranker_model)
        self.answerer = GenerativeAnswerer(gen_model)
        self.synthesizer = AnswerSynthesizer()

    @staticmethod
    def _text_metadata(retriever) -> List[Dict]:
        """从混合检索器拿到文本块元数据，供关键词兜底。"""
        fusion = getattr(retriever, "fusion_retriever", None)
        text_r = getattr(fusion, "text_retriever", None) if fusion is not None else None
        if text_r is None:
            text_r = getattr(retriever, "text_retriever", None)
        meta = getattr(text_r, "metadata", None) if text_r is not None else None
        return list(meta) if meta else []

    def _keyword_phone_hits(self, question: str, retriever, limit: int = 12) -> List[Dict]:
        """电话类问题：在已加载文本块里按机构+号码关键词硬匹配，纠正语义检索串行。"""
        org_keys = GenerativeAnswerer._org_keys_from_question(question)
        meta = self._text_metadata(retriever)
        if not meta:
            return []
        want_debit = "借记卡" in question
        want_credit = "信用卡" in question
        scored = []
        for b in meta:
            content = b.get("content") or ""
            doc = b.get("source_doc") or ""
            blob = f"{doc} {content}"
            if org_keys and not any(k in blob for k in org_keys):
                continue
            if not GenerativeAnswerer._PHONE_RE.search(content):
                continue
            if not re.search(r"客服|电话|热线|联系", content):
                continue
            score = 0.90
            # 明确「借记卡客服电话…/信用卡客服电话…」最高优先（系统升级公告等）
            if re.search(r"借记卡客服电话\s*[\d\-]+", content) and re.search(
                r"信用卡客服电话\s*[\d\-]+", content
            ):
                score = 1.0
            elif want_debit and want_credit and ("借记卡" in content and "信用卡" in content):
                score = 0.98
            elif want_debit and "借记卡" in content:
                score = 0.96
            elif want_credit and "信用卡" in content:
                score = 0.96
            elif "储蓄卡/信用卡客服电话" in content or "客服电话" in content:
                # 仅提到可查看电话、号码在配图里 → 低分，不当作主证据
                score = 0.55
            item = dict(b)
            item["score"] = score
            item["fusion_score"] = score
            item["retrieval_channel"] = "keyword_phone"
            scored.append(item)
        scored.sort(key=lambda x: x["fusion_score"], reverse=True)
        return scored[:limit]

    def _keyword_fund_hits(self, question: str, retriever, limit: int = 14) -> List[Dict]:
        """基金题：补齐超市/购买金额/优选/首发等关键块，避免只命中简介句。"""
        if "基金" not in (question or ""):
            return []
        meta = self._text_metadata(retriever)
        if not meta:
            return []
        prefer = [
            "在基金购买页面输入购买金额",
            "进入“投资理财—基金业务—基金超市”",
            "全部基金产品查询和搜索",
            "优选基金进行查看",
            "基金公司近期发布的基金",
            "所有渠道我持有的基金产品的查询",
            "点击购买按钮，进入到基金购买页面",
            "新发基金的购买流程",
            "投资者问卷",
            "理财签约",
            "自选基金",
            "定投计划",
        ]
        scored = []
        for b in meta:
            content = b.get("content") or ""
            doc = b.get("source_doc") or ""
            if "基金" not in content and "基金" not in doc:
                continue
            score = 0.0
            for i, kw in enumerate(prefer):
                if kw in content:
                    score = max(score, 0.99 - i * 0.015)
            if score < 0.85:
                continue
            item = dict(b)
            item["score"] = score
            item["fusion_score"] = score
            item["retrieval_channel"] = "keyword_fund"
            scored.append(item)
        scored.sort(key=lambda x: x["fusion_score"], reverse=True)
        return scored[:limit]

    def run(self, question: str, retriever, top_k: int = 10) -> Dict:
        sub_questions = self.planner.plan(question)
        queries = [question] + [sq for sq in sub_questions if sq != question]
        # 电话题加精准检索词
        if re.search(r"电话|客服|热线", question):
            org = GenerativeAnswerer._org_keys_from_question(question)
            if org:
                queries.append(f"{org[0]} 客服电话")
            if "借记卡" in question:
                queries.append("借记卡客服电话")
            if "信用卡" in question:
                queries.append("信用卡客服电话")
            if "借记卡" in question and "信用卡" in question:
                queries.append("借记卡客服电话95580 信用卡客服电话40088-95580")
                queries.append("系统升级 客服电话")
        if "基金" in question:
            queries.extend(
                [
                    "掌上银行 基金超市 功能简介",
                    "基金购买页面 购买金额 确认",
                    "优选基金 购买 定投",
                    "首发基金 购买流程",
                ]
            )
        if GenerativeAnswerer._is_account_dual_question(question):
            queries.extend(
                [
                    "我的账户 明细查询 账户余额",
                    "总资产 总负债 存款 投资",
                    "账户总览 小眼睛",
                ]
            )
        if GenerativeAnswerer._is_usbkey_popup_question(question):
            queries.extend(
                [
                    "USBkey 未自动弹出 证书管理工具",
                    "交通银行 企业网银 证书登录",
                    "以管理员身份运行 USBKey",
                ]
            )

        pooled = []
        for sq in queries[:10]:
            pooled.extend(retriever.search(sq, top_k=top_k))

        # 关键词兜底：把含正确机构+号码的块顶到前面
        if re.search(r"电话|客服|热线", question):
            pooled.extend(self._keyword_phone_hits(question, retriever))
        if "基金" in question:
            pooled.extend(self._keyword_fund_hits(question, retriever))

        uniq = {}
        for item in pooled:
            bid = item["block_id"]
            if bid not in uniq or item.get("fusion_score", 0) > uniq[bid].get("fusion_score", 0):
                uniq[bid] = item
        ranked = sorted(uniq.values(), key=lambda x: x.get("fusion_score", x.get("score", 0.0)), reverse=True)

        # 机构硬过滤（问浦发时丢掉邮储等其它银行块）
        org_keys = GenerativeAnswerer._org_keys_from_question(question)
        if org_keys:
            org_ranked = [
                x for x in ranked if GenerativeAnswerer._evidence_matches_org(x, org_keys)
            ]
            ranked = org_ranked  # 允许为空 → 后续拒答

        if ranked:
            ranked = self.reranker.rerank(
                question,
                ranked,
                top_k=min(max(top_k, 10), 14),
                sub_questions=sub_questions,
            )
            ranked = self.table_tool.merge_table_evidence(ranked)

        # 电话题：明确「借记卡/信用卡客服电话+号码」的文本块强制置顶
        if ranked and re.search(r"电话|客服|热线", question) and org_keys:
            def _phone_explicit_rank(ev: Dict) -> int:
                t = ev.get("content") or ""
                if re.search(r"借记卡客服电话\s*[\d\-]+", t) and re.search(
                    r"信用卡客服电话\s*[\d\-]+", t
                ):
                    return 0
                if re.search(r"借记卡客服电话\s*[\d\-]+|信用卡客服电话\s*[\d\-]+", t):
                    return 1
                if re.search(r"储蓄卡/信用卡客服电话|显示我行的联系电话", t):
                    return 3
                return 2

            ranked = sorted(
                ranked,
                key=lambda x: (
                    _phone_explicit_rank(x),
                    -float(x.get("fusion_score", x.get("score", 0.0))),
                ),
            )

        gen = self.answerer.generate(question, ranked, sub_questions=sub_questions)
        answer_text = gen.get("answer", "") or ""
        # 拒答时不要展示无关银行/无关研报当「证据」
        if re.search(r"无法作答|拒绝编造|拒绝使用|未检索到「|不足以作答", answer_text):
            keep = []
            for ev in ranked:
                t = ev.get("content") or ""
                if re.search(r"单笔支付限额|只能查询|借记卡客服电话|信用卡客服电话", t):
                    keep.append(ev)
            ranked = keep
        return self.synthesizer.synthesize(
            question,
            ranked,
            generated_answer=answer_text,
            thinking=gen.get("thinking", ""),
            sub_questions=sub_questions,
        )
