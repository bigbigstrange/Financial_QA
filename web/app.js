/**
 * 仅保留当前知识库可完整作答（或明确拒答）的题型示例。
 * 已剔除：建行网银盾手册 / 中企云链 / K宝 / 数字证书在线客服 / 组合明细 等库内无原文的题。
 */
const SAMPLES = [
  {
    type: "多意图·内容+流程",
    q: "请详细说明掌上银行中“基金”功能的主要服务内容以及进行基金认购的具体操作流程。",
  },
  {
    type: "多意图·功能+操作",
    q: "掌上银行中「我的账户」和「总资产」分别提供哪些查询能力，以及如何进行明细查询？",
  },
  {
    type: "多意图·步骤+异常",
    q: "交通银行企业网银证书用户登录时，若插入 USBkey 后证书管理工具未自动弹出应如何处理？并说明证书登录需注意什么。",
  },
  {
    type: "事实抽取",
    q: "中国邮政储蓄银行借记卡客服电话和信用卡客服电话分别是多少？",
  },
  {
    type: "事实·公告信息",
    q: "中国邮政储蓄银行系统升级公告中写明：如有疑问应致电哪些客服电话？",
  },
  {
    type: "故障排查",
    q: "交通银行企业网银激活时提示「没有待激活客户信息」通常是什么原因，应如何处理？",
  },
  {
    type: "拒答·机构约束",
    q: "上海浦东发展银行公司网银的“账户设置”功能中，“单笔支付限额”栏位设置为“0”时表示什么？",
  },
  {
    type: "拒答·证据不足",
    q: "在上海浦东发展银行公司网银中，如果某操作员被设置为“只能查询”某账户权限，则该操作员能否进行该账户的转账或授权操作？",
  },
];

const SAMPLE_QUESTIONS = SAMPLES.map((s) => s.q);

/** 按主题分组的追问推荐（跨类型，展示泛化） */
const RELATED_POOLS = [
  {
    keys: ["基金", "认购", "申购"],
    qs: [
      "掌上银行中「我的账户」和「总资产」分别提供哪些查询能力，以及如何进行明细查询？",
      "中国邮政储蓄银行借记卡客服电话和信用卡客服电话分别是多少？",
      "交通银行企业网银证书用户登录时，若插入 USBkey 后证书管理工具未自动弹出应如何处理？并说明证书登录需注意什么。",
    ],
  },
  {
    keys: ["账户", "总览", "总资产", "明细查询", "我的账户"],
    qs: [
      "请详细说明掌上银行中“基金”功能的主要服务内容以及进行基金认购的具体操作流程。",
      "中国邮政储蓄银行借记卡客服电话和信用卡客服电话分别是多少？",
      "中国邮政储蓄银行系统升级公告中写明：如有疑问应致电哪些客服电话？",
    ],
  },
  {
    keys: ["USBkey", "USB", "证书管理", "交通银行", "交行", "激活"],
    qs: [
      "交通银行企业网银证书用户登录时，若插入 USBkey 后证书管理工具未自动弹出应如何处理？并说明证书登录需注意什么。",
      "交通银行企业网银激活时提示「没有待激活客户信息」通常是什么原因，应如何处理？",
      "请详细说明掌上银行中“基金”功能的主要服务内容以及进行基金认购的具体操作流程。",
    ],
  },
  {
    keys: ["邮储", "邮政储蓄", "客服", "电话", "系统升级", "95580"],
    qs: [
      "中国邮政储蓄银行借记卡客服电话和信用卡客服电话分别是多少？",
      "中国邮政储蓄银行系统升级公告中写明：如有疑问应致电哪些客服电话？",
      "掌上银行中「我的账户」和「总资产」分别提供哪些查询能力，以及如何进行明细查询？",
    ],
  },
  {
    keys: ["浦发", "浦东发展", "限额", "只能查询"],
    qs: [
      "上海浦东发展银行公司网银的“账户设置”功能中，“单笔支付限额”栏位设置为“0”时表示什么？",
      "在上海浦东发展银行公司网银中，如果某操作员被设置为“只能查询”某账户权限，则该操作员能否进行该账户的转账或授权操作？",
      "中国邮政储蓄银行借记卡客服电话和信用卡客服电话分别是多少？",
    ],
  },
];

const LOADING_HINTS = [
  "正在解析问题意图…",
  "正在检索多模态文档证据…",
  "正在对齐来源与置信度…",
  "正在整理结构化回答…",
];

function pickRelatedQuestions(question, n = 3) {
  const q = (question || "").trim();
  const scored = [];
  for (const pool of RELATED_POOLS) {
    if (pool.keys.some((k) => q.includes(k))) {
      for (const cand of pool.qs) {
        if (cand !== q && !scored.includes(cand)) scored.push(cand);
      }
    }
  }
  for (const cand of SAMPLE_QUESTIONS) {
    if (cand !== q && !scored.includes(cand)) scored.push(cand);
  }
  return scored.slice(0, n);
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const statusDot = document.getElementById("statusDot");
const metaEl = document.getElementById("meta");
const chipsEl = document.getElementById("chips");
const suggestWrap = document.getElementById("suggestWrap");
const showThinkingEl = document.getElementById("showThinking");
const btnNew = document.getElementById("btnNew");

let busy = false;
let hasMessages = false;

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function setStatus(kind, text) {
  statusEl.textContent = text;
  statusDot.className = "status-dot" + (kind === "ready" ? " ready" : kind === "error" ? " error" : "");
}

function showWelcome() {
  hasMessages = false;
  chat.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">问</div>
      <h2>您好，我是金融文档助手</h2>
      <p>基于多模态金融文档检索作答。回答结构为「结论 → 说明 → 证据溯源」，证据不足时会明确告知，不编造。</p>
    </div>`;
  if (suggestWrap) suggestWrap.style.display = "";
}

function confPill(pct) {
  if (pct >= 70) return `<span class="pill pill-ok">置信度 ${pct}%</span>`;
  if (pct >= 50) return `<span class="pill pill-mid">置信度 ${pct}%</span>`;
  return `<span class="pill pill-low">置信度 ${pct}% · 偏低</span>`;
}

/** 将纯文本答案格式化为成熟问答结构 HTML */
function formatAnswerHtml(raw) {
  let text = (raw || "").trim();
  text = text.replace(/\n—+\n【证据溯源】[\s\S]*/g, "").trim();
  if (!text) return "<p>（空回答）</p>";

  const lines = text.split(/\n/);
  const blocks = [];
  let buf = [];
  let listMode = null; // 'ul' | 'ol'
  let olStart = 1;

  function flushBuf() {
    if (!buf.length) return;
    const joined = buf.join("\n").trim();
    buf = [];
    if (!joined) return;
    blocks.push({ type: "p", text: joined });
  }
  function flushList() {
    if (!listMode || !buf.length) {
      listMode = null;
      buf = [];
      olStart = 1;
      return;
    }
    blocks.push({
      type: listMode,
      items: buf.slice(),
      start: listMode === "ol" ? olStart : undefined,
    });
    listMode = null;
    buf = [];
    olStart = 1;
  }

  for (const line of lines) {
    const t = line.trim();
    // 空行：不打断列表（避免 1.2.3. 被拆成多个 <ol>，最后一项又从 1 开始）
    if (!t) {
      if (!listMode) flushBuf();
      continue;
    }
    if (/^[一二三四五六七八]、/.test(t) || /^#{1,3}\s+/.test(t) || /^要点\s*\d+/.test(t)) {
      flushList();
      flushBuf();
      blocks.push({ type: "sec", text: t.replace(/^#+\s*/, "") });
      continue;
    }
    const ol = t.match(/^(\d+)[\.、．\)]\s*(.+)$/);
    if (ol) {
      flushBuf();
      const num = parseInt(ol[1], 10) || 1;
      if (listMode !== "ol") {
        flushList();
        listMode = "ol";
        olStart = num;
      }
      buf.push(ol[2]);
      continue;
    }
    const ul = t.match(/^[-•·*]\s+(.+)$/);
    if (ul) {
      flushBuf();
      if (listMode !== "ul") {
        flushList();
        listMode = "ul";
      }
      buf.push(ul[1]);
      continue;
    }
    flushList();
    buf.push(t);
  }
  flushList();
  flushBuf();

  if (blocks.length && blocks[0].type === "p") {
    const first = blocks[0].text;
    if (
      /客服|电话|热线|限额|表示|权限|步骤|答案|无法作答|未检索到/.test(first) &&
      first.length < 220
    ) {
      blocks[0] = { type: "direct", text: first };
    }
  }

  const phoneRe =
    /(1[3-9]\d{9}|0\d{2,3}[-\s]?\d{7,8}|400\d{0,2}[-\s]?\d{2,5}[-\s]?\d{3,5}|95\d{3,5}|\+\d{1,3}[-\s]?\d{6,12})/g;

  function enrich(s) {
    let h = esc(s);
    h = h.replace(phoneRe, '<span class="phone">$1</span>');
    h = h.replace(/\(依据\[(\d+)\]\)|依据\[(\d+)\]|\[(\d+)\]/g, (_, a, b, c) => {
      const n = a || b || c;
      return `<sup class="cite-ref">[${n}]</sup>`;
    });
    return h;
  }

  return blocks
    .map((b) => {
      if (b.type === "direct") return `<div class="direct">${enrich(b.text)}</div>`;
      if (b.type === "sec") return `<div class="sec">${enrich(b.text)}</div>`;
      if (b.type === "ol") {
        const start = b.start && b.start > 1 ? ` start="${b.start}"` : "";
        return `<ol${start}>${b.items.map((i) => `<li>${enrich(i)}</li>`).join("")}</ol>`;
      }
      if (b.type === "ul")
        return `<ul>${b.items.map((i) => `<li>${enrich(i)}</li>`).join("")}</ul>`;
      return `<p>${enrich(b.text)}</p>`;
    })
    .join("");
}

function renderSources(evidence) {
  if (!evidence || !evidence.length) return "";
  const cards = evidence
    .map((ev, i) => {
      const doc = esc(ev.source_doc || "未知文档");
      const page = ev.page != null ? `第 ${esc(ev.page)} 页` : "";
      const block = ev.block_id ? String(ev.block_id) : "";
      const path = [page, block ? `块 ${esc(block)}` : ""].filter(Boolean).join(" · ");
      const score = ev.score != null ? Number(ev.score).toFixed(3) : "—";
      return `
        <div class="source-card" id="src-${i + 1}">
          <div class="source-num">${i + 1}</div>
          <div class="source-main">
            <div class="source-title">${doc}</div>
            <div class="source-path">${path || "来源片段"}</div>
            <div class="source-snippet">${esc(ev.text || "")}</div>
          </div>
          <div class="source-score">${score}</div>
        </div>`;
    })
    .join("");

  return `
    <div class="sources open" data-sources>
      <div class="sources-head">
        <h3>参考来源 · ${evidence.length}</h3>
        <span><span class="chevron">▾</span> 点击收起</span>
      </div>
      <div class="sources-list">${cards}</div>
    </div>`;
}

function addUserTurn(q) {
  if (!hasMessages) {
    chat.innerHTML = "";
    hasMessages = true;
  }
  if (suggestWrap) suggestWrap.style.display = "none";
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `<div class="q-bubble">${esc(q)}</div>`;
  chat.appendChild(turn);
  chat.scrollTop = chat.scrollHeight;
  return turn;
}

function addLoading(turn) {
  const el = document.createElement("div");
  el.className = "loading-row";
  el.innerHTML = `<div class="spinner"></div><span data-load-hint>${LOADING_HINTS[0]}</span>`;
  turn.appendChild(el);
  let i = 0;
  el._hintTimer = setInterval(() => {
    i = (i + 1) % LOADING_HINTS.length;
    const tip = el.querySelector("[data-load-hint]");
    if (tip) tip.textContent = LOADING_HINTS[i];
  }, 900);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function stopLoading(el) {
  if (el && el._hintTimer) clearInterval(el._hintTimer);
  if (el) el.remove();
}

function renderRelatedQuestions(question) {
  const related = pickRelatedQuestions(question, 3);
  if (!related.length) return "";
  const items = related
    .map(
      (q) =>
        `<button type="button" class="related-q" data-related="${esc(q)}">
          <span class="related-q-text">${esc(q)}</span>
          <span class="related-q-arr" aria-hidden="true">→</span>
        </button>`
    )
    .join("");
  return `
    <div class="related-wrap">
      <div class="related-title">相关问题推荐</div>
      <div class="related-list">${items}</div>
    </div>`;
}

function renderBotAnswer(turn, data, askedQuestion) {
  const confPct = Math.round(Number(data.confidence || 0) * 100);
  const answerRaw = data.answer || "";
  const reject =
    /未检索到|证据不足|拒答|不应编造|不会使用其它银行|无法作答|拒绝编造|拒绝使用/.test(answerRaw);

  let alertHtml = "";
  if (reject) {
    alertHtml = `<div class="a-alert reject">证据不足或相关度偏低 — 系统已避免编造，请核对下方来源。</div>`;
  } else if (confPct < 50) {
    alertHtml = `<div class="a-alert">置信度偏低，建议结合参考来源交叉核对。</div>`;
  }

  let intentHtml = "";
  if (data.sub_questions && data.sub_questions.length > 1) {
    intentHtml = `<div class="intent-row">${data.sub_questions
      .map((s, i) => `<span class="intent-chip">要点 ${i + 1}：${esc(s)}</span>`)
      .join("")}</div>`;
  }

  let thinkingHtml = "";
  if (data.show_thinking && data.thinking) {
    thinkingHtml = `
      <details class="thinking-box">
        <summary>模型思考过程（可选查看）</summary>
        <pre>${esc(data.thinking)}</pre>
      </details>`;
  }

  const elapsed =
    data.elapsed_sec != null ? Number(data.elapsed_sec).toFixed(1) : "—";

  const card = document.createElement("div");
  card.className = "a-card";
  card.innerHTML = `
    <div class="a-meta">
      <span class="label">回答</span>
      ${confPill(confPct)}
      <span class="pill pill-time">${elapsed}s</span>
      ${data.question_id ? `<span class="pill pill-id">${esc(data.question_id)}</span>` : ""}
    </div>
    ${alertHtml}
    ${intentHtml}
    ${thinkingHtml}
    <div class="a-body">
      <div class="answer">${formatAnswerHtml(answerRaw)}</div>
    </div>
    ${renderSources(data.evidence)}
    <div class="a-actions">
      <button type="button" data-copy>复制回答</button>
    </div>
    ${renderRelatedQuestions(askedQuestion || "")}`;

  const head = card.querySelector("[data-sources] .sources-head");
  if (head) {
    head.addEventListener("click", () => {
      const box = card.querySelector("[data-sources]");
      box.classList.toggle("open");
      const tip = head.querySelector("span:last-child");
      if (tip) {
        tip.innerHTML = box.classList.contains("open")
          ? `<span class="chevron">▾</span> 点击收起`
          : `<span class="chevron">▾</span> 点击展开`;
      }
    });
  }

  const copyBtn = card.querySelector("[data-copy]");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(answerRaw);
        copyBtn.textContent = "已复制";
        setTimeout(() => (copyBtn.textContent = "复制回答"), 1500);
      } catch {
        copyBtn.textContent = "复制失败";
      }
    });
  }

  card.querySelectorAll("[data-related]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = btn.getAttribute("data-related") || "";
      if (next) ask(next);
    });
  });

  turn.appendChild(card);
  chat.scrollTop = chat.scrollHeight;
}

function addError(turn, text) {
  const card = document.createElement("div");
  card.className = "a-card";
  card.innerHTML = `<div class="a-alert reject">${esc(text)}</div>`;
  turn.appendChild(card);
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    if (h.loading) {
      setStatus("loading", "引擎加载中…");
      metaEl.textContent = "正在加载索引与模型";
      return;
    }
    if (h.ready) {
      setStatus("ready", h.demo ? "服务就绪 · 演示库" : "服务就绪");
      metaEl.textContent = `${h.chunks || 0} 知识块 · ${h.embed_device || "-"}`;
      return;
    }
    setStatus("error", "未就绪");
    metaEl.textContent = (h.error || "加载失败").slice(0, 120);
  } catch (e) {
    setStatus("error", "无法连接服务");
    metaEl.textContent = String(e);
  }
}

async function ask(question) {
  if (busy) return;
  const q = (question || "").trim();
  if (!q) return;

  busy = true;
  sendBtn.disabled = true;
  input.value = "";
  input.style.height = "auto";

  const turn = addUserTurn(q);
  const loading = addLoading(turn);
  const t0 = performance.now();
  const targetMs = 3000 + Math.random() * 2000;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        show_thinking: !!(showThinkingEl && showThinkingEl.checked),
      }),
    });
    const data = await res.json().catch(() => ({}));
    const remain = Math.max(0, targetMs - (performance.now() - t0));
    if (remain > 0) await sleep(remain);
    stopLoading(loading);
    if (!res.ok) {
      addError(turn, data.detail || res.statusText || "请求失败");
      return;
    }
    const wallSec = (performance.now() - t0) / 1000;
    data.elapsed_sec = Math.max(Number(data.elapsed_sec) || 0, wallSec);
    renderBotAnswer(turn, data, q);
  } catch (e) {
    stopLoading(loading);
    addError(turn, `网络错误：${e}`);
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

SAMPLES.forEach((item) => {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "suggest-card";
  b.innerHTML = `<span class="suggest-type">${esc(item.type)}</span><span class="suggest-q">${esc(item.q)}</span>`;
  b.addEventListener("click", () => ask(item.q));
  chipsEl.appendChild(b);
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  ask(input.value);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask(input.value);
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
});

btnNew.addEventListener("click", () => {
  showWelcome();
  input.focus();
});

showWelcome();
refreshHealth();
setInterval(refreshHealth, 5000);
