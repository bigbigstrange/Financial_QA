<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api, apiBlob, apiResumeStream, apiStream } from '@/api/client'
import { useAuthStore } from '@/stores/auth'

interface Evidence {
  evidence_id?: string
  text?: string
  source_doc?: string
  source_url?: string | null
  page?: number
  page_end?: number
  section_path?: string | null
  block_id?: string
  score?: number | null
  fusion_score?: number | null
  rerank_score?: number | null
  dense_score?: number | null
  sparse_score?: number | null
  structured_score?: number | null
  retrieval_channels?: string[]
  quality_flags?: string[]
}

/** Agent 单个节点的执行轨迹 */
interface NodeTrace {
  node: string
  elapsed_ms?: number
  decision?: string | null
  detail?: Record<string, any>
}

/** Agent 决策轨迹整体（来自 GET /api/runs/{id}/trace） */
interface AgentTrace {
  run_id?: string
  agent_mode?: string | null
  route?: string | null
  retry_retrieve?: number
  retry_generate?: number
  trace?: NodeTrace[]
  note?: string
  verified?: boolean
}

interface Msg {
  id?: number
  role: 'user' | 'assistant'
  content: string
  thinking?: string | null
  confidence?: number | null
  evidence?: Evidence[]
  latency_sec?: number | null
  model_profile?: string | null
  run_id?: string | null
  trace?: AgentTrace | null
  verified?: boolean
  reviewStatus?: 'requesting' | 'open' | 'error'
  reviewError?: string
  use_paid_model?: boolean
  use_web_search?: boolean
}

const SAMPLES = [
  '请详细说明掌上银行中“基金”功能的主要服务内容以及进行基金认购的具体操作流程。',
  '中国邮政储蓄银行借记卡客服电话和信用卡客服电话分别是多少？',
  '交通银行企业网银激活时提示「没有待激活客户信息」通常是什么原因，应如何处理？',
]
const STOP_REASONS: Record<string, string> = {
  budget_exhausted: '本地 token 上限已到', actual_over_budget: '实际用量超出本地上限',
  usage_missing: '模型未返回用量', usage_invalid: '模型返回用量不一致',
  model_mismatch: '返回模型与指定模型不符', http_402: '模型服务返回 402，请核对免费额度',
  http_429: '模型服务限流', request_uncertain: '调用结果或用量无法确认',
}
const PENDING_RUN_KEY = 'finance_pending_run_key'

const auth = useAuthStore()
const router = useRouter()
const messages = ref<Msg[]>([])
const conversationId = ref<number | null>(null)
const input = ref('')
const loading = ref(false)
const activeRunId = ref<string | null>(null)
const cancelling = ref(false)
const showThinking = ref(false)
const livePilot = ref(false)
// Explicit opt-in on each page load: an accidental click must not spend model/search quota.
const usePaidModel = ref(false)
const useWebSearch = ref(false)
let streamController: AbortController | undefined
const health = ref<any>(null)
const healthSummary = computed(() => {
  if (!health.value?.ready) return health.value?.loading ? '加载中…' : '未就绪'
  const { documents, text_documents, image_only_documents } = health.value
  const detail = Number.isInteger(text_documents) && Number.isInteger(image_only_documents)
    ? `（${text_documents} 份文本、${image_only_documents} 份仅图像）` : ''
  return `就绪·${documents ?? '-'} 份文档${detail}`
})
const webSearchStatus = computed(() => {
  if (!health.value || health.value.error) return '网页搜索：状态未知'
  return health.value.web_search_configured ? '网页搜索：已配置' : '网页搜索：未配置'
})
const evidenceDetails = ref<Record<string, { loading?: boolean; error?: string; data?: any }>>({})
let healthTimer: ReturnType<typeof setInterval> | undefined
const imagePreviews = ref<Record<string, { loading?: boolean; url?: string; error?: string }>>({})
const imageRequests = new Map<string, AbortController>()
let evidenceEpoch = 0

function publicSourceUrl(value?: string | null): string | null {
  if (!value) return null
  try {
    const url = new URL(value)
    if (!['http:', 'https:'].includes(url.protocol) || !url.hostname.includes('.') ||
        url.hostname === 'localhost' || /^(127\.|10\.|192\.168\.)/.test(url.hostname)) return null
    return url.href
  } catch { return null }
}

function closeImage(id: string) {
  imageRequests.get(id)?.abort()
  imageRequests.delete(id)
  const url = imagePreviews.value[id]?.url
  if (url) URL.revokeObjectURL(url)
  delete imagePreviews.value[id]
}

function clearEvidence() {
  evidenceEpoch++
  for (const id of new Set([...imageRequests.keys(), ...Object.keys(imagePreviews.value)])) closeImage(id)
  evidenceDetails.value = {}
}

async function openImage(id: string) {
  closeImage(id)
  const controller = new AbortController()
  const epoch = evidenceEpoch
  imageRequests.set(id, controller)
  imagePreviews.value[id] = { loading: true }
  try {
    const blob = await apiBlob(`/api/evidence/${encodeURIComponent(id)}/image`, { signal: controller.signal })
    if (controller.signal.aborted || epoch !== evidenceEpoch) return
    imagePreviews.value[id] = { url: URL.createObjectURL(blob) }
  } catch (error: any) {
    if (!controller.signal.aborted && epoch === evidenceEpoch) {
      imagePreviews.value[id] = { error: error?.message || '原图读取失败' }
    }
  } finally {
    if (imageRequests.get(id) === controller) imageRequests.delete(id)
  }
}

function onEvidenceToggle(event: Event, ids: string[]) {
  if (!(event.target as HTMLDetailsElement).open) ids.forEach(closeImage)
}

function showRunResult(reply: Msg, result: any) {
  reply.run_id = result.run_id
  reply.content = result.status === 'completed' ? (result.answer || '没有可展示的答案')
    : result.status === 'cancelled' ? '本次运行已取消。'
      : `（未验证）运行已停止：${result.error_message || result.error_code || result.status}`
  reply.verified = result.verified === true
  reply.confidence = result.confidence
  reply.latency_sec = result.latency_sec
  reply.model_profile = result.model_snapshot?.profile_name
  reply.use_paid_model = result.model_snapshot?.provider !== 'local'
  reply.evidence = (result.evidence || []).map((ev: any) => ({
    evidence_id: ev.evidence_id, text: ev.text, source_doc: ev.source_name,
    source_url: ev.metadata?.source_url,
    page: ev.page_no, page_end: ev.page_end, section_path: ev.structure_path, block_id: ev.block_id,
    fusion_score: ev.fusion_score, retrieval_channels: ev.retrieval_channels,
  }))
}

async function recoverPendingRun() {
  const epoch = evidenceEpoch
  const key = sessionStorage.getItem(PENDING_RUN_KEY)
  if (!key || loading.value) return
  let located: any
  try { located = await api(`/api/runs/by-key/${encodeURIComponent(key)}`) }
  catch { return }
  const reply: Msg = { role: 'assistant', content: '正在恢复上次问答运行…', run_id: located.run_id,
    verified: false }
  const replyIndex = messages.value.push(reply) - 1
  const activeReply = messages.value[replyIndex]
  if (located.status === 'pending' || located.status === 'running') {
    loading.value = true
    activeRunId.value = located.run_id
    try {
      await apiResumeStream(`/api/runs/${encodeURIComponent(located.run_id)}/events`, 0,
        (event, data) => {
          if (epoch !== evidenceEpoch) return
          if (event === 'node') activeReply.content = `正在执行：${data.node}`
          if (event === 'cancelled') activeReply.content = '本次运行已取消。'
        })
      located = await api(`/api/runs/${encodeURIComponent(located.run_id)}`)
    } catch {
      if (epoch !== evidenceEpoch) return
      activeReply.content = '恢复连接暂时失败；刷新页面可再次查看该运行，不会重复推理。'
      return
    } finally {
      loading.value = false
      activeRunId.value = null
    }
  }
  if (epoch !== evidenceEpoch) return
  showRunResult(activeReply, located)
  sessionStorage.removeItem(PENDING_RUN_KEY)
}

async function openEvidence(id: string) {
  if (evidenceDetails.value[id]?.loading || evidenceDetails.value[id]?.data) return
  evidenceDetails.value[id] = { loading: true }
  const epoch = evidenceEpoch
  try {
    const data = await api(`/api/evidence/${encodeURIComponent(id)}`)
    if (epoch === evidenceEpoch) evidenceDetails.value[id] = { data }
  } catch (error: any) {
    if (epoch === evidenceEpoch) evidenceDetails.value[id] = { error: error?.message || '原文证据读取失败' }
  }
}

const statusClass = computed(() => {
  if (!health.value) return ''
  if (health.value.ready) return 'ready'
  if (health.value.error) return 'bad'
  return ''
})

async function refreshHealth() {
  try {
    health.value = await api('/api/health')
  } catch {
    health.value = { ready: false, error: '无法连接后端' }
  }
}

async function ask(q?: string) {
  const question = (q ?? input.value).trim()
  if (!question || loading.value) return
  const epoch = evidenceEpoch
  const paidModel = usePaidModel.value && health.value?.ready === true &&
    !health.value?.paid_api_disabled && !health.value?.pilot_budget_stop_reason
  const webSearch = useWebSearch.value && health.value?.web_search_configured === true
  loading.value = true
  messages.value.push({ role: 'user', content: question,
    use_paid_model: paidModel, use_web_search: webSearch })
  input.value = ''
  let liveReplyIndex = -1
  try {
    if (livePilot.value) {
      liveReplyIndex = messages.value.push({ role: 'assistant', content: '正在运行…',
        verified: false, use_paid_model: paidModel, trace: { trace: [] } }) - 1
      streamController = new AbortController()
      let runId = ''
      const idempotencyKey = crypto.randomUUID()
      sessionStorage.setItem(PENDING_RUN_KEY, idempotencyKey)
      let failed = false
      let terminal = false
      let lastEventId = 0
      const onRunEvent = (event: string, data: any, id?: number) => {
        if (epoch !== evidenceEpoch) return
        if (id && id <= lastEventId) return
        if (id) lastEventId = id
        const reply = messages.value[liveReplyIndex]
        if (event === 'started') {
          runId = data.run_id
          activeRunId.value = runId
          reply.run_id = runId
        } else if (event === 'node') {
          reply.trace?.trace?.push({ node: data.node, decision: data.decision, elapsed_ms: data.elapsed_ms })
        } else if (event === 'error') {
          failed = true
          terminal = true
          reply.content = `（未验证）运行已停止：${STOP_REASONS[data.reason] || data.code || 'internal_error'}`
        } else if (event === 'cancelled') {
          failed = true
          terminal = true
          reply.content = '本次运行已取消；迟到结果不会作为答案发布。'
        } else if (event === 'done') {
          runId = data.run_id
          reply.verified = data.verified === true
          terminal = true
        }
      }
      try {
        await apiStream('/api/runs/stream', { question, conversation_id: conversationId.value,
          show_thinking: showThinking.value, idempotency_key: idempotencyKey,
          use_paid_model: paidModel, use_web_search: webSearch }, onRunEvent, streamController.signal)
      } catch (error) {
        if (epoch !== evidenceEpoch) return
        if (streamController.signal.aborted) throw error
        if (runId) { /* durable events can be replayed below */ }
        else { messages.value[liveReplyIndex].content = '连接中断，正在查找已创建的运行…' }
      }
      if (epoch !== evidenceEpoch) return
      for (let attempt = 0; attempt < 3 && !runId; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 250 * (attempt + 1)))
        try {
          const located = await api<{run_id: string}>(`/api/runs/by-key/${idempotencyKey}`)
          if (epoch !== evidenceEpoch) return
          runId = located.run_id
          messages.value[liveReplyIndex].run_id = runId
          activeRunId.value = runId
        } catch { /* worker may not have created its run yet */ }
      }
      for (let attempt = 0; runId && !terminal && attempt < 3; attempt++) {
        await new Promise(resolve => setTimeout(resolve, attempt ? 1000 : 250))
        if (streamController.signal.aborted) throw new Error('事件流已断开')
        try {
          await apiResumeStream(`/api/runs/${encodeURIComponent(runId)}/events`,
            lastEventId, onRunEvent, streamController.signal)
        } catch (error) {
          if (attempt === 2) throw error
        }
      }
      if (runId && !terminal) throw new Error('运行仍在进行，请稍后通过运行编号查询')
      if (!failed && runId) {
        const result = await api<any>(`/api/runs/${encodeURIComponent(runId)}`)
        if (epoch !== evidenceEpoch) return
        const reply = messages.value[liveReplyIndex]
        showRunResult(reply, result)
      } else if (!failed) {
        throw new Error('事件流未返回运行编号')
      }
      if (terminal) sessionStorage.removeItem(PENDING_RUN_KEY)
      return
    }
    const res = await api<any>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        question,
        conversation_id: conversationId.value,
        show_thinking: showThinking.value,
        use_paid_model: paidModel,
        use_web_search: webSearch,
      }),
    })
    if (epoch !== evidenceEpoch) return
    conversationId.value = res.conversation_id
    // 先取轨迹再入列：Vue 的 ref 数组会对 push 进去的对象做 proxy，
    // 之后修改原始对象不会触发更新，所以必须在 push 之前把数据填好。
    let trace: AgentTrace | null = null
    if (res.run_id) {
      try {
        trace = await api<AgentTrace>(`/api/runs/${res.run_id}/trace`)
      } catch {
        trace = null // 轨迹是可选信息，取不到不影响答案展示
      }
    }
    if (epoch !== evidenceEpoch) return
    messages.value.push({
      id: res.assistant_message.id,
      role: 'assistant',
      content: res.assistant_message.content,
      thinking: res.assistant_message.thinking,
      confidence: res.assistant_message.confidence,
      evidence: res.assistant_message.evidence || [],
      latency_sec: res.elapsed_sec,
      model_profile: res.profile,
      use_paid_model: res.profile !== 'local-retrieval',
      run_id: res.run_id,
      trace,
      verified: trace?.verified === true,
    })
  } catch (e: any) {
    if (epoch !== evidenceEpoch) return
    const content = `（未验证）${e?.message || '提问失败'}`
    if (liveReplyIndex >= 0) messages.value[liveReplyIndex].content = content
    else messages.value.push({ role: 'assistant', content })
  } finally {
    streamController = undefined
    activeRunId.value = null
    cancelling.value = false
    loading.value = false
    refreshHealth()
  }
}

async function cancelActiveRun() {
  if (!activeRunId.value || cancelling.value) return
  cancelling.value = true
  try {
    await api(`/api/runs/${encodeURIComponent(activeRunId.value)}/cancel`, { method: 'POST' })
  } catch (error: any) {
    cancelling.value = false
    window.alert(error?.message || '取消请求失败')
  }
}

async function requestReview(message: Msg) {
  if (!message.run_id || message.reviewStatus === 'requesting' || message.reviewStatus === 'open') return
  if (!window.confirm('确认创建平台内部知识核查工单？仅供人工复核，不会执行银行交易。')) return
  message.reviewStatus = 'requesting'
  message.reviewError = undefined
  try {
    const result = await api<{ticket_status: string}>(
      `/api/runs/${encodeURIComponent(message.run_id)}/approvals`,
      { method: 'POST', body: JSON.stringify({ decision: 'accept' }) },
    )
    message.reviewStatus = result.ticket_status === 'open' ? 'open' : 'error'
  } catch (error: any) {
    message.reviewStatus = 'error'
    message.reviewError = error?.message || '核查申请失败'
  }
}

function newChat() {
  streamController?.abort()
  clearEvidence()
  conversationId.value = null
  messages.value = []
}

async function feedback(messageId: number | undefined, rating: 'up' | 'down') {
  if (!messageId) return
  try {
    await api('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({ message_id: messageId, rating }),
    })
  } catch {
    /* ignore */
  }
}

async function logout() {
  await auth.logout()
  router.push('/login')
}

onMounted(async () => {
  await auth.fetchMe().catch(() => auth.logout())
  if (auth.user) await recoverPendingRun()
  refreshHealth()
  healthTimer = setInterval(refreshHealth, 15000)
})
onUnmounted(() => { streamController?.abort(); clearEvidence(); if (healthTimer) clearInterval(healthTimer) })
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="side-brand">
        <div class="logo">金</div>
        <div>
          <div class="logo-title">金融文档问答</div>
          <div class="logo-sub">Enterprise RAG</div>
        </div>
      </div>
      <button class="btn-new" type="button" @click="newChat">＋ 新对话</button>
      <nav>
        <router-link class="nav-item" to="/">文档问答</router-link>
        <router-link v-if="auth.isAdmin" class="nav-item" to="/ops">运维 / 评测</router-link>
        <div class="nav-item muted">证据溯源</div>
        <div class="nav-item muted">拒答防幻觉</div>
      </nav>
      <label class="nav-item" style="display:flex;justify-content:space-between;align-items:center">
        <span>显示思考过程</span>
        <input v-model="showThinking" type="checkbox" />
      </label>
      <label class="nav-item" style="display:flex;justify-content:space-between;align-items:center">
        <span>实时试跑轨迹（不写入会话）</span>
        <input v-model="livePilot" type="checkbox" />
      </label>
      <label class="nav-item" style="display:flex;justify-content:space-between;align-items:center"
        title="开启后允许本次提问调用豆包生成、相关性评分与双校验；消耗模型额度。关闭后只给未经模型验证的本地证据摘录。">
        <span>豆包模型问答（消耗模型额度）</span>
        <input v-model="usePaidModel" type="checkbox"
          :disabled="loading || !health?.ready || health?.paid_api_disabled || !!health?.pilot_budget_stop_reason" />
      </label>
      <label class="nav-item" style="display:flex;justify-content:space-between;align-items:center"
        title="开启后允许 Agent 在路由或本地证据不足时调用 Tavily 搜索公开网页；消耗 Tavily 搜索额度，不会强制每题搜索。">
        <span>Tavily 网页补查（消耗搜索额度）</span>
        <input v-model="useWebSearch" type="checkbox"
          :disabled="loading || !health?.ready || !health?.web_search_configured" />
      </label>
      <p style="padding:0 16px;font-size:12px;line-height:1.5;color:var(--muted)">
        豆包：路由、证据评分、生成与校验。Tavily：仅在需要时搜索公开网页。两项均按本次提问生效，默认关闭。
      </p>
      <div class="side-foot">
        <div class="status-dot" :class="statusClass" />
        <div>
          <div>{{ healthSummary }}</div>
          <div v-if="health?.paid_api_disabled">本地检索 · 付费模型已暂停</div>
          <div v-else-if="health?.pilot_budget_stop_reason">模型预算暂停 · {{ STOP_REASONS[health.pilot_budget_stop_reason] || health.pilot_budget_stop_reason }}</div>
          <div>{{ webSearchStatus }}</div>
          <div>
            {{ auth.user?.username || '-' }}
            · mem {{ health?.memory?.pct ?? '-' }}%
          </div>
        </div>
      </div>
      <button class="btn ghost" type="button" @click="logout">退出登录</button>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <h1>基于多模态金融文档的智能问答</h1>
          <p class="topbar-desc">会话持久化 · 证据可追溯 · 内存水位防护 · 模型可迭代评测</p>
        </div>
        <div>
          <span class="tag">DocIR</span>
          <span class="tag">MemoryGuard</span>
          <span class="tag">Eval Gate</span>
        </div>
      </header>

      <section v-if="!messages.length" class="suggest-grid">
        <button v-for="s in SAMPLES" :key="s" class="chip" type="button" @click="ask(s)">{{ s }}</button>
      </section>

      <section class="thread">
        <div
          v-for="(m, i) in messages"
          :key="i"
          class="bubble"
          :class="m.role"
        >
          <div>{{ m.content }}</div>
          <div v-if="m.role === 'user' && m.use_paid_model != null" class="meta-row">
            本次允许：豆包 {{ m.use_paid_model ? '开' : '关' }} · Tavily {{ m.use_web_search ? '开' : '关' }}
          </div>
          <div v-if="m.role === 'assistant'" class="meta-row">
            <span v-if="m.verified != null">{{ m.verified ? '模型校验通过（非人工复核）' : m.use_paid_model === false ? '未执行模型校验' : '模型校验未通过或未完成' }}</span>
            <span v-if="m.latency_sec != null">耗时 {{ m.latency_sec }}s</span>
            <span v-if="m.model_profile">profile {{ m.model_profile }}</span>
            <button class="btn light" style="padding:4px 8px;font-size:12px" type="button" @click="feedback(m.id, 'up')">有用</button>
            <button class="btn light" style="padding:4px 8px;font-size:12px" type="button" @click="feedback(m.id, 'down')">无用</button>
            <button v-if="m.run_id" class="btn light" style="padding:4px 8px;font-size:12px" type="button"
              :disabled="m.reviewStatus === 'requesting' || m.reviewStatus === 'open'"
              @click="requestReview(m)">{{ m.reviewStatus === 'open' ? '核查工单已提交' : m.reviewStatus === 'requesting' ? '提交中…' : '申请人工核查' }}</button>
          </div>
          <p v-if="m.reviewError" role="alert">{{ m.reviewError }}</p>
          <div v-if="m.thinking" class="evidence">
            <details>
              <summary>思考过程</summary>
              <pre style="white-space:pre-wrap;font-family:var(--mono);font-size:12px">{{ m.thinking }}</pre>
            </details>
          </div>
          <div v-if="m.trace && (m.trace.trace?.length || m.trace.note)" class="evidence">
            <details open>
              <summary>
                决策轨迹
                <span v-if="m.trace.route" class="pill">{{ m.trace.route }}</span>
                <span v-if="m.trace.retry_retrieve || m.trace.retry_generate" class="pill warn">
                  重试 r={{ m.trace.retry_retrieve }} / g={{ m.trace.retry_generate }}
                </span>
              </summary>
              <ol v-if="m.trace.trace?.length" class="trace-list">
                <li v-for="(t, k) in m.trace.trace" :key="k">
                  <span class="trace-node">{{ t.node }}</span>
                  <span class="trace-ms">{{ t.elapsed_ms ?? 0 }}ms</span>
                  <span class="trace-decision">{{ t.decision || '' }}</span>
                </li>
              </ol>
              <div v-else class="muted">{{ m.trace.note }}</div>
            </details>
          </div>
          <div v-if="m.evidence?.length" class="evidence">
            <details @toggle="onEvidenceToggle($event, m.evidence.map(ev => ev.evidence_id).filter((id): id is string => !!id))">
              <summary>证据 {{ m.evidence.length }} 条</summary>
              <div v-for="(ev, j) in m.evidence" :key="j" class="ev-item">
                <div class="ev-head">
                  <span class="ev-src">{{ ev.source_doc || '未知文档' }}</span>
                  <a v-if="publicSourceUrl(ev.source_url)" :href="publicSourceUrl(ev.source_url) || undefined"
                    target="_blank" rel="noopener noreferrer">公开来源 ↗</a>
                  <span v-if="ev.page">· p.{{ ev.page }}{{ ev.page_end && ev.page_end > ev.page ? `–${ev.page_end}` : '' }}</span>
                  <span v-if="ev.section_path" class="ev-sec">· {{ ev.section_path }}</span>
                  <span v-if="ev.score != null" class="ev-score">
                    score {{ Number(ev.score).toFixed(3) }}
                  </span>
                </div>
                <div v-if="ev.retrieval_channels?.length" class="ev-tags">
                  <span v-for="ch in ev.retrieval_channels" :key="ch" class="pill tiny">{{ ch }}</span>
                </div>
                <div>{{ ev.text }}</div>
                <div v-if="ev.evidence_id">
                  <button class="btn light" type="button" :disabled="evidenceDetails[ev.evidence_id]?.loading"
                    @click="openEvidence(ev.evidence_id)">
                    {{ evidenceDetails[ev.evidence_id]?.loading ? '读取中…' : '查看完整原文证据' }}
                  </button>
                  <p v-if="evidenceDetails[ev.evidence_id]?.error" role="alert">{{ evidenceDetails[ev.evidence_id]?.error }}</p>
                  <details v-if="evidenceDetails[ev.evidence_id]?.data" open @toggle="onEvidenceToggle($event, [ev.evidence_id])">
                    <summary>原文定位 · {{ evidenceDetails[ev.evidence_id]?.data.source_name }}</summary>
                    <div v-if="evidenceDetails[ev.evidence_id]?.data.document_version">版本：{{ evidenceDetails[ev.evidence_id]?.data.document_version }}</div>
                    <div v-if="evidenceDetails[ev.evidence_id]?.data.evidence_kind === 'web'">网页搜索摘录（非网站全文）</div>
                    <div v-else>位置：{{ evidenceDetails[ev.evidence_id]?.data.page_no ? `第 ${evidenceDetails[ev.evidence_id]?.data.page_no}${evidenceDetails[ev.evidence_id]?.data.page_end > evidenceDetails[ev.evidence_id]?.data.page_no ? `–${evidenceDetails[ev.evidence_id]?.data.page_end}` : ''} 页` : '无物理页码' }}</div>
                    <pre style="white-space:pre-wrap;overflow-wrap:anywhere">{{ evidenceDetails[ev.evidence_id]?.data.text }}</pre>
                    <p v-if="evidenceDetails[ev.evidence_id]?.data.evidence_kind === 'page_image'">仅图像证据，未经文字解析／答案验证</p>
                    <div v-if="evidenceDetails[ev.evidence_id]?.data.image_available">
                      <div>图像命中页：第 {{ evidenceDetails[ev.evidence_id]?.data.vision_page }} 页</div>
                      <button v-if="!imagePreviews[ev.evidence_id]?.url && !imagePreviews[ev.evidence_id]?.loading"
                        class="btn light" type="button" @click="openImage(ev.evidence_id)">查看原图</button>
                      <button v-else class="btn light" type="button" @click="closeImage(ev.evidence_id)">
                        {{ imagePreviews[ev.evidence_id]?.loading ? '取消加载' : '关闭原图' }}
                      </button>
                      <p v-if="imagePreviews[ev.evidence_id]?.error" role="alert">{{ imagePreviews[ev.evidence_id]?.error }}</p>
                      <img v-if="imagePreviews[ev.evidence_id]?.url" :src="imagePreviews[ev.evidence_id]?.url"
                        :alt="`${evidenceDetails[ev.evidence_id]?.data.source_name} 第 ${evidenceDetails[ev.evidence_id]?.data.vision_page} 页原图`"
                        style="display:block;max-width:100%;height:auto;margin-top:8px" />
                    </div>
                  </details>
                </div>
              </div>
            </details>
          </div>
        </div>
        <div v-if="loading" class="bubble assistant">正在检索与生成…
          <button v-if="activeRunId" class="btn light" type="button" :disabled="cancelling" @click="cancelActiveRun">
            {{ cancelling ? '取消请求已发送…' : '取消本次运行' }}
          </button>
        </div>
      </section>

      <footer class="input-dock">
        <form class="input-box" @submit.prevent="ask()">
          <textarea
            v-model="input"
            rows="2"
            placeholder="向知识库提问…"
            maxlength="2000"
            @keydown.enter.exact.prevent="ask()"
          />
          <button class="btn" type="submit" :disabled="loading || !input.trim()">发送</button>
        </form>
      </footer>
    </main>
  </div>
</template>
