<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '@/api/client'
import { useAuthStore } from '@/stores/auth'

interface Evidence {
  text?: string
  source_doc?: string
  page?: number
  score?: number
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
}

const SAMPLES = [
  '请详细说明掌上银行中“基金”功能的主要服务内容以及进行基金认购的具体操作流程。',
  '中国邮政储蓄银行借记卡客服电话和信用卡客服电话分别是多少？',
  '交通银行企业网银激活时提示「没有待激活客户信息」通常是什么原因，应如何处理？',
]

const auth = useAuthStore()
const router = useRouter()
const messages = ref<Msg[]>([])
const conversationId = ref<number | null>(null)
const input = ref('')
const loading = ref(false)
const showThinking = ref(false)
const health = ref<any>(null)

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
  loading.value = true
  messages.value.push({ role: 'user', content: question })
  input.value = ''
  try {
    const res = await api<any>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        question,
        conversation_id: conversationId.value,
        show_thinking: showThinking.value,
      }),
    })
    conversationId.value = res.conversation_id
    messages.value.push({
      id: res.assistant_message.id,
      role: 'assistant',
      content: res.assistant_message.content,
      thinking: res.assistant_message.thinking,
      confidence: res.assistant_message.confidence,
      evidence: res.assistant_message.evidence || [],
      latency_sec: res.elapsed_sec,
      model_profile: res.profile,
    })
  } catch (e: any) {
    messages.value.push({
      role: 'assistant',
      content: `（系统）${e?.message || '提问失败'}`,
    })
  } finally {
    loading.value = false
    refreshHealth()
  }
}

function newChat() {
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

function logout() {
  auth.logout()
  router.push('/login')
}

onMounted(async () => {
  await auth.fetchMe().catch(() => auth.logout())
  refreshHealth()
  setInterval(refreshHealth, 15000)
})
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
      <div class="side-foot">
        <div class="status-dot" :class="statusClass" />
        <div>
          <div>{{ health?.ready ? (health?.demo ? '就绪·演示库' : '就绪·全量') : health?.loading ? '加载中…' : '未就绪' }}</div>
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
          <div v-if="m.role === 'assistant'" class="meta-row">
            <span v-if="m.confidence != null">置信度 {{ (m.confidence * 100).toFixed(0) }}%</span>
            <span v-if="m.latency_sec != null">耗时 {{ m.latency_sec }}s</span>
            <span v-if="m.model_profile">profile {{ m.model_profile }}</span>
            <button class="btn light" style="padding:4px 8px;font-size:12px" type="button" @click="feedback(m.id, 'up')">有用</button>
            <button class="btn light" style="padding:4px 8px;font-size:12px" type="button" @click="feedback(m.id, 'down')">无用</button>
          </div>
          <div v-if="m.thinking" class="evidence">
            <details>
              <summary>思考过程</summary>
              <pre style="white-space:pre-wrap;font-family:var(--mono);font-size:12px">{{ m.thinking }}</pre>
            </details>
          </div>
          <div v-if="m.evidence?.length" class="evidence">
            <details open>
              <summary>证据 {{ m.evidence.length }} 条</summary>
              <div v-for="(ev, j) in m.evidence" :key="j" style="margin-top:8px">
                <div>{{ ev.source_doc || '未知文档' }} · p.{{ ev.page ?? '-' }} · score {{ ev.score ?? '-' }}</div>
                <div>{{ ev.text }}</div>
              </div>
            </details>
          </div>
        </div>
        <div v-if="loading" class="bubble assistant">正在检索与生成…</div>
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
