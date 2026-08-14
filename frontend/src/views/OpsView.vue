<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '@/api/client'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const router = useRouter()
const health = ref<any>(null)
const memory = ref<any>(null)
const profiles = ref<any[]>([])
const evals = ref<any[]>([])
const events = ref<any[]>([])
const msg = ref('')
const busy = ref(false)
const evalProfile = ref('candidate-v2')
const maxSamples = ref(10)

async function refresh() {
  health.value = await api('/api/ops/health')
  memory.value = await api('/api/ops/memory')
  profiles.value = await api('/api/ops/models/profiles')
  evals.value = await api('/api/ops/eval/runs?limit=20')
  events.value = await api('/api/ops/memory/events?limit=20')
}

async function run(action: () => Promise<any>, okText: string) {
  busy.value = true
  msg.value = ''
  try {
    await action()
    msg.value = okText
    await refresh()
  } catch (e: any) {
    msg.value = e?.message || '操作失败'
  } finally {
    busy.value = false
  }
}

onMounted(async () => {
  if (!auth.user) await auth.fetchMe().catch(() => auth.logout())
  if (!auth.isAdmin) {
    await router.replace('/')
    return
  }
  await refresh()
})
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="side-brand">
        <div class="logo">运</div>
        <div>
          <div class="logo-title">运维控制台</div>
          <div class="logo-sub">Memory · Model · Eval</div>
        </div>
      </div>
      <router-link class="nav-item" to="/">← 返回问答</router-link>
      <router-link class="nav-item" to="/ops">看板</router-link>
      <div class="side-foot">
        <div class="status-dot" :class="health?.ready ? 'ready' : ''" />
        <div>管理员 {{ auth.user?.username }}</div>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <h1>可上线运维看板</h1>
          <p class="topbar-desc">内存水位 · 模型 profile · 持续评测门禁 · 事件审计</p>
        </div>
      </header>

      <p v-if="msg" style="padding: 0 28px; color: var(--brand)">{{ msg }}</p>

      <section class="ops-grid">
        <div class="card">
          <h3>系统内存</h3>
          <div class="val" :class="'level-' + (memory?.snapshot?.level || 'ok')">
            {{ memory?.snapshot?.pct ?? '-' }}%
          </div>
          <div style="color: var(--muted); font-size: 13px; margin-top: 6px">
            RSS {{ memory?.snapshot?.rss_mb }} MB · 可用 {{ memory?.snapshot?.available_mb }} MB
            <span v-if="memory?.snapshot?.cuda_allocated_mb != null">
              · CUDA {{ memory.snapshot.cuda_allocated_mb }} MB
            </span>
          </div>
        </div>
        <div class="card">
          <h3>引擎</h3>
          <div class="val" style="font-size: 20px">{{ health?.ready ? 'READY' : 'WAIT' }}</div>
          <div style="color: var(--muted); font-size: 13px; margin-top: 6px">
            profile {{ health?.profile }} · chunks {{ health?.chunks }} · demo {{ health?.demo }}
          </div>
          <div class="actions">
            <button class="btn light" :disabled="busy" type="button" @click="run(() => api('/api/ops/engine/load', { method: 'POST' }), '引擎已重载')">
              重载引擎
            </button>
            <button class="btn light" :disabled="busy" type="button" @click="run(() => api('/api/ops/engine/unload-generator', { method: 'POST' }), '已卸载生成模型')">
              卸载生成模型
            </button>
          </div>
        </div>
        <div class="card">
          <h3>评测门禁</h3>
          <div class="field">
            <label>候选 profile</label>
            <input v-model="evalProfile" />
          </div>
          <div class="field">
            <label>样本数</label>
            <input v-model.number="maxSamples" type="number" min="1" max="100" />
          </div>
          <div class="actions">
            <button
              class="btn"
              :disabled="busy"
              type="button"
              @click="run(() => api('/api/ops/eval/run', {
                method: 'POST',
                body: JSON.stringify({
                  profile_name: evalProfile,
                  max_samples: maxSamples,
                  promote_if_pass: false,
                  async_mode: false,
                }),
              }), '评测完成')"
            >
              同步评测
            </button>
            <button
              class="btn light"
              :disabled="busy"
              type="button"
              @click="run(() => api('/api/ops/models/rollback', { method: 'POST' }), '已回滚')"
            >
              回滚生产
            </button>
          </div>
        </div>
      </section>

      <section style="padding: 0 28px 20px">
        <div class="card">
          <h3>模型 Profiles</h3>
          <table class="table">
            <thead>
              <tr>
                <th>名称</th>
                <th>标签</th>
                <th>active</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="p in profiles" :key="p.name">
                <td>{{ p.name }}</td>
                <td>{{ p.label }}</td>
                <td>{{ p.is_active ? '✓' : '' }}</td>
                <td>
                  <button
                    v-if="!p.is_active"
                    class="btn light"
                    style="padding: 4px 8px; font-size: 12px"
                    type="button"
                    :disabled="busy"
                    @click="run(() => api('/api/ops/models/promote', {
                      method: 'POST',
                      body: JSON.stringify({ profile_name: p.name, force: true }),
                    }), '已强制晋级（生产请走评测门禁）')"
                  >
                    强制晋级
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section style="padding: 0 28px 20px">
        <div class="card">
          <h3>评测历史</h3>
          <table class="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Profile</th>
                <th>状态</th>
                <th>成功率</th>
                <th>时延</th>
                <th>证据覆盖</th>
                <th>门禁</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="e in evals" :key="e.id">
                <td>{{ e.id }}</td>
                <td>{{ e.profile_name }}</td>
                <td>{{ e.status }}</td>
                <td>{{ e.success_rate ?? '-' }}</td>
                <td>{{ e.avg_latency_sec ?? '-' }}</td>
                <td>{{ e.evidence_coverage ?? '-' }}</td>
                <td>{{ e.gate_passed ? 'PASS' : 'FAIL' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section style="padding: 0 28px 28px">
        <div class="card">
          <h3>内存事件</h3>
          <table class="table">
            <thead>
              <tr>
                <th>级别</th>
                <th>动作</th>
                <th>RSS</th>
                <th>%</th>
                <th>详情</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="ev in events" :key="ev.id">
                <td>{{ ev.level }}</td>
                <td>{{ ev.action }}</td>
                <td>{{ ev.rss_mb }}</td>
                <td>{{ ev.pct }}</td>
                <td>{{ ev.detail }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
</template>
