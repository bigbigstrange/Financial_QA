<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
const router = useRouter()
const mode = ref<'login' | 'register'>('login')
const username = ref('')
const email = ref('')
const password = ref('')
const error = ref('')
const loading = ref(false)

async function submit() {
  error.value = ''
  loading.value = true
  try {
    if (mode.value === 'login') {
      await auth.login(username.value.trim(), password.value)
    } else {
      await auth.register(username.value.trim(), email.value.trim(), password.value)
    }
    await router.replace('/')
  } catch (e: any) {
    error.value = e?.message || '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-card">
      <div class="side-brand" style="margin-bottom: 16px">
        <div class="logo">金</div>
        <div>
          <div class="logo-title" style="color: var(--ink)">金融文档问答平台</div>
          <div class="logo-sub">Enterprise RAG · 可观测 · 可迭代</div>
        </div>
      </div>
      <h1>{{ mode === 'login' ? '登录' : '注册' }}</h1>
      <p>使用已创建的账号登录；部署管理员凭据只在服务端配置。</p>
      <div v-if="error" class="error">{{ error }}</div>
      <form @submit.prevent="submit">
        <div class="field">
          <label>用户名</label>
          <input v-model="username" required autocomplete="username" />
        </div>
        <div v-if="mode === 'register'" class="field">
          <label>邮箱</label>
          <input v-model="email" type="email" required />
        </div>
        <div class="field">
          <label>密码</label>
          <input v-model="password" type="password" required
            :minlength="mode === 'register' ? 12 : undefined"
            :autocomplete="mode === 'register' ? 'new-password' : 'current-password'" />
        </div>
        <button class="btn" style="width: 100%" :disabled="loading">
          {{ loading ? '提交中…' : mode === 'login' ? '进入平台' : '创建账号' }}
        </button>
      </form>
      <p style="margin-top: 14px">
        <a href="#" @click.prevent="mode = mode === 'login' ? 'register' : 'login'" style="color: var(--brand)">
          {{ mode === 'login' ? '没有账号？注册' : '已有账号？登录' }}
        </a>
      </p>
    </div>
  </div>
</template>
