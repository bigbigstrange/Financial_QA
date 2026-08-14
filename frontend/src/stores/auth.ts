import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api, getToken, setToken } from '@/api/client'

export interface UserInfo {
  id: number
  username: string
  email: string
  role: string
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(getToken())
  const user = ref<UserInfo | null>(null)

  const isLogin = computed(() => !!token.value)
  const isAdmin = computed(() => user.value?.role === 'admin')

  async function login(username: string, password: string) {
    const data = await api<{ access_token: string }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    })
    token.value = data.access_token
    setToken(data.access_token)
    await fetchMe()
  }

  async function register(username: string, email: string, password: string) {
    await api('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password }),
    })
    await login(username, password)
  }

  async function fetchMe() {
    if (!token.value) return
    user.value = await api<UserInfo>('/api/auth/me')
  }

  function logout() {
    token.value = null
    user.value = null
    setToken(null)
  }

  return { token, user, isLogin, isAdmin, login, register, fetchMe, logout }
})
