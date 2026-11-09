import { createRouter, createWebHistory } from 'vue-router'
import { api, getToken, hasRefreshSession } from '@/api/client'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: () => import('@/views/LoginView.vue') },
    { path: '/', name: 'chat', component: () => import('@/views/ChatView.vue') },
    { path: '/ops', name: 'ops', component: () => import('@/views/OpsView.vue'), meta: { admin: true } },
  ],
})

router.beforeEach(async (to) => {
  if (!getToken() && hasRefreshSession()) {
    try { await api('/api/auth/me') } catch { /* expired cookie: show login */ }
  }
  if (to.name !== 'login' && !getToken()) return { name: 'login' }
  if (to.name === 'login' && getToken()) return { name: 'chat' }
  return true
})

export default router
