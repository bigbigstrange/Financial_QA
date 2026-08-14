import { createRouter, createWebHistory } from 'vue-router'
import { getToken } from '@/api/client'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: () => import('@/views/LoginView.vue') },
    { path: '/', name: 'chat', component: () => import('@/views/ChatView.vue') },
    { path: '/ops', name: 'ops', component: () => import('@/views/OpsView.vue'), meta: { admin: true } },
  ],
})

router.beforeEach((to) => {
  if (to.name !== 'login' && !getToken()) return { name: 'login' }
  if (to.name === 'login' && getToken()) return { name: 'chat' }
  return true
})

export default router
