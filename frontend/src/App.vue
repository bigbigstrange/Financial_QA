<script setup lang="ts">
import { onMounted } from 'vue'
import { useAuthStore } from '@/stores/auth'

const auth = useAuthStore()
onMounted(() => {
  if (auth.isLogin) auth.fetchMe().catch(() => auth.logout())
})
</script>

<template>
  <router-view />
</template>
