import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
import ts from 'typescript'
import { parse } from '@vue/compiler-sfc'

const require = createRequire(import.meta.url)
const source = parse(readFileSync(new URL('../src/views/ChatView.vue', import.meta.url), 'utf8')).descriptor.scriptSetup.content
const js = ts.transpileModule(source + '\n exports.test = { health, healthSummary, webSearchStatus };',
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const context = { exports: {}, AbortController, clearInterval,
  require: (name) => name === 'vue' ? { ...require('vue'), onMounted: () => {}, onUnmounted: () => {} }
    : name === 'vue-router' ? { useRouter: () => ({}) }
    : name === '@/api/client' ? { api: async () => ({}), apiBlob: async () => new Blob() }
    : { useAuthStore: () => ({}) } }
vm.runInNewContext(js, context)

context.exports.test.health.value = {
  ready: true, documents: 920, text_documents: 918, image_only_documents: 2,
}
assert.equal(context.exports.test.healthSummary.value, '就绪·920 份文档（918 份文本、2 份仅图像）')
context.exports.test.health.value.web_search_configured = true
assert.equal(context.exports.test.webSearchStatus.value, '网页搜索：已配置')
context.exports.test.health.value.web_search_configured = false
assert.equal(context.exports.test.webSearchStatus.value, '网页搜索：未配置')
console.log('health count includes text and image-only documents')
