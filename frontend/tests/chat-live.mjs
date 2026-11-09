import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
import { randomUUID } from 'node:crypto'
import ts from 'typescript'
import { parse } from '@vue/compiler-sfc'

const require = createRequire(import.meta.url)
const source = parse(readFileSync(new URL('../src/views/ChatView.vue', import.meta.url), 'utf8')).descriptor.scriptSetup.content
const js = ts.transpileModule(source + '\n exports.test = { ask, livePilot, messages, newChat, conversationId };',
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
let release
let wait = new Promise(resolve => { release = resolve })
const api = {
  apiBlob: async () => new Blob(),
  api: async path => {
    if (path === '/api/chat') {
      await wait
      return { conversation_id: 999, assistant_message: { content: 'late answer', evidence: [] } }
    }
    if (path === '/api/runs/r1') return { run_id: 'r1', status: 'completed', answer: '证据不足', evidence: [],
      verified: false, trace: [{ node: 'retrieve', decision: '0 条' }], model_snapshot: { profile_name: 'pilot' } }
    return {}
  },
  apiStream: async (_path, _body, onEvent) => {
    onEvent('started', { run_id: 'r1' })
    onEvent('node', { node: 'retrieve', decision: '0 条', elapsed_ms: 4 })
    await wait
    onEvent('done', { run_id: 'r1', verified: false })
  },
}
const context = { exports: {}, AbortController, console, clearInterval, setInterval,
  crypto: { randomUUID },
  sessionStorage: { setItem: () => {}, removeItem: () => {}, getItem: () => null },
  require: name => name === 'vue' ? { ...require('vue'), onMounted: () => {}, onUnmounted: () => {} }
    : name === 'vue-router' ? { useRouter: () => ({}) }
    : name === '@/api/client' ? api : { useAuthStore: () => ({}) } }
vm.runInNewContext(js, context)
const ui = context.exports.test
ui.livePilot.value = true
const run = ui.ask('基金认购')
await Promise.resolve()
assert.equal(ui.messages.value[1].trace.trace[0].node, 'retrieve', 'node renders before completion')
release()
await run
assert.equal(ui.messages.value[1].verified, false)
assert.equal(ui.messages.value[1].content, '证据不足')
console.log('live nodes before done and unverified result passed')

wait = new Promise(resolve => { release = resolve })
const oldLive = ui.ask('旧对话问题')
await Promise.resolve()
ui.newChat()
release()
await oldLive
assert.equal(ui.messages.value.length, 0, 'late SSE must not enter new chat or throw on cleared reply')

ui.livePilot.value = false
wait = new Promise(resolve => { release = resolve })
const oldChat = ui.ask('旧普通问答')
ui.newChat()
release()
await oldChat
assert.equal(ui.messages.value.length, 0, 'late HTTP answer must not enter new chat')
assert.equal(ui.conversationId.value, null, 'old HTTP result must not restore old conversation ID')
console.log('late SSE and HTTP after new chat are discarded')
