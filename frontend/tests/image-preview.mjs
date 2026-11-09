import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import { createRequire } from 'node:module'
import ts from 'typescript'
import { parse } from '@vue/compiler-sfc'

const require = createRequire(import.meta.url)
const pending = []
const revoked = []
let created = 0
let unmount
const api = { api: async () => ({}), apiBlob: (_path, options) => new Promise((resolve, reject) => pending.push({ resolve, reject, options })) }
const source = parse(readFileSync(new URL('../src/views/ChatView.vue', import.meta.url), 'utf8')).descriptor.scriptSetup.content
const js = ts.transpileModule(source + '\n exports.test = { openImage, closeImage, newChat, imagePreviews };',
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const context = { exports: {}, AbortController, console, clearInterval,
  URL: { createObjectURL: () => `blob:${++created}`, revokeObjectURL: (url) => revoked.push(url) },
  require: (name) => name === 'vue' ? { ...require('vue'), onMounted: () => {}, onUnmounted: (fn) => { unmount = fn } }
    : name === 'vue-router' ? { useRouter: () => ({}) }
    : name === '@/api/client' ? api : { useAuthStore: () => ({}) } }
vm.runInNewContext(js, context)
const ui = context.exports.test
const first = ui.openImage('a')
ui.newChat()
assert.equal(pending[0].options.signal.aborted, true)
pending[0].resolve(new Blob(['image']))
await first
assert.equal(created, 0, 'late response must not allocate a URL')
const second = ui.openImage('a')
pending[1].resolve(new Blob(['image']))
await second
assert.equal(ui.imagePreviews.value.a.url, 'blob:1')
ui.closeImage('a')
assert.deepEqual(revoked, ['blob:1'])
const failed = ui.openImage('a')
pending[2].reject(new Error('read failed'))
await failed
assert.equal(ui.imagePreviews.value.a.error, 'read failed')
const retry = ui.openImage('a')
pending[3].resolve(new Blob(['image']))
await retry
unmount()
assert.deepEqual(revoked, ['blob:1', 'blob:2'])
console.log('image lifecycle: late response, close, failure/retry, unmount passed')

const clientSource = ts.transpileModule(readFileSync(new URL('../src/api/client.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText
let requestSeen
const cleared = []
let response = new Response(new Blob(['png']), { headers: { 'content-type': 'image/png' } })
const client = { exports: {}, Headers,
  localStorage: { getItem: () => 'test-token', removeItem: key => cleared.push(key) },
  document: { cookie: '' },
  fetch: async (path, options) => { requestSeen = { path, options }; return response } }
vm.runInNewContext(clientSource, client)
assert.deepEqual(cleared, ['finance_qa_token'], 'legacy token must leave localStorage after migration')
const blob = await client.exports.apiBlob('/api/evidence/id/image')
assert.equal(await blob.text(), 'png')
assert.equal(requestSeen.options.headers.get('Authorization'), 'Bearer test-token')
assert.equal(requestSeen.path, '/api/evidence/id/image')
response = new Response(JSON.stringify({ detail: 'forbidden' }), { status: 404 })
await assert.rejects(client.exports.apiBlob('/api/evidence/id/image'), /forbidden/)
response = new Response(JSON.stringify({ text: 'original' }))
assert.equal((await client.exports.api('/api/evidence/id')).text, 'original')
console.log('client: Bearer header, blob bytes, error, old JSON passed')
