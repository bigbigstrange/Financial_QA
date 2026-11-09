import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import ts from 'typescript'

const source = ts.transpileModule(readFileSync(new URL('../src/api/client.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText
const seen = []
let request
const encoder = new TextEncoder()
const fragments = [
  'event: started\ndata: {"run_id":"r1"}\n\nevent: no',
  'de\ndata: {"node":"retrieve","decision":"2 条证据"}\n\n',
  'event: done\ndata: {"run_id":"r1","verified":false}\n\n',
]
const context = { exports: {}, Headers, Response, TextDecoder,
  localStorage: { getItem: () => 'test-token', removeItem: () => {} },
  document: { cookie: '' },
  fetch: async (path, options) => {
    request = { path, options }
    return new Response(new ReadableStream({
      start(controller) { fragments.forEach(part => controller.enqueue(encoder.encode(part))); controller.close() },
    }))
  } }
vm.runInNewContext(source, context)
await context.exports.apiStream('/api/runs/stream', { question: '测试' }, (event, data) => seen.push([event, data]))
assert.equal(request.path, '/api/runs/stream')
assert.equal(request.options.headers.get('Authorization'), 'Bearer test-token')
assert.equal(request.options.method, 'POST')
assert.deepEqual(seen.map(([event]) => event), ['started', 'node', 'done'])
assert.equal(seen[1][1].node, 'retrieve')
assert.equal(seen[2][1].verified, false)
console.log('authenticated SSE streaming and split chunks passed')
