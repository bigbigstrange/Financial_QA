# 阶段 0 基线运行手册

## 前置条件

- 使用 Python 3.11。
- 金融项目已安装 `requirements.txt`；Agent 项目已在 `.venv` 安装 `pyproject.toml` 所需依赖。
- 基线产物只保存在 `output/baseline/`，不得提交或对外发送原始内容。

## 路径

```powershell
$FinanceRoot = 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'
$AgentRoot = 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag'
$FinancePython = Join-Path $FinanceRoot '.venv\Scripts\python.exe'
$AgentPython = Join-Path $AgentRoot '.venv\Scripts\python.exe'
$env:HF_ENDPOINT = 'https://huggingface.co'
$env:HF_HOME = Join-Path (Split-Path $FinanceRoot -Parent) 'model_cache\huggingface'
```

首次创建环境：

```powershell
Set-Location $FinanceRoot
uv venv .venv --python 3.11
uv pip install --python $FinancePython -r requirements.txt
uv pip install --python $FinancePython hf-xet
& $FinancePython -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3')"
```

## 离线硬门禁

```powershell
Set-Location $FinanceRoot
& $FinancePython -m unittest discover -s tests -v
& $FinancePython scripts/capture_phase0_manifest.py --agent-root $AgentRoot
& $FinancePython scripts/audit_parser_baseline.py

Set-Location $AgentRoot
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $AgentPython -m unittest graph.tests.test_graph_routes -v
```

`audit_parser_baseline.py` 退出 2 表示当前解析基线发现真实缺陷，报告仍然有效；必须保留问题，不能改成退出 0。Agent 离线测试不需要 API Key，也不会加载 Embedding 模型。

## 三案例 smoke

```powershell
Set-Location $FinanceRoot
$env:FINANCE_MAX_BLOCKS = '20'
$env:FINANCE_SKIP_IMAGE = '1'
$env:FINANCE_SKIP_GEN = '1'
$env:FINANCE_EMBED_BATCH_SIZE = '16'
& $FinancePython scripts/benchmark_regression.py `
  --kb-path output/knowledge_base.json `
  --test-set test/merged.xlsx `
  --output output/baseline/qa_report.json `
  --final-output output/baseline/final_report.json `
  --max-samples 3
```

模型或缓存不可用时，命令仍会输出三个错误案例。此时 `error` 字段记录共同的初始化原因，不能把这次运行解释为问答效果通过。
三案例 smoke 使用 20 个块和纯文本检索，目的只是在 CPU 环境快速验证完整链路；它不能用于判断召回或回答效果，也不能替代下方 2,000 块性能检查或全量基线。

需要检查演示规模性能时，将 `FINANCE_MAX_BLOCKS` 改为 `2000`；当前机器 CPU 首次编码会超过 10 分钟。

## 全量基线

```powershell
Set-Location $FinanceRoot
$BaselineParse = Join-Path $FinanceRoot 'output/baseline/parsed'
$BaselineTemp = Join-Path $FinanceRoot 'output/baseline/temp'

& $FinancePython scripts/preprocess.py `
  --input data/documents `
  --output $BaselineParse `
  --tesseract-cmd "" `
  --temp-dir $BaselineTemp

& $FinancePython scripts/audit_parser_baseline.py `
  --kb-path "$BaselineParse/knowledge_base.json" `
  --parser-report "$BaselineParse/parser_report.json" `
  --output output/baseline/parser_audit_full.json

& $FinancePython scripts/benchmark_regression.py `
  --kb-path "$BaselineParse/knowledge_base.json" `
  --test-set test/merged.xlsx `
  --output output/baseline/qa_report_full.json `
  --final-output output/baseline/final_report_full.json `
  --max-samples 0
```

## 在线集成测试

```powershell
Set-Location $AgentRoot
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:TRANSFORMERS_OFFLINE -ErrorAction SilentlyContinue
& $AgentPython -m pytest graph/chains/tests/test_chains.py -q
```

该测试需要完整的 `all-MiniLM-L6-v2` 缓存或可用的 Hugging Face 网络访问，以及有效的模型 API Key。缺少条件时记为 `blocked_external_dependency`，不能记为代码失败或成功。

## 产物

- `output/baseline/manifest.json`
- `output/baseline/parser_audit.json`
- `output/baseline/qa_report.json`
- `output/baseline/final_report.json`
- 全量运行对应的 `*_full.json`

这些文件包含本地数据、问题、答案和环境信息，不进入 Git。对外展示前需要脱敏。

## 判读规则

- `evidence_response_rate` 只说明系统返回了非空答案和证据，不等于正确率。
- 解析审计失败是后续阶段的输入，不应修改报告来伪造通过。
- QA 案例的 `status=error` 表示运行受阻，不进入效果结论。
- 在线链测试缺少模型、缓存或 API Key 时标记为外部依赖阻塞；离线测试必须通过。
- manifest 只记录秘密是否配置，不记录秘密值。
