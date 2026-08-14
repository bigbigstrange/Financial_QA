# 启动金融问答 Web（默认演示精简模式，避免全量索引 OOM）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "安装/检查 fastapi uvicorn ..."
python -m pip install -q fastapi "uvicorn[standard]" ijson

# 演示模式默认开启；如需全量：$env:FINANCE_WEB_DEMO='0'
if (-not $env:FINANCE_WEB_DEMO) { $env:FINANCE_WEB_DEMO = "1" }

Write-Host "启动服务: http://127.0.0.1:7860  (FINANCE_WEB_DEMO=$env:FINANCE_WEB_DEMO)"
python web_app.py
