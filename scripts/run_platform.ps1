# 启动企业平台 API（默认演示内存模式）
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $env:FINANCE_WEB_DEMO) { $env:FINANCE_WEB_DEMO = "1" }

Write-Host "Starting finance QA platform on http://127.0.0.1:8000 ..."
python -m app.main
