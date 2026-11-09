param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw "Port $Port already has a listener. Do not start a duplicate API."
}
$python = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create the Python 3.11 .venv first.' }
$env:PORT = "$Port"
$env:HOST = '127.0.0.1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$logRoot = Join-Path $taskRoot 'output\logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$process = Start-Process -FilePath $python -ArgumentList '-u','-m','app.main' `
    -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot "service-$stamp.out.log") `
    -RedirectStandardError (Join-Path $logRoot "service-$stamp.err.log")
Write-Output "API launcher PID: $($process.Id); URL: http://127.0.0.1:$Port/; logs: output/logs/service-$stamp.*.log"
