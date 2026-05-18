Write-Host "=== CodeAgent 进程清理 ===" -ForegroundColor Cyan

# 1. 清理 Celery Worker (按命令行匹配)
Write-Host "[1/3] 清理 Celery worker..." -NoNewline
$celery = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'celery.*codeagent\.worker' }
if ($celery) {
    $celery | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host " 清理了 $($celery.Count) 个进程" -ForegroundColor Yellow
} else {
    Write-Host " 无残留" -ForegroundColor Green
}

# 2. 清理 Uvicorn (按端口 8000/8001 或按命令行)
Write-Host "[2/3] 清理 Uvicorn 后端..." -NoNewline
$uvicorn = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'uvicorn.*codeagent' }
if ($uvicorn) {
    $uvicorn | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host " 清理了 $($uvicorn.Count) 个进程" -ForegroundColor Yellow
} else {
    Write-Host " 无残留" -ForegroundColor Green
}

# 3. 清理 Vite 前端 (按命令行匹配)
Write-Host "[3/3] 清理 Vite 前端..." -NoNewline
$vite = Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" | Where-Object { $_.CommandLine -match 'vite' }
if ($vite) {
    $vite | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host " 清理了 $($vite.Count) 个进程" -ForegroundColor Yellow
} else {
    Write-Host " 无残留" -ForegroundColor Green
}

# 验证
Start-Sleep 1
$remaining = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'celery|uvicorn' }
$nodeRemaining = Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" | Where-Object { $_.CommandLine -match 'vite' }
if (-not $remaining -and -not $nodeRemaining) {
    Write-Host "`n=== 全部清理完成 ===" -ForegroundColor Green
} else {
    Write-Host "`n=== 部分进程未能清理，请手动检查 ===" -ForegroundColor Red
}
