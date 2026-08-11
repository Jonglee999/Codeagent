[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup','doctor','up','status','logs','down','check','core','test','lint','typecheck','integration','eval','security','verify','sandbox-build','sandbox-clean','swe-sync','swe-list','swe-status','swe-export','swe-infer','swe-setup','swe-doctor','swe-validate','swe-gold','swe-image-prepare','swe-evaluate','swe-results')]
    [string]$Command = 'status',
    [Parameter(Position = 1)]
    [string]$Argument = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$FrontendRoot = Join-Path $RepoRoot 'frontend'
$RuntimeRoot = Join-Path $RepoRoot '.harness'
$LogRoot = Join-Path $RuntimeRoot 'logs'
$PidRoot = Join-Path $RuntimeRoot 'pids'

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }
function Invoke-Checked([scriptblock]$Action) { & $Action; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Error "Required command '$Name' is not available. Run harness setup."
    }
}
function Read-Pid([string]$Name) {
    $file = Join-Path $PidRoot "$Name.pid"
    if (-not (Test-Path -LiteralPath $file)) { return $null }
    $value = (Get-Content -LiteralPath $file -Raw).Trim()
    if ($value -notmatch '^\d+$') { return $null }
    return [int]$value
}
function Test-ProcessAlive([string]$Name) {
    $processId = Read-Pid $Name
    if ($null -eq $processId) { return $false }
    return $null -ne (Get-Process -Id $processId -ErrorAction SilentlyContinue)
}
function Stop-HarnessProcess([string]$Name) {
    $processId = Read-Pid $Name
    if ($null -ne $processId) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    $pidFile = Join-Path $PidRoot "$Name.pid"
    if (Test-Path -LiteralPath $pidFile) { Remove-Item -LiteralPath $pidFile -Force }
}
function Start-HarnessProcess([string]$Name, [string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory) {
    if (Test-ProcessAlive $Name) { Write-Host "$Name already running"; return }
    New-Item -ItemType Directory -Force -Path $LogRoot, $PidRoot | Out-Null
    $stdout = Join-Path $LogRoot "$Name.out.log"
    $stderr = Join-Path $LogRoot "$Name.err.log"
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Set-Content -LiteralPath (Join-Path $PidRoot "$Name.pid") -Value $process.Id
    Write-Host "Started $Name (PID $($process.Id))"
}
function Invoke-IsolatedPytest([string[]]$Arguments) {
    $testRoot = Join-Path $RuntimeRoot ("test-runtime-" + $PID + "-" + [guid]::NewGuid().ToString('N'))
    $testState = Join-Path $testRoot 'state\product.sqlite3'
    $testRepository = Join-Path $testRoot 'repository'
    $previousState = [Environment]::GetEnvironmentVariable('CODEAGENT_STATE_DB', 'Process')
    $previousRepository = [Environment]::GetEnvironmentVariable('CODEAGENT_REPOSITORY_ROOT', 'Process')
    New-Item -ItemType Directory -Force -Path $testRepository | Out-Null
    try {
        $env:CODEAGENT_STATE_DB = $testState
        $env:CODEAGENT_REPOSITORY_ROOT = $testRepository
        & uv run pytest @Arguments
        $testExit = $LASTEXITCODE
    }
    finally {
        if ($null -eq $previousState) { Remove-Item Env:CODEAGENT_STATE_DB -ErrorAction SilentlyContinue }
        else { $env:CODEAGENT_STATE_DB = $previousState }
        if ($null -eq $previousRepository) { Remove-Item Env:CODEAGENT_REPOSITORY_ROOT -ErrorAction SilentlyContinue }
        else { $env:CODEAGENT_REPOSITORY_ROOT = $previousRepository }

        $resolvedRuntime = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\') + '\'
        $resolvedTestRoot = [System.IO.Path]::GetFullPath($testRoot)
        if (-not $resolvedTestRoot.StartsWith($resolvedRuntime, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean test path outside .harness: $resolvedTestRoot"
        }
        if (Test-Path -LiteralPath $resolvedTestRoot) {
            Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
        }
    }
    if ($testExit -ne 0) { exit $testExit }
}
function Remove-CodeAgentSandboxes([switch]$IncludeLegacy) {
    $ids = @(docker ps -aq --filter 'label=com.codeagent.managed=true' --filter 'label=com.codeagent.kind=sandbox' 2>$null)
    if ($IncludeLegacy) {
        $legacy = @(docker ps -aq --filter 'ancestor=codeagent-sandbox:latest' 2>$null)
        $ids = @($ids + $legacy | Sort-Object -Unique)
    }
    if (-not $ids) { Write-Host 'No CodeAgent sandbox containers to remove.'; return }
    Write-Host "Removing $($ids.Count) CodeAgent sandbox container(s)."
    docker container rm -f $ids | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Set-Location $RepoRoot

switch ($Command) {
    'setup' {
        Write-Step 'Checking bootstrap tools'
        Require-Command python
        Require-Command node
        Require-Command npm
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            python -m pip install uv
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        Write-Step 'Syncing locked Python dependencies'
        Invoke-Checked { uv sync --all-extras }
        Write-Step 'Installing locked frontend dependencies'
        Push-Location $FrontendRoot
        try { Invoke-Checked { npm ci } } finally { Pop-Location }
        Write-Host 'Setup complete.' -ForegroundColor Green
    }
    'doctor' {
        $failed = $false
        foreach ($tool in @('git','python','uv','node','npm','docker')) {
            if (Get-Command $tool -ErrorAction SilentlyContinue) { Write-Host "[ok] $tool" -ForegroundColor Green }
            else { Write-Host "[missing] $tool" -ForegroundColor Red; $failed = $true }
        }
        docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-Host '[ok] Docker daemon' -ForegroundColor Green }
        else { Write-Host '[missing] Docker daemon' -ForegroundColor Red; $failed = $true }
        if (Test-Path -LiteralPath '.env') { Write-Host '[ok] .env present' -ForegroundColor Green }
        else { Write-Host '[optional] .env absent; LLM-backed runs are unavailable' -ForegroundColor Yellow }
        $contextJson = uv run python -m codeagent.context_engine.capabilities
        if ($LASTEXITCODE -eq 0) {
            $context = $contextJson | ConvertFrom-Json
            if ($context.rg_available) { Write-Host '[ok] ripgrep' -ForegroundColor Green }
            else { Write-Host '[degraded] ripgrep unavailable; bounded Python search will be used' -ForegroundColor Yellow }
            if ($context.tree_sitter_languages.Count -ge 3) { Write-Host '[ok] tree-sitter parsers' -ForegroundColor Green }
            else { Write-Host '[degraded] tree-sitter parsers incomplete; regex fallback will be used' -ForegroundColor Yellow }
            if ($context.lancedb_available -and $context.lancedb_writable) { Write-Host '[ok] LanceDB runtime' -ForegroundColor Green }
            else { Write-Host '[degraded] LanceDB unavailable or index path is not writable' -ForegroundColor Yellow }
            if ($context.effective_context_budget -gt $context.model_context_window) {
                Write-Host '[invalid] context budget exceeds model context window' -ForegroundColor Red
                $failed = $true
            }
            else { Write-Host "[ok] context budget $($context.effective_context_budget)/$($context.model_context_window)" -ForegroundColor Green }
        }
        else { Write-Host '[invalid] context capability check failed' -ForegroundColor Red; $failed = $true }
        if ($failed) { exit 1 }
    }
    'up' {
        Write-Step 'Starting Redis'
        docker compose up -d redis
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $env:USE_INLINE_RUNNER = 'true'
        $env:REDIS_URL = 'redis://127.0.0.1:6379'
        $python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
        Start-HarnessProcess 'api' $python @('-m','uvicorn','codeagent.interaction.api.main:app','--host','127.0.0.1','--port','8000') $RepoRoot
        Start-HarnessProcess 'frontend' (Get-Command npm.cmd).Source @('run','dev','--','--host','127.0.0.1') $FrontendRoot
        Start-Sleep -Seconds 2
        & $PSCommandPath status
    }
    'status' {
        $api = Test-ProcessAlive 'api'
        $frontend = Test-ProcessAlive 'frontend'
        $redis = docker compose ps --status running --services 2>$null | Select-String -SimpleMatch 'redis'
        Write-Host ("API      : " + $(if ($api) {'running http://127.0.0.1:8000/docs'} else {'stopped'}))
        Write-Host ("Frontend : " + $(if ($frontend) {'running http://127.0.0.1:5173'} else {'stopped'}))
        Write-Host ("Redis    : " + $(if ($redis) {'running'} else {'stopped'}))
    }
    'logs' {
        $service = if ($Argument) { $Argument } else { 'api' }
        if ($service -eq 'redis') { docker compose logs --tail 150 redis; break }
        $files = Get-ChildItem -LiteralPath $LogRoot -Filter "$service.*.log" -ErrorAction SilentlyContinue
        if (-not $files) { Write-Host "No logs for $service"; break }
        $files | ForEach-Object { Write-Step $_.Name; Get-Content -LiteralPath $_.FullName -Tail 150 }
    }
    'down' {
        Stop-HarnessProcess 'frontend'
        Stop-HarnessProcess 'api'
        docker compose stop redis | Out-Host
        Remove-CodeAgentSandboxes
        Write-Host 'Local services stopped.' -ForegroundColor Green
    }
    'check' {
        Write-Step 'Python compile check'
        Invoke-Checked { uv run python -m compileall -q codeagent scripts }
        Write-Step 'Frontend lint and production build'
        Push-Location $FrontendRoot
        try { Invoke-Checked { npm run lint }; Invoke-Checked { npm run build } } finally { Pop-Location }
        Write-Step 'Compose contract'
        Invoke-Checked { docker compose config --quiet }
    }
    'test' {
        if ($Argument) {
            Invoke-IsolatedPytest @($Argument, '-q', '-p', 'no:cacheprovider')
            break
        }
        Invoke-IsolatedPytest @('tests/unit', '-q', '-p', 'no:cacheprovider')
        Write-Step 'Frontend component tests'
        Push-Location $FrontendRoot
        try { Invoke-Checked { npm run test } } finally { Pop-Location }
    }
    'core' {
        Write-Step 'Risk-focused Agent core tests'
        Invoke-IsolatedPytest @(
            'tests/unit/api',
            'tests/unit/gateway/test_orchestration_gateway.py',
            'tests/unit/gateway/test_orchestration_gateway_impl.py',
            'tests/unit/gateway/test_model_gateway.py',
            'tests/unit/gateway/test_redis_resilience.py',
            'tests/unit/extensions',
            'tests/unit/memory/test_transcript.py',
            'tests/unit/orchestration',
            'tests/unit/tools',
            'tests/unit/sandbox',
            'tests/unit/test_workspaces.py',
            'tests/unit/walking_skeleton',
            '-q', '-p', 'no:cacheprovider'
        )
        Write-Step 'Frontend component tests'
        Push-Location $FrontendRoot
        try { Invoke-Checked { npm run test } } finally { Pop-Location }
    }
    'lint' {
        uv run ruff check codeagent tests scripts
        $pythonExit = $LASTEXITCODE
        Push-Location $FrontendRoot
        try { npm run lint; $frontendExit = $LASTEXITCODE } finally { Pop-Location }
        if ($pythonExit -ne 0) { exit $pythonExit }
        exit $frontendExit
    }
    'typecheck' {
        uv run mypy codeagent
        $pythonExit = $LASTEXITCODE
        Push-Location $FrontendRoot
        try { npm run build; $frontendExit = $LASTEXITCODE } finally { Pop-Location }
        if ($pythonExit -ne 0) { exit $pythonExit }
        exit $frontendExit
    }
    'integration' {
        Invoke-IsolatedPytest @('tests/integration', '-q', '-p', 'no:cacheprovider')
    }
    'eval' {
        Invoke-IsolatedPytest @(
            'tests/unit/harness', 'tests/unit/walking_skeleton',
            '-q', '-p', 'no:cacheprovider'
        )
    }
    'security' {
        Invoke-Checked { uv run bandit -q -r codeagent -ll }
        Push-Location $FrontendRoot
        try { Invoke-Checked { npm audit --audit-level=high } } finally { Pop-Location }
    }
    'verify' {
        foreach ($gate in @('doctor','check','test','eval','security')) {
            Write-Step "verify: $gate"
            & $PSCommandPath $gate
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        Write-Host 'All enforceable base gates passed.' -ForegroundColor Green
    }
    'sandbox-build' {
        Write-Step 'Building the locked local execution sandbox'
        Invoke-Checked { docker build -f Dockerfile.sandbox -t codeagent-sandbox:latest . }
        Write-Host 'Sandbox image ready: codeagent-sandbox:latest' -ForegroundColor Green
    }
    'sandbox-clean' {
        Write-Step 'Removing CodeAgent-owned sandbox containers'
        Remove-CodeAgentSandboxes -IncludeLegacy
    }
    'swe-sync' {
        Invoke-Checked { uv run python scripts/sync_swe_smoke.py evals/swe_smoke/tasks.json }
    }
    'swe-list' {
        $manifest = Get-Content -LiteralPath 'evals\swe_smoke\tasks.json' -Raw | ConvertFrom-Json
        $manifest.tasks | Select-Object instance_id, repo, gold_patch_changed_lines | Format-Table -AutoSize
    }
    'swe-status' {
        Invoke-Checked { uv run python scripts/export_swe_predictions.py --status }
    }
    'swe-export' {
        $destination = if ($Argument) { $Argument } else { '.codeagent\benchmarks\predictions.jsonl' }
        Invoke-Checked { uv run python scripts/export_swe_predictions.py $destination }
    }
    'swe-infer' {
        if ($env:CONFIRM_LLM_API_COST -ne 'true') {
            Write-Error 'Set CONFIRM_LLM_API_COST=true to acknowledge real LLM API usage.'
        }
        $taskSet = if ($Argument) { $Argument } else { 'smoke-5' }
        Invoke-Checked {
            uv run python scripts/run_swe_inference.py --set $taskSet --confirm-api-cost
        }
    }
    'swe-setup' {
        Invoke-Checked { uv run python scripts/swe_evaluator.py setup }
    }
    'swe-doctor' {
        Invoke-Checked { uv run python scripts/swe_evaluator.py doctor }
    }
    'swe-validate' {
        $predictionFile = if ($Argument) { $Argument } else { '.codeagent\benchmarks\predictions.jsonl' }
        Invoke-Checked {
            uv run python scripts/swe_evaluator.py validate --predictions $predictionFile
        }
    }
    'swe-gold' {
        if ($env:CONFIRM_SWE_DOCKER_RESOURCES -ne 'true') {
            Write-Error 'Set CONFIRM_SWE_DOCKER_RESOURCES=true to acknowledge official image downloads and Docker resource usage.'
        }
        if (-not $Argument) {
            Write-Error 'Provide one allow-listed SWE-bench instance ID as the second argument.'
        }
        Invoke-Checked {
            uv run python scripts/swe_evaluator.py gold --instance $Argument --confirm-resources
        }
    }
    'swe-image-prepare' {
        if ($env:CONFIRM_SWE_DOCKER_RESOURCES -ne 'true') {
            Write-Error 'Set CONFIRM_SWE_DOCKER_RESOURCES=true to acknowledge official image downloads and Docker resource usage.'
        }
        if (-not $Argument) {
            Write-Error 'Provide one allow-listed SWE-bench instance ID as the second argument.'
        }
        Invoke-Checked {
            uv run python scripts/swe_evaluator.py image-prepare --instance $Argument --confirm-resources
        }
    }
    'swe-evaluate' {
        if ($env:CONFIRM_SWE_DOCKER_RESOURCES -ne 'true') {
            Write-Error 'Set CONFIRM_SWE_DOCKER_RESOURCES=true to acknowledge official image downloads and Docker resource usage.'
        }
        $parameters = @('scripts/swe_evaluator.py','evaluate','--confirm-resources')
        if ($Argument) { $parameters += @('--instance', $Argument) }
        Invoke-Checked { uv run python @parameters }
    }
    'swe-results' {
        Invoke-Checked { uv run python scripts/swe_evaluator.py results }
    }
}
