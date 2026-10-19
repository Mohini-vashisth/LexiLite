# Runs ON the Windows PC. Deploys a branch (default: main):
# pull -> install deps -> migrate -> (re)start the server -> verify it STAYS up.
# Normally invoked from the Mac via scripts/deploy.sh.
#
# Reliability notes (a first deploy once left the server dead although the log
# showed it had answered /api/health/):
#   - health is checked with curl.exe and a hard timeout, not Invoke-WebRequest
#   - after the first healthy answer we wait and check AGAIN, so a server that
#     starts and then dies is reported as a failure, not a success
#   - if the server isn't healthy-and-stable, we kill it and retry the launch once
#   - everything is also written to deploy.log in the repo's parent logs folder
param([string]$Branch = "main")

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$logFile = Join-Path $env:USERPROFILE "deploy.log"

function Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $logFile -Value $line
}

function Test-Health {
    # -f: HTTP errors (e.g. 503 "degraded") count as failure. -m 5: never hang.
    $out = & curl.exe -s -f -m 5 http://127.0.0.1:8000/api/health/ 2>$null
    if ($LASTEXITCODE -eq 0 -and $out -match '"status":"healthy"') { return $out }
    return $null
}

function Stop-Server {
    Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
}

function Start-Server {
    $bat = Join-Path $repo "scripts\run_server.bat"
    # WMI so the process is NOT a child of this (SSH) session and survives it closing.
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"$bat`"" }
    Log "Launched server (WMI return=$($r.ReturnValue), pid=$($r.ProcessId))"
}

function Wait-Healthy([int]$timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Health) {
            # Healthy once is not enough: confirm it is still up a few seconds later.
            Start-Sleep -Seconds 8
            $again = Test-Health
            if ($again) { return $again }
            Log "Server answered once, then stopped answering - treating as failed"
            return $null
        }
        Start-Sleep -Seconds 3
    }
    return $null
}

Log "=== deploy start (branch: $Branch) ==="
git fetch origin
git checkout $Branch
git pull --ff-only origin $Branch
Log ("Code at: " + (git log --oneline -1))

$py = Join-Path $repo ".venv\Scripts\python.exe"
& $py -m pip install -q -r api\requirements.txt
Push-Location api
& $py manage.py migrate --noinput
Pop-Location

$healthy = $null
foreach ($attempt in 1, 2) {
    Log "Starting server (attempt $attempt of 2)"
    Stop-Server
    Start-Server
    $healthy = Wait-Healthy 150
    if ($healthy) { break }
    Log "Attempt $attempt failed. Last server.err lines:"
    Get-Content (Join-Path $env:USERPROFILE "server.err") -Tail 8 -ErrorAction SilentlyContinue |
        ForEach-Object { Log "  $_" }
}

if ($healthy) {
    Log "Health: $healthy"
    Log "=== deploy OK ==="
    exit 0
}
Log "=== deploy FAILED: server not healthy after 2 attempts (see $env:USERPROFILE\server.err) ==="
exit 1
