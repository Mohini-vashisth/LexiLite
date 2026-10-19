# Runs ON the Windows PC. Deploys whatever is on origin/main:
# pull -> install deps -> migrate -> restart the server.
# Normally invoked from the Mac via scripts/deploy.sh.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

git fetch origin
git checkout main
git pull --ff-only origin main
git log --oneline -1

$py = Join-Path $repo ".venv\Scripts\python.exe"
& $py -m pip install -q -r api\requirements.txt
Push-Location api
& $py manage.py migrate --noinput
Pop-Location

# Stop whatever is listening on :8000 (the old server), if anything.
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# Start detached via WMI so it survives the SSH session closing.
$bat = Join-Path $repo "scripts\run_server.bat"
$null = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd /c `"$bat`"" }

# Wait for the models to load, then confirm health.
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest http://127.0.0.1:8000/api/health/ -UseBasicParsing
        Write-Output "Health: $($r.Content)"
        exit 0
    } catch { }
}
Write-Error "Server did not become healthy within 2 minutes - see $env:USERPROFILE\server.err"
exit 1
