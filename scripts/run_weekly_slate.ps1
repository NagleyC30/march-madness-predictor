# run_weekly_slate.ps1 - Build and publish next week's Me-vs-Machine slate from
# this PC. Used by the "BOB Weekly Slate" scheduled task (install_weekly_task.ps1),
# or run it by hand:  powershell -ExecutionPolicy Bypass -File scripts\run_weekly_slate.ps1
# Extra arguments go to weekly_slate.py, e.g.  ... -File scripts\run_weekly_slate.ps1 --dry-run
#
# Steps: switch to main and pull, run weekly_slate.py with the project venv, then
# commit and push data/slates so Streamlit Cloud redeploys. Everything is logged
# to logs\weekly_slate.log (git-ignored).
#
# Native commands (git, python) are checked with $LASTEXITCODE, not exceptions:
# git writes progress to stderr, which Windows PowerShell 5.1 would otherwise
# treat as an error. "| Out-Host" puts their output into the transcript log.

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path (Join-Path $repo "logs") | Out-Null
Start-Transcript -Path (Join-Path $repo "logs\weekly_slate.log") -Append | Out-Null
$code = 0
try {
    Write-Output "=== Weekly slate run: $(Get-Date -Format s) ==="
    git checkout main | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "git checkout main failed (commit or stash your changes first)" }
    git pull --ff-only | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "git pull failed" }

    & (Join-Path $repo "venv\Scripts\python.exe") weekly_slate.py @args | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "weekly_slate.py failed (exit $LASTEXITCODE) - see the output above" }

    git add data/slates | Out-Host
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Output "No new slate to commit."
    } else {
        git commit -m "Weekly slate (built on PC)" | Out-Host
        git push | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "git push failed" }
        Write-Output "Slate pushed. Streamlit will redeploy in a minute or two."
    }
} catch {
    Write-Output "FAILED: $_"
    $code = 1
} finally {
    Stop-Transcript | Out-Null
}
exit $code
