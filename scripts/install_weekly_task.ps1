# install_weekly_task.ps1 - Register the "BOB Weekly Slate" Windows scheduled task.
# It runs scripts\run_weekly_slate.ps1 every Sunday at 9:00 a.m. If the PC is off
# or asleep then, it runs as soon as you next log in (StartWhenAvailable).
#
# Install (once):  powershell -ExecutionPolicy Bypass -File scripts\install_weekly_task.ps1
# Test it now:     Start-ScheduledTask -TaskName "BOB Weekly Slate"
# Remove it:       Unregister-ScheduledTask -TaskName "BOB Weekly Slate" -Confirm:$false

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "run_weekly_slate.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 9:00am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "BOB Weekly Slate" -Action $action -Trigger $trigger `
    -Settings $settings -Force `
    -Description "Builds next weeks B.O.B. Me-vs-Machine slate and pushes it to GitHub." | Out-Null
Write-Output "Installed. Next run: $((Get-ScheduledTask -TaskName BOB Weekly Slate | Get-ScheduledTaskInfo).NextRunTime)"
