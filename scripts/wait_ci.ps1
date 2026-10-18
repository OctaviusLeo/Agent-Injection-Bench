# Wait for the CI run on main for a commit, with double checks: a run that never starts, hangs, or fails for an
# infrastructure reason (cancelled, timed out, startup failure, or a failed setup or install step) is re-run up to
# $MaxReruns times; a run that fails in the test steps is reported with its failing lines for a real fix.
# Usage (PowerShell, repository root): pwsh scripts/wait_ci.ps1 [-Sha <commit>] [-RunMinutes 45] [-MaxReruns 2]
# Exit codes: 0 success, 1 test failure (needs a fix), 2 no run or reruns exhausted.
param([string]$Sha = "", [int]$StartMinutes = 10, [int]$RunMinutes = 45, [int]$MaxReruns = 2, [int]$PollSeconds = 30)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (-not $Sha) { $Sha = (git rev-parse HEAD).Trim() }
$repo = "OctaviusLeo/Agent-Injection-Bench"
$t0 = Get-Date; $reruns = 0; $runStart = $null; $runId = $null
while ($true) {
    $run = gh run list --repo $repo --branch main --limit 30 --json databaseId,headSha,status,conclusion,createdAt |
        ConvertFrom-Json | Where-Object { $_.headSha -eq $Sha } | Select-Object -First 1
    if (-not $run) {
        if (((Get-Date) - $t0).TotalMinutes -gt $StartMinutes) { "no CI run started for $Sha within $StartMinutes min"; exit 2 }
        Start-Sleep -Seconds $PollSeconds; continue
    }
    if ($run.databaseId -ne $runId) { $runId = $run.databaseId; $runStart = Get-Date }
    if ($run.status -eq "completed") {
        if ($run.conclusion -eq "success") { "CI run $runId on $Sha succeeded"; exit 0 }
        $jobs = gh run view $runId --repo $repo --json jobs | ConvertFrom-Json
        $failedSteps = @($jobs.jobs.steps | Where-Object { $_.conclusion -eq "failure" } | ForEach-Object { $_.name })
        $infra = $run.conclusion -in @("cancelled", "timed_out", "startup_failure", "stale") -or
                 ($failedSteps | Where-Object { $_ -match "Set up job|checkout|setup-uv|Install the locked environment" })
        if ($infra -and $reruns -lt $MaxReruns) {
            "CI run $runId ended $($run.conclusion) in [$($failedSteps -join ', ')]: infrastructure, re-running"
            gh run rerun $runId --repo $repo | Out-Null; $reruns++; $runStart = Get-Date
            Start-Sleep -Seconds $PollSeconds; continue
        }
        if ($infra) { "CI run $runId keeps failing for infrastructure reasons after $reruns re-runs"; exit 2 }
        "CI run $runId failed in [$($failedSteps -join ', ')]:"
        gh run view $runId --repo $repo --log-failed 2>&1 | Select-String -Pattern "FAILED|Error|assert|FAIL " | Select-Object -First 30
        exit 1
    }
    if (((Get-Date) - $runStart).TotalMinutes -gt $RunMinutes) {
        if ($reruns -ge $MaxReruns) { "CI run $runId still $($run.status) after $RunMinutes min and $reruns re-runs"; exit 2 }
        "CI run $runId still $($run.status) after $RunMinutes min: cancelling and re-running"
        gh run cancel $runId --repo $repo | Out-Null
        do { Start-Sleep -Seconds 15; $s = (gh run view $runId --repo $repo --json status | ConvertFrom-Json).status } while ($s -ne "completed")
        gh run rerun $runId --repo $repo | Out-Null; $reruns++; $runStart = Get-Date
    }
    Start-Sleep -Seconds $PollSeconds
}
