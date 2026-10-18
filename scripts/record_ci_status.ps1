# Record HEAD and the latest CI run on main for the M5 gate (gh runs only on Windows).
# Usage (PowerShell, repository root): pwsh scripts/record_ci_status.ps1
# Writes runs/head_commit.txt and runs/ci_latest.json (both gitignored).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Force runs | Out-Null
$head = (git rev-parse HEAD).Trim()
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "runs/head_commit.txt"), "$head`n", $utf8)
$runs = gh run list --repo OctaviusLeo/Agent-Injection-Bench --branch main --limit 20 --json databaseId,headSha,status,conclusion,createdAt | ConvertFrom-Json
$forHead = $runs | Where-Object { $_.headSha -eq $head } | Select-Object -First 1
if (-not $forHead) { $forHead = [pscustomobject]@{ headSha = ""; status = "missing"; conclusion = "" } }
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "runs/ci_latest.json"), ($forHead | ConvertTo-Json -Compress), $utf8)
"HEAD $head; CI run for HEAD: $($forHead.databaseId) $($forHead.status) $($forHead.conclusion)"
