$Percentages = @(1, 2, 3, 4, 5, 20)
$NumWorkers = 4

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "      Starting Complete Training/Unlearning Pipeline       " -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Demographic Runs
Write-Host ">>> PHASE A: Demographic Unlearning" -ForegroundColor Yellow
foreach ($pct in $Percentages) {
    Write-Host "Starting Demographic for $pct`% forget set..." -ForegroundColor Green
    # Force wait until the script actually finishes before moving to the next
    .\demographic_run_parallel.ps1 -ForgetPct $pct -NumWorkers $NumWorkers
    Write-Host "Finished Demographic for $pct`%." -ForegroundColor Green
    Write-Host ""
}

# 2. Normal (Random) Runs
Write-Host ">>> PHASE B: Normal (Random) Unlearning" -ForegroundColor Yellow
foreach ($pct in $Percentages) {
    Write-Host "Starting Normal for $pct`% forget set..." -ForegroundColor Green
    .\run_parallel.ps1 -ForgetPct $pct -NumWorkers $NumWorkers
    Write-Host "Finished Normal for $pct`%." -ForegroundColor Green
    Write-Host ""
}

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "                 ALL TASKS COMPLETED!                      " -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
