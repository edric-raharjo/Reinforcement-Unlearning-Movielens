param(
    [int]$ForgetPct = 20,
    [int]$NumWorkers = 8
)

$ScriptDir = "C:\Bob\train"
$Python    = "C:\Bob\.venv\Scripts\python.exe"

Write-Host "=== PHASE 1: Training ($NumWorkers workers) ===" -ForegroundColor Cyan
$jobs = @()
for ($w = 0; $w -lt $NumWorkers; $w++) {
    $jobs += Start-Job -ScriptBlock {
        param($fp, $nw, $wid, $dir, $py)
        Set-Location $dir
        $env:PYTHONHASHSEED = 0
        $env:PYTHONUTF8 = 1
        & $py GPU_Enabled_Combined_MT.py $fp $nw $wid 1
    } -ArgumentList $ForgetPct, $NumWorkers, $w, $ScriptDir, $Python
}
$jobs | Wait-Job | Receive-Job
Write-Host "Phase 1 complete." -ForegroundColor Green

Write-Host "=== Merging Phase 1 results ===" -ForegroundColor Cyan
Set-Location $ScriptDir
$env:PYTHONUTF8 = 1
& $Python merge_results.py $ForgetPct

# Wait for file system to flush merged results before Phase 2 workers check for it
Start-Sleep -Seconds 5

Write-Host "=== PHASE 2: Unlearning ($NumWorkers workers) ===" -ForegroundColor Cyan
$jobs = @()
for ($w = 0; $w -lt $NumWorkers; $w++) {
    $jobs += Start-Job -ScriptBlock {
        param($fp, $nw, $wid, $dir, $py)
        Set-Location $dir
        $env:PYTHONHASHSEED = 0
        $env:PYTHONUTF8 = 1
        & $py GPU_Enabled_Combined_MT.py $fp $nw $wid 2
    } -ArgumentList $ForgetPct, $NumWorkers, $w, $ScriptDir, $Python
}
$jobs | Wait-Job | Receive-Job
Write-Host "Phase 2 complete." -ForegroundColor Green

Write-Host "=== Merging final results ===" -ForegroundColor Cyan
& $Python merge_results.py $ForgetPct
Write-Host "All done!" -ForegroundColor Green
