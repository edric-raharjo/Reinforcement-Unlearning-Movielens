param(
    [int]$ForgetPct = 20,
    [int]$NumWorkers = 8
)

$ScriptDir = "C:\Bob\train"
$Python    = "C:\Bob\.venv\Scripts\python.exe"

function Wait-AndStreamJobs {
    param(
        [Parameter(Mandatory = $true)]
        [array]$Jobs,

        [int]$PollMs = 500
    )

    while (($Jobs | Where-Object { $_.State -in 'NotStarted', 'Running' }).Count -gt 0) {
        foreach ($job in $Jobs) {
            Receive-Job -Job $job
        }
        Start-Sleep -Milliseconds $PollMs
    }

    foreach ($job in $Jobs) {
        Receive-Job -Job $job
    }

    $failed = $Jobs | Where-Object { $_.State -eq 'Failed' }
    if ($failed.Count -gt 0) {
        Write-Host "`nOne or more jobs failed." -ForegroundColor Red
        foreach ($job in $failed) {
            Write-Host "Failed job: $($job.Id) / $($job.Name)" -ForegroundColor Red
        }
        $Jobs | Remove-Job -Force
        throw "Stopping because at least one background job failed."
    }

    $Jobs | Remove-Job -Force
}

Write-Host "=== PHASE 1: Training ($NumWorkers workers) ===" -ForegroundColor Cyan
$jobs = @()

for ($w = 0; $w -lt $NumWorkers; $w++) {
    $jobs += Start-Job -Name "phase1_worker_$w" -ScriptBlock {
        param($fp, $nw, $wid, $dir, $py)

        Set-Location $dir
        $env:PYTHONHASHSEED = 0
        $env:PYTHONUTF8 = 1
        $env:PYTHONUNBUFFERED = 1

        & $py -u GPU_Enabled_Combined_MT.py $fp $nw $wid 1
    } -ArgumentList $ForgetPct, $NumWorkers, $w, $ScriptDir, $Python
}

Wait-AndStreamJobs -Jobs $jobs
Write-Host "Phase 1 complete." -ForegroundColor Green

Write-Host "=== Merging Phase 1 results ===" -ForegroundColor Cyan
Set-Location $ScriptDir
$env:PYTHONUTF8 = 1
$env:PYTHONUNBUFFERED = 1
& $Python -u merge_results.py $ForgetPct 1

Start-Sleep -Seconds 5

Write-Host "=== PHASE 2: Unlearning ($NumWorkers workers) ===" -ForegroundColor Cyan
$jobs = @()

for ($w = 0; $w -lt $NumWorkers; $w++) {
    $jobs += Start-Job -Name "phase2_worker_$w" -ScriptBlock {
        param($fp, $nw, $wid, $dir, $py)

        Set-Location $dir
        $env:PYTHONHASHSEED = 0
        $env:PYTHONUTF8 = 1
        $env:PYTHONUNBUFFERED = 1

        & $py -u GPU_Enabled_Combined_MT.py $fp $nw $wid 2
    } -ArgumentList $ForgetPct, $NumWorkers, $w, $ScriptDir, $Python
}

Wait-AndStreamJobs -Jobs $jobs
Write-Host "Phase 2 complete." -ForegroundColor Green

Write-Host "=== Merging final results ===" -ForegroundColor Cyan
$env:PYTHONUTF8 = 1
$env:PYTHONUNBUFFERED = 1
& $Python -u merge_results.py $ForgetPct 1

Write-Host "All done!" -ForegroundColor Green
