param(
    [int]$NumWorkers = 4,
    [string]$Python = "C:\Bob\.venv\Scripts\python.exe",
    [string]$BaseResultsRoot = "D:\Bob_Skripsi_Do Not Delete",
    [switch]$CleanWorkerOutputs
)

$ScriptDir = "C:\Bob\Reinforcement-Unlearning-Movielens\train"
$FinalResultsRoot = Join-Path $BaseResultsRoot "results_ugp_analysis"

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

$WorkerRoots = @()
for ($w = 0; $w -lt $NumWorkers; $w++) {
    $WorkerRoots += (Join-Path $BaseResultsRoot "results_ugp_analysis_worker$w")
}

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "        Starting UGP Single-Attribute Worker Pipeline      " -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "Script dir        : $ScriptDir" -ForegroundColor DarkGray
Write-Host "Python            : $Python" -ForegroundColor DarkGray
Write-Host "Workers           : $NumWorkers" -ForegroundColor DarkGray
Write-Host "Final results root: $FinalResultsRoot" -ForegroundColor DarkGray
Write-Host ""

if ($CleanWorkerOutputs) {
    Write-Host "Cleaning old worker output roots..." -ForegroundColor Yellow
    foreach ($root in $WorkerRoots) {
        if (Test-Path -LiteralPath $root) {
            Remove-Item -LiteralPath $root -Recurse -Force
        }
    }
    Write-Host "Worker output roots cleaned." -ForegroundColor Green
    Write-Host ""
}

Write-Host "=== PHASE 1: UGP unlearning workers ($NumWorkers workers) ===" -ForegroundColor Cyan
$jobs = @()

for ($w = 0; $w -lt $NumWorkers; $w++) {
    $WorkerRoot = $WorkerRoots[$w]
    $jobs += Start-Job -Name "ugp_worker_$w" -ScriptBlock {
        param($nw, $wid, $dir, $py, $workerRoot)

        Set-Location $dir
        $env:PYTHONHASHSEED = 0
        $env:PYTHONUTF8 = 1
        $env:PYTHONUNBUFFERED = 1
        $env:UGP_RESULTS_ROOT = $workerRoot

        Write-Host "Worker $wid writing to $workerRoot" -ForegroundColor DarkGray
        & $py -u GPU_Enabled_UGP_Analysis.py $nw $wid
        if ($LASTEXITCODE -ne 0) {
            throw "Worker $wid exited with code $LASTEXITCODE"
        }
    } -ArgumentList $NumWorkers, $w, $ScriptDir, $Python, $WorkerRoot
}

Wait-AndStreamJobs -Jobs $jobs
Write-Host "UGP worker phase complete." -ForegroundColor Green
Write-Host ""

Write-Host "=== Merging UGP worker outputs ===" -ForegroundColor Cyan
Set-Location $ScriptDir
$env:PYTHONUTF8 = 1
$env:PYTHONUNBUFFERED = 1
& $Python -u merge_ugp_worker_outputs.py $FinalResultsRoot @WorkerRoots
if ($LASTEXITCODE -ne 0) {
    throw "Merge script exited with code $LASTEXITCODE"
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "                     UGP RUN COMPLETED                     " -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
