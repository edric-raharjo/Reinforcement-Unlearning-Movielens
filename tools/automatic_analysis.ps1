Write-Host "Starting Automatic Analysis"
$percentagesNormal = @(1, 2, 3, 4, 5, 20)
$percentagesDemo = @(1, 2, 3, 4, 5)
$thresholds = @(1, 5, 10)

$MaxJobs = 5
$CurrentDir = $PWD.Path
$global:JobList = @()

function Run-With-Concurrency {
    param(
        [scriptblock]$ScriptBlock,
        [object[]]$ArgumentList
    )
    
    while (($global:JobList | Where-Object { $_.State -eq 'Running' }).Count -ge $MaxJobs) {
        Start-Sleep -Seconds 2
    }
    
    $job = Start-Job -ScriptBlock $ScriptBlock -ArgumentList $ArgumentList
    $global:JobList += $job
}

Write-Host "=== NORMAL ==="
foreach ($pct in $percentagesNormal) {
    foreach ($k in $thresholds) {
        Write-Host "Queueing Analysis (Normal) for $pct% with threshold $k..."
        Run-With-Concurrency -ScriptBlock {
            param($pct, $k, $dir)
            Set-Location $dir
            $env:PYTHONIOENCODING="utf-8"
            $env:RUN_MODE="Normal"
            python q_analysis_detailed.py $pct $k 8 2>&1
            python q_dashboard.py $pct $k 2>&1
            python dashboard_fair.py $pct $k 10 "standard" 8 2>&1
            python diagnosis.py $pct $k 8 2>&1
        } -ArgumentList $pct, $k, $CurrentDir
    }
}

Write-Host "=== DEMOGRAPHY ==="
foreach ($pct in $percentagesDemo) {
    foreach ($k in $thresholds) {
        Write-Host "Queueing Analysis (Demography) for $pct% with threshold $k..."
        Run-With-Concurrency -ScriptBlock {
            param($pct, $k, $dir)
            Set-Location $dir
            $env:PYTHONIOENCODING="utf-8"
            $env:RUN_MODE="Demography"
            python q_analysis_detailed.py $pct $k 8 2>&1
            python q_dashboard.py $pct $k 2>&1
            python dashboard_fair.py $pct $k 10 "standard" 8 2>&1
            python diagnosis.py $pct $k 8 2>&1
        } -ArgumentList $pct, $k, $CurrentDir
    }
}

Write-Host "Waiting for all jobs to complete..."
$global:JobList | Wait-Job
$global:JobList | Receive-Job
$global:JobList | Remove-Job

Write-Host "Done analysis"
