Write-Host "Starting Auto Verify"
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

Write-Host "=== VERIFY NORMAL ==="
foreach ($pct in $percentagesNormal) {
    foreach ($k in $thresholds) {
        Write-Host "Queueing Verify (Normal) for $pct% with threshold $k..."
        Run-With-Concurrency -ScriptBlock {
            param($pct, $k, $dir)
            Set-Location $dir
            python verify.py $pct $k
        } -ArgumentList $pct, $k, $CurrentDir
    }
}

Write-Host "=== VERIFY DEMOGRAPHY ==="
foreach ($pct in $percentagesDemo) {
    foreach ($k in $thresholds) {
        Write-Host "Queueing Verify (Demography) for $pct% with threshold $k..."
        Run-With-Concurrency -ScriptBlock {
            param($pct, $k, $dir)
            Set-Location $dir
            python verify_demo.py $pct $k
        } -ArgumentList $pct, $k, $CurrentDir
    }
}

Write-Host "Waiting for all jobs to complete..."
if ($global:JobList.Count -gt 0) {
    $global:JobList | Wait-Job
    $global:JobList | Receive-Job
    $global:JobList | Remove-Job
}

Write-Host "Done verify"

Write-Host "=== VERIFICATION SUMMARY ==="
python check_verifications.py
