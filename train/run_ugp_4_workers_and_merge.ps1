param(
    [string]$PythonExe = "py",
    [string]$RepoRoot = "C:\Bob",
    [string]$BaseRoot = "D:\Bob_Skripsi_Do Not Delete",
    [switch]$CleanWorkerOutputs
)

$ErrorActionPreference = "Stop"

$scriptRel = "GPU_Enabled_UGP_Analysis.py"
$mergeRel = "merge_ugp_worker_outputs.py"

$finalRoot = Join-Path $BaseRoot "results_ugp_analysis"
$workerRoots = 0..3 | ForEach-Object { Join-Path $BaseRoot ("results_ugp_analysis_worker{0}" -f $_) }

if ($CleanWorkerOutputs) {
    foreach ($workerRoot in $workerRoots) {
        if (Test-Path $workerRoot) {
            Remove-Item -LiteralPath $workerRoot -Recurse -Force
        }
    }
}

Push-Location $RepoRoot
try {
    $processes = @()
    foreach ($workerId in 0..3) {
        $workerRoot = $workerRoots[$workerId]
        New-Item -ItemType Directory -Force -Path $workerRoot | Out-Null
        $cmd = '$env:UGP_RESULTS_ROOT="{0}"; {1} "{2}" 4 {3}' -f $workerRoot, $PythonExe, $scriptRel, $workerId
        Write-Host "Starting worker $workerId -> $workerRoot"
        $p = Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList @("-NoProfile", "-Command", $cmd) `
            -PassThru
        $processes += $p
    }

    Write-Host "Waiting for all 4 workers to finish..."
    $processes | Wait-Process

    Write-Host "All workers finished. Running merge..."
    $mergeCmd = '{0} "{1}" "{2}" "{3}" "{4}" "{5}" "{6}"' -f `
        $PythonExe, `
        $mergeRel, `
        $finalRoot, `
        $workerRoots[0], `
        $workerRoots[1], `
        $workerRoots[2], `
        $workerRoots[3]
    & powershell.exe -NoProfile -Command $mergeCmd
}
finally {
    Pop-Location
}
