Write-Host "Starting Automatic Analysis"
$percentagesNormal = @(1, 2, 3, 4, 5, 20)
$percentagesDemo = @(1, 2, 3, 4, 5)
$thresholds = @(1, 5, 10)

Write-Host "=== NORMAL ==="
foreach ($pct in $percentagesNormal) {
    foreach ($k in $thresholds) {
        Write-Host "Running Analysis (Normal) for $pct% with threshold $k..."
        $env:RUN_MODE="Normal"
        python q_analysis_detailed.py $pct $k
        python q_dashboard.py $pct $k
        python dashboard_fair.py $pct $k
        python diagnosis.py $pct $k
    }
}

Write-Host "=== DEMOGRAPHY ==="
foreach ($pct in $percentagesDemo) {
    foreach ($k in $thresholds) {
        Write-Host "Running Analysis (Demography) for $pct% with threshold $k..."
        $env:RUN_MODE="Demography"
        python q_analysis_detailed.py $pct $k
        python q_dashboard.py $pct $k
        python dashboard_fair.py $pct $k
        python diagnosis.py $pct $k
    }
}
Write-Host "Done analysis"
