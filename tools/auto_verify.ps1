Write-Host "Starting Auto Verify"
$percentagesNormal = @(1, 2, 3, 4, 5, 20)
$percentagesDemo = @(1, 2, 3, 4, 5)
$thresholds = @(1, 5, 10)

Write-Host "=== VERIFY NORMAL ==="
foreach ($pct in $percentagesNormal) {
    foreach ($k in $thresholds) {
        Write-Host "Running Verify (Normal) for $pct% with threshold $k..."
        python verify.py $pct $k
    }
}

Write-Host "=== VERIFY DEMOGRAPHY ==="
foreach ($pct in $percentagesDemo) {
    foreach ($k in $thresholds) {
        Write-Host "Running Verify (Demography) for $pct% with threshold $k..."
        python verify_demo.py $pct $k
    }
}
Write-Host "Done verify"
