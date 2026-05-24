param (
    [int]$WorkerID = 0,
    [int]$NumWorkers = 1,  # Change to 5 if running 5 separate terminals
    [int]$NumRuns = 10     # How many times to test each config
)

# Unified Automated Runner for Best Hardcoded Configurations across ALL Data Modalities

$Configurations = @(
    # --- Normal 1 Percent Configurations ---
    @{ ScriptType = "Normal"; Name = "Normal_1_percent - Ye_multi"; ForgetPct = 1; TrainLr = 0.001; Gamma = 0.97; HiddenDim = 256; TrainBatch = 2; UnlearnLr = 0.001; UnlearnIters = 2000; LambdaRetain = 1.0; Method = "Ye_multi" },
    @{ ScriptType = "Normal"; Name = "Normal_1_percent - New_True_inf"; ForgetPct = 1; TrainLr = 0.001; Gamma = 0.98; HiddenDim = 256; TrainBatch = 2; UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 0.8; Method = "New_True_inf" },
    @{ ScriptType = "Normal"; Name = "Normal_1_percent - Gradient_Ascent"; ForgetPct = 1; TrainLr = 0.001; Gamma = 0.97; HiddenDim = 256; TrainBatch = 2; UnlearnLr = 0.001; UnlearnIters = 1000; LambdaRetain = 0.0; Method = "Gradient_Ascent" },

    # --- Demography 1 Percent Configurations ---
    @{ ScriptType = "Demography"; Name = "Demography_1_percent - Ye_multi"; ForgetPct = 1; TrainLr = 0.001; Gamma = 0.99; HiddenDim = 256; TrainBatch = 4; UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 1.0; Method = "Ye_multi" },
    @{ ScriptType = "Demography"; Name = "Demography_1_percent - New_True_inf"; ForgetPct = 1; TrainLr = 0.001; Gamma = 0.99; HiddenDim = 256; TrainBatch = 4; UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 0.7; Method = "New_True_inf" },
    @{ ScriptType = "Demography"; Name = "Demography_1_percent - Gradient_Ascent"; ForgetPct = 1; TrainLr = 0.001; Gamma = 0.98; HiddenDim = 256; TrainBatch = 2; UnlearnLr = 0.001; UnlearnIters = 500; LambdaRetain = 0.0; Method = "Gradient_Ascent" },

    # --- UGP Analysis Configurations ---
    @{ ScriptType = "UGP"; Name = "UGP_Analysis - Ye_multi"; SettingID = 25; SettingType = "occupation"; SettingValueRaw = "15"; TrainLr = 0.001; Gamma = 0.97; HiddenDim = 256; TrainBatch = 2; UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 1.0; Method = "Ye_multi" },
    @{ ScriptType = "UGP"; Name = "UGP_Analysis - New_True_inf"; SettingID = 28; SettingType = "occupation"; SettingValueRaw = "19"; TrainLr = 0.001; Gamma = 0.98; HiddenDim = 256; TrainBatch = 2; UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 0.8; Method = "New_True_inf" },
    @{ ScriptType = "UGP"; Name = "UGP_Analysis - Gradient_Ascent"; SettingID = 7; SettingType = "age"; SettingValueRaw = "45"; TrainLr = 0.001; Gamma = 0.99; HiddenDim = 256; TrainBatch = 4; UnlearnLr = 0.0001; UnlearnIters = 2000; LambdaRetain = 0.0; Method = "Gradient_Ascent" }
)

# Flatten Configurations X Runs into a list of tasks
$AllTasks = @()
foreach ($Config in $Configurations) {
    for ($r = 1; $r -le $NumRuns; $r++) {
        $Task = $Config.Clone()
        $Task.Add("RunIdx", $r)
        $AllTasks += $Task
    }
}

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Starting Master Unlearning Worker ($WorkerID/$NumWorkers)..." -ForegroundColor Cyan
Write-Host "Total Configs : $($Configurations.Count)" -ForegroundColor Cyan
Write-Host "Runs Per Config: $NumRuns" -ForegroundColor Cyan
Write-Host "Total Tasks   : $($AllTasks.Count)" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# Route Only Tasks Assigned to This Worker ID
for ($i = 0; $i -lt $AllTasks.Count; $i++) {
    if (($i % $NumWorkers) -ne $WorkerID) {
        continue
    }

    $Task = $AllTasks[$i]
    Write-Host "`n🚀 [Worker $WorkerID] Executing Task $($i + 1)/$($AllTasks.Count): $($Task.Name) | Run: $($Task.RunIdx)" -ForegroundColor Green
    Write-Host "---------------------------------------------------------" -ForegroundColor DarkGray
    
    $ScriptPath = ""
    $Arguments = ""
    
    if ($Task.ScriptType -eq "Normal") {
        $ScriptPath = "GPU_Enabled_Combined_MT.py"
        $Arguments = "$($Task.ForgetPct) --phase 0 --train_lr $($Task.TrainLr) --gamma $($Task.Gamma) --hidden_dim $($Task.HiddenDim) --train_batch $($Task.TrainBatch) --unlearn_lr $($Task.UnlearnLr) --unlearn_iters $($Task.UnlearnIters) --lambda_retain $($Task.LambdaRetain) --method $($Task.Method) --run_idx $($Task.RunIdx)"
    } 
    elseif ($Task.ScriptType -eq "Demography") {
        $ScriptPath = "GPU_Enabled_Combined_Demo_MT.py"
        $Arguments = "$($Task.ForgetPct) --phase 0 --train_lr $($Task.TrainLr) --gamma $($Task.Gamma) --hidden_dim $($Task.HiddenDim) --train_batch $($Task.TrainBatch) --unlearn_lr $($Task.UnlearnLr) --unlearn_iters $($Task.UnlearnIters) --lambda_retain $($Task.LambdaRetain) --method $($Task.Method) --run_idx $($Task.RunIdx)"
    } 
    elseif ($Task.ScriptType -eq "UGP") {
        $ScriptPath = "GPU_Enabled_UGP_Analysis.py"
        $Arguments = "--setting_id $($Task.SettingID) --setting_type $($Task.SettingType) --setting_value_raw $($Task.SettingValueRaw) --train_lr $($Task.TrainLr) --gamma $($Task.Gamma) --hidden_dim $($Task.HiddenDim) --train_batch $($Task.TrainBatch) --unlearn_lr $($Task.UnlearnLr) --unlearn_iters $($Task.UnlearnIters) --lambda_retain $($Task.LambdaRetain) --method $($Task.Method) --run_idx $($Task.RunIdx)"
    }

    Write-Host "Running: python $ScriptPath $Arguments" -ForegroundColor DarkGray
    Start-Process python -ArgumentList "$ScriptPath $Arguments" -NoNewWindow -Wait
}

Write-Host "`n✅ Worker $WorkerID sequence fully complete." -ForegroundColor Cyan