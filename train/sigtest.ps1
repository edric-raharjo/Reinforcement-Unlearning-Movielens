# Unified Automated Runner for Best Hardcoded Configurations across ALL Data Modalities

$Configurations = @(
    # --- Normal 1 Percent Configurations ---
    @{
        ScriptType = "Normal"; Name = "Normal_1_percent - Ye_multi"
        ForgetPct = 1; TrainLr = 0.001; Gamma = 0.97; HiddenDim = 256; TrainBatch = 2
        UnlearnLr = 0.001; UnlearnIters = 2000; LambdaRetain = 1.0; Method = "Ye_multi"
    },
    @{
        ScriptType = "Normal"; Name = "Normal_1_percent - New_True_inf"
        ForgetPct = 1; TrainLr = 0.001; Gamma = 0.98; HiddenDim = 256; TrainBatch = 2
        UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 0.8; Method = "New_True_inf"
    },
    @{
        ScriptType = "Normal"; Name = "Normal_1_percent - Gradient_Ascent"
        ForgetPct = 1; TrainLr = 0.001; Gamma = 0.97; HiddenDim = 256; TrainBatch = 2
        UnlearnLr = 0.001; UnlearnIters = 1000; LambdaRetain = 0.0; Method = "Gradient_Ascent"
    },

    # --- Demography 1 Percent Configurations ---
    @{
        ScriptType = "Demography"; Name = "Demography_1_percent - Ye_multi"
        ForgetPct = 1; TrainLr = 0.001; Gamma = 0.99; HiddenDim = 256; TrainBatch = 4
        UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 1.0; Method = "Ye_multi"
    },
    @{
        ScriptType = "Demography"; Name = "Demography_1_percent - New_True_inf"
        ForgetPct = 1; TrainLr = 0.001; Gamma = 0.99; HiddenDim = 256; TrainBatch = 4
        UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 0.7; Method = "New_True_inf"
    },
    @{
        ScriptType = "Demography"; Name = "Demography_1_percent - Gradient_Ascent"
        ForgetPct = 1; TrainLr = 0.001; Gamma = 0.98; HiddenDim = 256; TrainBatch = 2
        UnlearnLr = 0.001; UnlearnIters = 500; LambdaRetain = 0.0; Method = "Gradient_Ascent"
    },

    # --- UGP Analysis Configurations ---
    @{
        ScriptType = "UGP"; Name = "UGP_Analysis - Ye_multi"
        SettingID = 25; SettingType = "occupation"; SettingValueRaw = "15";
        TrainLr = 0.001; Gamma = 0.97; HiddenDim = 256; TrainBatch = 2
        UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 1.0; Method = "Ye_multi"
    },
    @{
        ScriptType = "UGP"; Name = "UGP_Analysis - New_True_inf"
        SettingID = 28; SettingType = "occupation"; SettingValueRaw = "19";
        TrainLr = 0.001; Gamma = 0.98; HiddenDim = 256; TrainBatch = 2
        UnlearnLr = 0.001; UnlearnIters = 1500; LambdaRetain = 0.8; Method = "New_True_inf"
    },
    @{
        ScriptType = "UGP"; Name = "UGP_Analysis - Gradient_Ascent"
        SettingID = 7; SettingType = "age"; SettingValueRaw = "45";
        TrainLr = 0.001; Gamma = 0.99; HiddenDim = 256; TrainBatch = 4
        UnlearnLr = 0.0001; UnlearnIters = 2000; LambdaRetain = 0.0; Method = "Gradient_Ascent"
    }
)

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Starting Master Automated Unlearning Sequence..." -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

foreach ($Config in $Configurations) {
    Write-Host "`n🚀 Executing: $($Config.Name)" -ForegroundColor Green
    Write-Host "---------------------------------------------------------" -ForegroundColor DarkGray
    
    # Route logic to appropriate Python Script
    $ScriptPath = ""
    $Arguments = ""
    
    if ($Config.ScriptType -eq "Normal") {
        $ScriptPath = "GPU_Enabled_Combined_MT.py"
        $Arguments = "$($Config.ForgetPct) --phase 0 --train_lr $($Config.TrainLr) --gamma $($Config.Gamma) --hidden_dim $($Config.HiddenDim) --train_batch $($Config.TrainBatch) --unlearn_lr $($Config.UnlearnLr) --unlearn_iters $($Config.UnlearnIters) --lambda_retain $($Config.LambdaRetain) --method $($Config.Method)"
    } 
    elseif ($Config.ScriptType -eq "Demography") {
        $ScriptPath = "GPU_Enabled_Combined_Demo_MT.py"
        $Arguments = "$($Config.ForgetPct) --phase 0 --train_lr $($Config.TrainLr) --gamma $($Config.Gamma) --hidden_dim $($Config.HiddenDim) --train_batch $($Config.TrainBatch) --unlearn_lr $($Config.UnlearnLr) --unlearn_iters $($Config.UnlearnIters) --lambda_retain $($Config.LambdaRetain) --method $($Config.Method)"
    } 
    elseif ($Config.ScriptType -eq "UGP") {
        $ScriptPath = "GPU_Enabled_UGP_Analysis.py"
        # We pass setting_type and setting_value_raw in addition to setting_id to guarantee explicit targeting
        $Arguments = "--setting_id $($Config.SettingID) --setting_type $($Config.SettingType) --setting_value_raw $($Config.SettingValueRaw) --train_lr $($Config.TrainLr) --gamma $($Config.Gamma) --hidden_dim $($Config.HiddenDim) --train_batch $($Config.TrainBatch) --unlearn_lr $($Config.UnlearnLr) --unlearn_iters $($Config.UnlearnIters) --lambda_retain $($Config.LambdaRetain) --method $($Config.Method)"
    }

    Write-Host "Routing to: $ScriptPath" -ForegroundColor Yellow
    Write-Host "Flags: $Arguments" -ForegroundColor DarkGray

    # Run Python Command synchronously
    Start-Process python -ArgumentList "$ScriptPath $Arguments" -NoNewWindow -Wait
}

Write-Host "`n✅ Master sequence fully complete. All scripts ran." -ForegroundColor Cyan