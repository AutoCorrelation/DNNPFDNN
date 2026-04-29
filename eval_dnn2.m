%% DNN2 Evaluation (PF-DNN2 + LS-DNN2)
% Evaluate both postprocessor branches:
%   - DNN1-PF-DNN2
%   - DNN1-LS-DNN2
%
% Compare against first-stage PF / LS baselines.

clear; clc; rng(42);

fprintf("=== DNN2 MODEL EVALUATION (PF + LS) ===\n\n");

noiseLabels = ["001", "01", "1", "10", "100"];
pfDatasetPath = "checkpoints/dnn2_dataset.h5";
lsDatasetPath = "checkpoints/dnn2_dataset_ls.h5";
pfResultPath = "checkpoints/pf_results_cv.h5";
lsResultPath = "checkpoints/lls_dnn1_results.h5";

results = table('Size', [numel(noiseLabels), 9], ...
    'VariableTypes', ["string", "double", "double", "double", "double", "double", "double", "double", "double"], ...
    'VariableNames', ["Noise", "PF_DNN2_TestRMSE", "LS_DNN2_TestRMSE", "PF_RMSE", "LS_RMSE", ...
    "PF_DNN2_vs_PF", "PF_DNN2_vs_LS", "LS_DNN2_vs_PF", "LS_DNN2_vs_LS"]);

fprintf("%-8s %-12s %-12s %-11s %-11s %-12s %-12s\n", ...
    "Noise", "PF-DNN2", "LS-DNN2", "PF", "LS", "PF-DNN2vsPF", "LS-DNN2vsLS");
fprintf("%-8s %-12s %-12s %-11s %-11s %-12s %-12s\n", ...
    "-------", "--------", "--------", "--", "--", "-----------", "-----------");

for k = 1:numel(noiseLabels)
    noiseLabel = string(noiseLabels(k));

    % First-stage baselines
    pf_rmse = safeReadMean(pfResultPath, "/" + noiseLabel + "/rmseAll");
    ls_rmse = safeReadMean(lsResultPath, "/" + noiseLabel + "/rmseSample");

    % PF-DNN2 evaluation
    pf_dnn2_rmse = NaN;
    pfCkpt = fullfile("checkpoints", "dnn2_postprocess_" + noiseLabel + ".mat");
    if isfile(pfCkpt) && isfile(pfDatasetPath)
        ckptPF = load(pfCkpt, "net");
        dataPF = dnn2_load_dataset(pfDatasetPath, noiseLabel, 0.80, 0.10, 0.10, 42);
        yHatPF = predict(ckptPF.net, dataPF.XTest);
        pf_dnn2_rmse = rmse(yHatPF, dataPF.YTest);
    end

    % LS-DNN2 evaluation
    ls_dnn2_rmse = NaN;
    lsCkpt = fullfile("checkpoints", "dnn2_postprocess_ls_" + noiseLabel + ".mat");
    if isfile(lsCkpt) && isfile(lsDatasetPath)
        ckptLS = load(lsCkpt, "net");
        dataLS = dnn2_load_dataset(lsDatasetPath, noiseLabel, 0.80, 0.10, 0.10, 42);
        yHatLS = predict(ckptLS.net, dataLS.XTest);
        ls_dnn2_rmse = rmse(yHatLS, dataLS.YTest);
    end

    pf_dnn2_vs_pf = improvePct(pf_rmse, pf_dnn2_rmse);
    pf_dnn2_vs_ls = improvePct(ls_rmse, pf_dnn2_rmse);
    ls_dnn2_vs_pf = improvePct(pf_rmse, ls_dnn2_rmse);
    ls_dnn2_vs_ls = improvePct(ls_rmse, ls_dnn2_rmse);

    results.Noise(k) = noiseLabel;
    results.PF_DNN2_TestRMSE(k) = pf_dnn2_rmse;
    results.LS_DNN2_TestRMSE(k) = ls_dnn2_rmse;
    results.PF_RMSE(k) = pf_rmse;
    results.LS_RMSE(k) = ls_rmse;
    results.PF_DNN2_vs_PF(k) = pf_dnn2_vs_pf;
    results.PF_DNN2_vs_LS(k) = pf_dnn2_vs_ls;
    results.LS_DNN2_vs_PF(k) = ls_dnn2_vs_pf;
    results.LS_DNN2_vs_LS(k) = ls_dnn2_vs_ls;

    fprintf("%-8s %-12s %-12s %-11s %-11s %-12s %-12s\n", ...
        noiseLabel, fmtNum(pf_dnn2_rmse), fmtNum(ls_dnn2_rmse), fmtNum(pf_rmse), fmtNum(ls_rmse), ...
        fmtPct(pf_dnn2_vs_pf), fmtPct(ls_dnn2_vs_ls));
end

fprintf("\n=== SUMMARY TABLE ===\n");
disp(results);

fprintf("\n=== OVERALL (mean over noise labels) ===\n");
fprintf("PF-DNN2 RMSE: %.6f\n", mean(results.PF_DNN2_TestRMSE, "omitnan"));
fprintf("LS-DNN2 RMSE: %.6f\n", mean(results.LS_DNN2_TestRMSE, "omitnan"));
fprintf("PF baseline : %.6f\n", mean(results.PF_RMSE, "omitnan"));
fprintf("LS baseline : %.6f\n", mean(results.LS_RMSE, "omitnan"));

fprintf("\nImprovement (%%):\n");
fprintf("PF-DNN2 vs PF: %.2f%%\n", mean(results.PF_DNN2_vs_PF, "omitnan"));
fprintf("PF-DNN2 vs LS: %.2f%%\n", mean(results.PF_DNN2_vs_LS, "omitnan"));
fprintf("LS-DNN2 vs PF: %.2f%%\n", mean(results.LS_DNN2_vs_PF, "omitnan"));
fprintf("LS-DNN2 vs LS: %.2f%%\n", mean(results.LS_DNN2_vs_LS, "omitnan"));

fprintf("\nEvaluation complete.\n");


function v = rmse(yHat, y)
v = sqrt(mean((yHat - y).^2, "all"));
end

function v = safeReadMean(h5Path, datasetPath)
v = NaN;
if ~isfile(h5Path)
    return;
end
try
    x = h5read(h5Path, datasetPath);
    v = mean(x, "omitnan");
catch
    v = NaN;
end
end

function p = improvePct(baseRmse, modelRmse)
p = NaN;
if ~isnan(baseRmse) && ~isnan(modelRmse) && baseRmse > 0
    p = (baseRmse - modelRmse) / baseRmse * 100;
end
end

function s = fmtNum(x)
if isnan(x)
    s = "N/A";
else
    s = sprintf("%.6f", x);
end
end

function s = fmtPct(x)
if isnan(x)
    s = "N/A";
else
    s = sprintf("%+.2f%%", x);
end
end
