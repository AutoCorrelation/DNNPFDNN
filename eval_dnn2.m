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
        ckptPF = load(pfCkpt, "net", "normalization");
        dataPF = loadDnn2Dataset(pfDatasetPath, noiseLabel, 0.80, 0.10, 0.10, 42);
        [xTestPF, muYPF, sigmaYPF] = normalizeForInference(dataPF.XTest, ckptPF);
        yHatPFZ = minibatchpredict(ckptPF.net, xTestPF);
        yHatPF = zscoreInverse(yHatPFZ, muYPF, sigmaYPF);
        pf_dnn2_rmse = rmse(yHatPF, dataPF.YTest);
    end

    % LS-DNN2 evaluation
    ls_dnn2_rmse = NaN;
    lsCkpt = fullfile("checkpoints", "dnn2_postprocess_ls_" + noiseLabel + ".mat");
    if isfile(lsCkpt) && isfile(lsDatasetPath)
        ckptLS = load(lsCkpt, "net", "normalization");
        dataLS = loadDnn2Dataset(lsDatasetPath, noiseLabel, 0.80, 0.10, 0.10, 42);
        [xTestLS, muYLS, sigmaYLS] = normalizeForInference(dataLS.XTest, ckptLS);
        yHatLSZ = minibatchpredict(ckptLS.net, xTestLS);
        yHatLS = zscoreInverse(yHatLSZ, muYLS, sigmaYLS);
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

function [XNorm, muY, sigmaY] = normalizeForInference(X, ckpt)
% Apply checkpoint normalization consistently with training script.
XNorm = X;
muY = [0 0];
sigmaY = [1 1];

if ~isfield(ckpt, "normalization")
    return;
end

normInfo = ckpt.normalization;
if isfield(normInfo, "muX") && isfield(normInfo, "sigmaX") && ...
        ~isempty(normInfo.muX) && ~isempty(normInfo.sigmaX)
    XNorm = zscoreApply(X, normInfo.muX, normInfo.sigmaX);
end

if isfield(normInfo, "muY") && isfield(normInfo, "sigmaY") && ...
        ~isempty(normInfo.muY) && ~isempty(normInfo.sigmaY)
    muY = normInfo.muY;
    sigmaY = normInfo.sigmaY;
end
end

function data = loadDnn2Dataset(h5Path, noiseLabel, trainRatio, valRatio, testRatio, seed)
% Dataset layout:
% /<noiseLabel>/xPre [4, T, N]
% /<noiseLabel>/xPF  [2, T, N]
% /<noiseLabel>/gtPos [2, T, N]

groupPath = "/" + noiseLabel;

xPre = h5read(h5Path, groupPath + "/xPre");
xPF = h5read(h5Path, groupPath + "/xPF");
gtPos = h5read(h5Path, groupPath + "/gtPos");

[nPre, numSteps, numSamples] = size(xPre);
[nPF, numStepsPF, numSamplesPF] = size(xPF);
[nGT, numStepsGT, numSamplesGT] = size(gtPos);

if nPre ~= 4 || nPF ~= 2 || nGT ~= 2 || ...
        numSteps ~= numStepsPF || numSteps ~= numStepsGT || ...
        numSamples ~= numSamplesPF || numSamples ~= numSamplesGT
    error("Unexpected DNN2 dataset shape for noise %s.", noiseLabel);
end

rng(seed);
perm = randperm(numSamples);
nTrain = floor(trainRatio * numSamples);
nVal = floor(valRatio * numSamples);

idxTrain = perm(1:nTrain);
idxVal = perm(nTrain + 1:nTrain + nVal);
idxTest = perm(nTrain + nVal + 1:end);

data = struct();
data.XTrain = packSamples(xPre, xPF, idxTrain);
data.YTrain = packTargets(gtPos, idxTrain);
data.XVal = packSamples(xPre, xPF, idxVal);
data.YVal = packTargets(gtPos, idxVal);
data.XTest = packSamples(xPre, xPF, idxTest);
data.YTest = packTargets(gtPos, idxTest);
end

function X = packSamples(xPre, xPF, indices)
xPreSel = xPre(:, :, indices);
xPFSel = xPF(:, :, indices);
X = [transposeToRows(xPFSel), transposeToRows(xPreSel)];
end

function Y = packTargets(gtPos, indices)
Y = transposeToRows(gtPos(:, :, indices));
end

function X = transposeToRows(A)
% [C, T, N] -> [N*T, C]
X = permute(A, [3 2 1]);
X = reshape(X, [], size(A, 1));
end

function Z = zscoreApply(A, mu, sigma)
Z = (A - mu) ./ sigma;
end

function A = zscoreInverse(Z, mu, sigma)
A = Z .* sigma + mu;
end
