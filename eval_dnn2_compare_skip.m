%% eval_dnn2_compare_skip.m
% Evaluate residual vs no-skip DNN2 variants for PF branch only.

clear; clc; rng(42);

noiseLabels = ["001", "01", "1", "10", "100"];
variants = ["res", "noskip"];

datasetPath = "checkpoints/dnn2_dataset.h5";
ckptPrefix = "dnn2_postprocess_";

rows = numel(noiseLabels) * numel(variants);
resultsPF = initResultTable(rows);

resultsPF = evalBranch("pf", noiseLabels, variants, datasetPath, ckptPrefix, resultsPF);

fprintf("\n=== PF Branch (Residual vs NoSkip) ===\n");
disp(resultsPF);
printSummary(resultsPF, "PF");


function T = initResultTable(n)
T = table('Size', [n, 4], ...
    'VariableTypes', ["string", "string", "double", "string"], ...
    'VariableNames', ["Noise", "Variant", "TestRMSE", "Checkpoint"]);
end

function T = evalBranch(branchName, noiseLabels, variants, h5Path, prefix, T)
row = 1;
for v = 1:numel(variants)
    variant = variants(v);
    for k = 1:numel(noiseLabels)
        noiseLabel = string(noiseLabels(k));
        ckpt = fullfile("checkpoints", prefix + variant + "_" + noiseLabel + ".mat");
        rmseVal = NaN;

        if isfile(ckpt) && isfile(h5Path)
            ck = load(ckpt, "net");
            data = dnn2_load_dataset(h5Path, noiseLabel, 0.80, 0.10, 0.10, 42);
            yHat = predict(ck.net, data.XTest);
            rmseVal = sqrt(mean((yHat - data.YTest).^2, "all"));
        end

        T.Noise(row) = noiseLabel;
        T.Variant(row) = variant;
        T.TestRMSE(row) = rmseVal;
        T.Checkpoint(row) = string(ckpt);
        row = row + 1;
    end
end

fprintf("\n[%s] evaluation done.\n", upper(branchName));
end

function printSummary(T, branchName)
resMask = T.Variant == "res";
nosMask = T.Variant == "noskip";
resMean = mean(T.TestRMSE(resMask), "omitnan");
nosMean = mean(T.TestRMSE(nosMask), "omitnan");

fprintf("[%s] mean RMSE (res):    %.6f\n", branchName, resMean);
fprintf("[%s] mean RMSE (noskip): %.6f\n", branchName, nosMean);

if ~isnan(resMean) && ~isnan(nosMean) && nosMean > 0
    gain = (nosMean - resMean) / nosMean * 100;
    fprintf("[%s] residual gain vs noskip: %+0.2f%%\n", branchName, gain);
end
end
