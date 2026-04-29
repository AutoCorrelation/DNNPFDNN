%% build_dnn2_dataset.m
% Build a DNN2 training dataset from existing PF results.
% Inputs per sample:
%   - DNN1 denoised measurements: xPre [4, T]
%   - PF estimated state:         xPF  [2, T]
% Targets per sample:
%   - Ground-truth position:      gtPos [2, T]
%
% This script converts checkpoints/pf_results_cv.h5 into a dedicated H5 file
% for the DNN2 postprocessor stage.

clear; clc;
rng(42);

%% Configuration
srcFile = "checkpoints/pf_results_cv.h5";
outFile = "checkpoints/dnn2_dataset.h5";
noiseLabels = ["001", "01", "1", "10", "100"];

if ~isfile(srcFile)
    error("Source PF results file not found: %s", srcFile);
end

if isfile(outFile)
    delete(outFile);
end

fprintf("=== BUILD DNN2 DATASET ===\n");
fprintf("Source: %s\n", srcFile);
fprintf("Output: %s\n\n", outFile);

summary = table('Size', [numel(noiseLabels), 5], ...
    'VariableTypes', ["string", "double", "double", "double", "double"], ...
    'VariableNames', ["Noise", "NumSamples", "T", "MeanPF_RMSE", "MeanInputRMSE"]);

for k = 1:numel(noiseLabels)
    noiseLabel = string(noiseLabels(k));
    grp = "/" + noiseLabel;

    xPre = h5read(srcFile, grp + "/xPre");       % [4, T, N]
    xPF = h5read(srcFile, grp + "/xEst");         % [2, T, N]
    gtPos = h5read(srcFile, grp + "/gtPos");      % [2, T]
    rmsePF = h5read(srcFile, grp + "/rmseAll");   % [N, 1]

    [nPre, T, N] = size(xPre);
    [nState, Tstate, Nstate] = size(xPF);
    [nGt, Tgt] = size(gtPos);

    if nPre ~= 4 || nState ~= 2 || nGt ~= 2 || T ~= Tstate || T ~= Tgt || N ~= Nstate
        error("Shape mismatch for noise %s.", noiseLabel);
    end

    % Repeat GT across samples so each sample is self-contained.
    gtPosStack = repmat(gtPos, 1, 1, N);  % [2, T, N]

    % Optional derived input: concatenate DNN1 output and PF estimate.
    % DNN2 can learn to refine PF state using the denoised measurement skip connection.
    xDNN2 = cat(1, xPre, xPF);             % [6, T, N]

    % Simple quality statistics for later analysis.
    inputRmse = zeros(N, 1);
    for s = 1:N
        inputRmse(s) = sqrt(mean((xPF(:, :, s) - gtPos).^2, "all"));
    end

    summary.Noise(k) = noiseLabel;
    summary.NumSamples(k) = N;
    summary.T(k) = T;
    summary.MeanPF_RMSE(k) = mean(rmsePF);
    summary.MeanInputRMSE(k) = mean(inputRmse);

    % Write datasets for this variance.
    h5create(outFile, grp + "/xPre", size(xPre));
    h5write(outFile, grp + "/xPre", xPre);

    h5create(outFile, grp + "/xPF", size(xPF));
    h5write(outFile, grp + "/xPF", xPF);

    h5create(outFile, grp + "/xDNN2", size(xDNN2));
    h5write(outFile, grp + "/xDNN2", xDNN2);

    h5create(outFile, grp + "/gtPos", size(gtPosStack));
    h5write(outFile, grp + "/gtPos", gtPosStack);

    h5create(outFile, grp + "/rmsePF", size(rmsePF));
    h5write(outFile, grp + "/rmsePF", rmsePF);

    fprintf("[%s] samples=%d | T=%d | PF RMSE=%.6f | input RMSE=%.6f\n", ...
        noiseLabel, N, T, summary.MeanPF_RMSE(k), summary.MeanInputRMSE(k));
end

% Write dataset-level metadata.
writeSummaryMeta(outFile, summary);

fprintf("\n=== DATASET READY ===\n");
disp(summary);
fprintf("Saved: %s\n", outFile);


function writeSummaryMeta(outFile, summary)
metaFile = outFile;
if isfile(metaFile)
    % Store summary in a compact MAT-sidecar for easy MATLAB loading.
    save(strrep(metaFile, ".h5", ".mat"), "summary");
end
end