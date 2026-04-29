%% build_dnn2_dataset_ls.m
% Build a DNN2 training dataset from existing LS results.
% Inputs per sample:
%   - DNN1 denoised measurements: xPre [4, T]
%   - LS estimated state:         xLS  [2, T]
% Targets per sample:
%   - Ground-truth position:      gtPos [2, T]
%
% To keep the same DNN2 interface as DNN1-PF-DNN2, this script writes
% LS estimates to the dataset key "/xPF" as a drop-in replacement.

clear; clc;
rng(42);

%% Configuration
srcFile = "checkpoints/lls_dnn1_results.h5";
outFile = "checkpoints/dnn2_dataset_ls.h5";
noiseLabels = ["001", "01", "1", "10", "100"];

if ~isfile(srcFile)
    error("Source LS results file not found: %s", srcFile);
end

if isfile(outFile)
    delete(outFile);
end

fprintf("=== BUILD DNN2 LS DATASET ===\n");
fprintf("Source: %s\n", srcFile);
fprintf("Output: %s\n\n", outFile);

summary = table('Size', [numel(noiseLabels), 5], ...
    'VariableTypes', ["string", "double", "double", "double", "double"], ...
    'VariableNames', ["Noise", "NumSamples", "T", "MeanLS_RMSE", "MeanInputRMSE"]);

for k = 1:numel(noiseLabels)
    noiseLabel = string(noiseLabels(k));
    grp = "/" + noiseLabel;

    xPre = h5read(srcFile, grp + "/xPre");          % [4, T, N]
    xLS = h5read(srcFile, grp + "/xLLS");           % [2, T, N]
    gtPosRaw = h5read(srcFile, grp + "/gtPos");     % [2, T]
    rmseLS = h5read(srcFile, grp + "/rmseSample");  % [N, 1]

    [nPre, T, N] = size(xPre);
    [nState, Tstate, Nstate] = size(xLS);
    [nGt, Tgt] = size(gtPosRaw);

    if nPre ~= 4 || nState ~= 2 || nGt ~= 2 || T ~= Tstate || T ~= Tgt || N ~= Nstate
        error("Shape mismatch for noise %s.", noiseLabel);
    end

    % Repeat GT across samples so each sample is self-contained.
    gtPosStack = repmat(gtPosRaw, 1, 1, N);  % [2, T, N]

    % Keep DNN2 input layout identical: [4ch DNN1 output ; 2ch state estimate].
    xDNN2 = cat(1, xPre, xLS);               % [6, T, N]

    inputRmse = zeros(N, 1);
    for s = 1:N
        inputRmse(s) = sqrt(mean((xLS(:, :, s) - gtPosRaw).^2, "all"));
    end

    summary.Noise(k) = noiseLabel;
    summary.NumSamples(k) = N;
    summary.T(k) = T;
    summary.MeanLS_RMSE(k) = mean(rmseLS);
    summary.MeanInputRMSE(k) = mean(inputRmse);

    h5create(outFile, grp + "/xPre", size(xPre));
    h5write(outFile, grp + "/xPre", xPre);

    % Write to /xPF for compatibility with existing dnn2_load_dataset().
    h5create(outFile, grp + "/xPF", size(xLS));
    h5write(outFile, grp + "/xPF", xLS);

    h5create(outFile, grp + "/xDNN2", size(xDNN2));
    h5write(outFile, grp + "/xDNN2", xDNN2);

    h5create(outFile, grp + "/gtPos", size(gtPosStack));
    h5write(outFile, grp + "/gtPos", gtPosStack);

    h5create(outFile, grp + "/rmsePF", size(rmseLS));
    h5write(outFile, grp + "/rmsePF", rmseLS);

    fprintf("[%s] samples=%d | T=%d | LS RMSE=%.6f | input RMSE=%.6f\n", ...
        noiseLabel, N, T, summary.MeanLS_RMSE(k), summary.MeanInputRMSE(k));
end

writeSummaryMeta(outFile, summary);

fprintf("\n=== LS DATASET READY ===\n");
disp(summary);
fprintf("Saved: %s\n", outFile);


function writeSummaryMeta(outFile, summary)
metaFile = outFile;
if isfile(metaFile)
    save(strrep(metaFile, ".h5", ".mat"), "summary");
end
end
