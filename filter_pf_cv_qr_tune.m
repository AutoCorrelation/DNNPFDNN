%% filter_pf_cv_qr_tune.m
% Keep existing filter script unchanged.
% This script tunes Q scale with auto-estimated R per noise variance.

clear; clc;
rng(42);

%% Configuration
h5Path = "ranging_data_cv.h5";
anchors = [0 10; 0 0; 10 0; 10 10]';  % [2, 4]
noiseDatasets = ["/ranging_001", "/ranging_01", "/ranging_1", "/ranging_10", "/ranging_100"];
noiseLabels = ["001", "01", "1", "10", "100"];
noiseVariances = [0.01, 0.1, 1, 10, 100];  % Variance magnitudes corresponding to each label

numParticles = 500;
numMC = 1;

% Run mode:
%   "quick" - fast validation run for day-to-day use
%   "full"  - larger sample set for more stable tuning
runMode = "quick";

switch runMode
    case "quick"
        tuneSamplesForR = 300;
        evalSamplesForQ = 300;
    case "full"
        tuneSamplesForR = inf;
        evalSamplesForQ = inf;
    otherwise
        error("Unsupported run mode: %s", runMode);
end

% Base Q multipliers; will be scaled by noise variance for each condition
qBaseMultipliers = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 3e0, 10, 30];

qMinDiag = 1e-8;
rMinDiag = 1e-8;

if ~isfile(h5Path)
    error("Data file not found: %s", h5Path);
end

gtPos = h5read(h5Path, "/gt_position");        % [2, step]
gtRanging = h5read(h5Path, "/gt_ranging");     % [4, step]
[~, nStep] = size(gtPos);

summary = table('Size', [numel(noiseLabels), 4], ...
    'VariableTypes', ["string", "double", "double", "double"], ...
    'VariableNames', ["Noise", "BestQAlpha", "BestMeanRMSE", "UsedSamples"]);

allTuneResults = struct();

% Regular for loop (parfor has issues with struct array assignments)
for k = 1:numel(noiseLabels)
    noiseLabel = string(noiseLabels(k));
    noiseDataset = string(noiseDatasets(k));
    noiseVar = noiseVariances(k);  % Get variance for this noise condition
    % dnn1File = "checkpoints/dnn1_residual_single_" + noiseLabel + ".mat";
    dnn1File = "checkpoints/dnn1_residual_trainnet_" + noiseLabel + ".mat";
    

    if ~isfile(dnn1File)
        error("DNN1 checkpoint not found: %s", dnn1File);
    end

    % Compute variance-scaled Q grid for this noise condition
    qAlphaGrid = qBaseMultipliers * noiseVar;

    dnnData = load(dnn1File, "net", "normalization");
    net = dnnData.net;
    normalization = dnnData.normalization;

    xRaw = h5read(h5Path, noiseDataset);  % [4, step, sample]
    [~, nStepRaw, nSample] = size(xRaw);
    if nStepRaw ~= nStep
        error("Step mismatch for %s", noiseDataset);
    end

    nR = min(nSample, tuneSamplesForR);
    nEval = min(nSample, evalSamplesForQ);

    % 1) Auto-estimate R from DNN residuals against GT ranging.
    resBuf = zeros(4, nStep * nR);
    idx = 1;
    for s = 1:nR
        xPre = preprocess_dnn1_predict(net, normalization, xRaw(:, :, s));  % [4, step]
        res = xPre - gtRanging;  % [4, step]
        resBuf(:, idx:idx + nStep - 1) = res;
        idx = idx + nStep;
    end

    R_est = cov(resBuf');
    if any(~isfinite(R_est), "all")
        error("R estimation failed for noise %s", noiseLabel);
    end
    R_est = 0.5 * (R_est + R_est');
    R_est = R_est + rMinDiag * eye(4);

    % 2) Q-grid search with fixed R_est (using variance-scaled grid)
    gridRMSE = zeros(numel(qAlphaGrid), 1);

    for qIdx = 1:numel(qAlphaGrid)
        qAlpha = qAlphaGrid(qIdx);
        Q = eye(2) * max(qAlpha, qMinDiag);

        rmseMC = zeros(nEval, numMC);
        for s = 1:nEval
            xPre = preprocess_dnn1_predict(net, normalization, xRaw(:, :, s));

            for mc = 1:numMC
                [~, rmseVec] = runParticleFilter_CV(xPre, gtPos, anchors, numParticles, Q, R_est);
                rmseMC(s, mc) = mean(rmseVec);
            end
        end

        gridRMSE(qIdx) = mean(rmseMC, "all");
    end

    [bestRMSE, bestIdx] = min(gridRMSE);
    bestQAlpha = qAlphaGrid(bestIdx);

    summary.Noise(k) = noiseLabel;
    summary.BestQAlpha(k) = bestQAlpha;
    summary.BestMeanRMSE(k) = bestRMSE;
    summary.UsedSamples(k) = nEval;

    allTuneResults(k).noiseLabel = noiseLabel; 
    allTuneResults(k).R_est = R_est; 
    allTuneResults(k).qAlphaGrid = qAlphaGrid; 
    allTuneResults(k).gridRMSE = gridRMSE; 
    allTuneResults(k).bestQAlpha = bestQAlpha; 
    allTuneResults(k).bestRMSE = bestRMSE; 
    allTuneResults(k).nEval = nEval; 

    fprintf("[Noise %s] best Q alpha = %.6g | mean RMSE = %.6f\n", noiseLabel, bestQAlpha, bestRMSE);
end

fprintf("\n=== Q Tuning Summary (Variance-Scaled Grid, mode=%s, %d samples) ===\n", runMode, evalSamplesForQ);
disp(summary);

if ~isfolder("checkpoints")
    mkdir("checkpoints");
end
save("checkpoints/pf_qr_tuning_results.mat", "summary", "allTuneResults");
fprintf("Saved: checkpoints/pf_qr_tuning_results.mat\n");


function [xEst, rmseVec] = runParticleFilter_CV(zMeas, gtPos, anchors, Np, Q, R)
[nAnchors, T] = size(zMeas);
if nAnchors ~= 4
    error("Expected 4 anchors, got %d", nAnchors);
end

xEst = zeros(2, T);
rmseVec = zeros(T, 1);
w = ones(1, Np) / Np;

xLS = zeros(2, min(2, T));
for t = 1:min(2, T)
    xLS(:, t) = lsPositionFromRanging(zMeas(:, t), anchors);
    xEst(:, t) = xLS(:, t);
    rmseVec(t) = sqrt(mean((xEst(:, t) - gtPos(:, t)).^2));
end

if T <= 2
    return;
end

xPrev2 = repmat(xLS(:, 1), 1, Np);
xPrev1 = repmat(xLS(:, 2), 1, Np);

Q_chol = chol(Q, "lower");
R_inv = inv(R);

for t = 3:T
    v = xPrev1 - xPrev2;
    xPred = xPrev1 + v + Q_chol * randn(2, Np);

    z_t = zMeas(:, t);
    dx = xPred(1, :) - anchors(1, :)';
    dy = xPred(2, :) - anchors(2, :)';
    zPred = sqrt(dx.^2 + dy.^2);

    diffMat = z_t - zPred;
    logLik = -0.5 * sum((R_inv * diffMat) .* diffMat, 1);

    maxLogLik = max(logLik);
    lik = exp(logLik - maxLogLik);
    w = w .* lik;
    wSum = sum(w);
    if ~isfinite(wSum) || wSum <= 0
        w = ones(1, Np) / Np;
    else
        w = w / wSum;
    end

    xEst(:, t) = xPred * w';

    ESS = 1 / sum(w.^2);
    if ESS < Np / 2
        idx = resample_systematic(w);
        particles = xPred(:, idx);
        xPrev2 = xPrev1(:, idx);
        xPrev1 = particles;
        w = ones(1, Np) / Np;
    else
        particles = xPred;
        xPrev2 = xPrev1;
        xPrev1 = particles;
    end

    rmseVec(t) = sqrt(mean((xEst(:, t) - gtPos(:, t)).^2));
end
end

function xLS = lsPositionFromRanging(ranges, anchors)
refAnchor = anchors(:, 1);
refRange = ranges(1);

A = zeros(3, 2);
b = zeros(3, 1);

for i = 2:4
    ai = anchors(:, i);
    ri = ranges(i);
    A(i - 1, :) = 2 * (ai - refAnchor)';
    b(i - 1) = refRange^2 - ri^2 - sum(refAnchor.^2) + sum(ai.^2);
end

xLS = A \ b;
end

function idx = resample_systematic(w)
Np = length(w);
cumW = cumsum(w);
u = (0:Np-1)' / Np + rand() / Np;

idx = zeros(Np, 1);
j = 1;
for i = 1:Np
    while cumW(j) < u(i) && j < Np
        j = j + 1;
    end
    idx(i) = j;
end
end
