%% filter_pf_cv.m
% Particle filter with constant-velocity-like transition.
% State: [pos_x, pos_y] (2D position)
% Transition: x_t = x_{t-1} + v_{t-1} + q_t where v = x_{t-1} - x_{t-2}
% Measurement: y_t = H(x_t) nonlinear (distance to anchors)
% Evaluation: RMSE at each timestep, saved to HDF5

clear; clc;
rng(42);

%% Configuration
h5Path = "ranging_data_cv.h5";
anchors = [0 10; 0 0; 10 0; 10 10]';  % [2, 4]
noiseDatasets = ["/ranging_001", "/ranging_01", "/ranging_1", "/ranging_10", "/ranging_100"];
noiseLabels = ["001", "01", "1", "10", "100"];

numParticles = 1000;
numMC = 1;  % Can increase for ensemble averaging

% Load GT data
gtPos = h5read(h5Path, "/gt_position");        % [2, step]
gtRanging = h5read(h5Path, "/gt_ranging");     % [4, step]
[nPos, nStep] = size(gtPos);

% Probe the first variance to get sample count
xRaw = h5read(h5Path, noiseDatasets(1));  % [4, step, nSample]
[nAnchors, ~, nSample] = size(xRaw);

fprintf("=== PF Configuration ===\n");
fprintf("Particles: %d | MC: %d | Step: %d | Sample: %d\n", ...
    numParticles, numMC, nStep, nSample);
fprintf("Anchors:\n");
disp(anchors);

%% === Main: Variance loop with parfor ===
results = cell(numel(noiseDatasets), 1);
% Use regular for loop for simpler execution (parfor can have issues with struct array assignments)
for k = 1:numel(noiseDatasets)
    noiseDataset = string(noiseDatasets(k));
    noiseLabel = string(noiseLabels(k));

    cfg = getNoiseConfig(noiseLabel);
    if ~isfile(cfg.dnn1File)
        error("DNN1 checkpoint not found: %s", cfg.dnn1File);
    end
    dnnData = load(cfg.dnn1File, "net", "normalization");
    net = dnnData.net;
    normalization = dnnData.normalization;
    Q = cfg.Q;
    R = cfg.R;
    
    % Load noisy data for this variance
    xRaw_k = h5read(h5Path, noiseDataset);  % [4, step, nSample]
    
    % Containers for results
    xEst_all = cell(nSample, numMC);
    rmseAll = zeros(nSample, numMC);
    xPreStack = zeros(nAnchors, nStep, nSample);
    xEstStack = zeros(nPos, nStep, nSample);
    
    % Sample loop (sequential within worker)
    for s = 1:nSample
        xNoisySample = xRaw_k(:, :, s);  % [4, step]
        
        % Apply DNN1 preprocessing
        xPre = preprocess_dnn1_predict(net, normalization, xNoisySample);  % [4, step]
        
        % MC loop
        for mc = 1:numMC
            % Run PF
            [xEst, rmseVec] = runParticleFilter_CV( ...
                xPre, gtPos, anchors, numParticles, Q, R);
            
            % Store results
            xEst_all{s, mc} = xEst;  % [2, step] estimated positions
            rmseAll(s, mc) = mean(rmseVec);  % scalar, mean RMSE across steps

            if mc == 1
                xPreStack(:, :, s) = xPre;
                xEstStack(:, :, s) = xEst;
            end
        end
    end
    
    % Save results for this variance to HDF5
    results{k} = struct( ...
        "noiseLabel", noiseLabel, ...
        "xEst_all", {xEst_all}, ...
        "xEstStack", xEstStack, ...
        "rmseAll", rmseAll, ...
        "xPreStack", xPreStack, ...
        "gtPos", gtPos, ...
        "gtRanging", gtRanging, ...
        "dnn1File", cfg.dnn1File, ...
        "Q", Q, ...
        "R", R);
    
    fprintf("Completed variance %s: mean RMSE = %.6f\n", ...
        noiseLabel, mean(rmseAll, "all"));
end

%% === Save results to HDF5 ===
outFile = "checkpoints/pf_results_cv.h5";
if isfile(outFile)
    delete(outFile);
end

% Save per-variance results
for k = 1:numel(noiseDatasets)
    res = results{k};
    grpPath = "/" + string(noiseLabels(k));
    
    % Save RMSE summary
    h5create(outFile, grpPath + "/rmseAll", size(res.rmseAll));
    h5write(outFile, grpPath + "/rmseAll", res.rmseAll);

    % Save PF estimate and DNN-denoised measurements
    h5create(outFile, grpPath + "/xEst", size(res.xEstStack));
    h5write(outFile, grpPath + "/xEst", res.xEstStack);
    h5create(outFile, grpPath + "/xPre", size(res.xPreStack));
    h5write(outFile, grpPath + "/xPre", res.xPreStack);
    
    % Save GT
    h5create(outFile, grpPath + "/gtPos", size(res.gtPos));
    h5write(outFile, grpPath + "/gtPos", res.gtPos);
    
    h5create(outFile, grpPath + "/gtRanging", size(res.gtRanging));
    h5write(outFile, grpPath + "/gtRanging", res.gtRanging);
end

fprintf("Saved: %s\n", outFile);

%% === Summary table ===
summaryTable = table('Size', [numel(noiseLabels), 2], ...
    'VariableTypes', ["string", "double"], ...
    'VariableNames', ["Noise", "MeanRMSE"]);

for k = 1:numel(noiseLabels)
    summaryTable.Noise(k) = string(noiseLabels(k));
    summaryTable.MeanRMSE(k) = mean(results{k}.rmseAll, "all");
end

fprintf("\n=== Summary Table ===\n");
disp(summaryTable);

%% ======== Local Functions ========

function [xEst, rmseVec] = runParticleFilter_CV(zMeas, gtPos, anchors, Np, Q, R)
% Particle filter with CV-like dynamics.
% zMeas: [4, T] preprocessed measurements (ranging)
% gtPos: [2, T] ground truth positions
% anchors: [2, 4] anchor positions
% Np: number of particles
% Q: process noise covariance [2, 2]
% R: measurement noise covariance [4, 4]
%
% Output:
% xEst: [2, T] estimated positions
% rmseVec: [T, 1] RMSE at each timestep

[nAnchors, T] = size(zMeas);
if nAnchors ~= 4
    error("Expected 4 anchors, got %d", nAnchors);
end

xEst = zeros(2, T);
rmseVec = zeros(T, 1);
w = ones(1, Np) / Np;

% First two steps are initialized by LS, then PF starts from t = 3.
xLS = zeros(2, min(2, T));
for t = 1:min(2, T)
    xLS(:, t) = lsPositionFromRanging(zMeas(:, t), anchors);
    xEst(:, t) = xLS(:, t);
    rmseVec(t) = sqrt(mean((xEst(:, t) - gtPos(:, t)).^2));
end

if T <= 2
    return;
end

% Start PF from the LS estimate at t=2, with velocity from LS(t=2)-LS(t=1)
xPrev2 = repmat(xLS(:, 1), 1, Np);
xPrev1 = repmat(xLS(:, 2), 1, Np);

% Precompute Cholesky factors for efficiency
Q_chol = chol(Q, "lower");
R_inv = inv(R);

for t = 3:T
    % === Prediction: x_t = x_{t-1} + v_{t-1} + q_t ===
    % Velocity for each particle: v = x_{t-1} - x_{t-2}
    v = xPrev1 - xPrev2;  % [2, Np] velocity of each particle
    
    % Transition with process noise
    xPred = xPrev1 + v + Q_chol * randn(2, Np);  % [2, Np]
    
    % === Update: likelihood from measurement ===
    z_t = zMeas(:, t);  % [4, 1]
    
    % Measurement model: predict range to anchors from particle positions
    dx = xPred(1, :) - anchors(1, :)';  % [4, Np]
    dy = xPred(2, :) - anchors(2, :)';  % [4, Np]
    zPred = sqrt(dx.^2 + dy.^2);        % [4, Np]
    
    % Likelihood: multivariate Gaussian
    % logLik = -0.5 * (z - zPred)'*R_inv*(z - zPred)
    diffMat = z_t - zPred;  % [4, Np]
    logLik = -0.5 * sum((R_inv * diffMat) .* diffMat, 1);
    
    % Normalize weights
    maxLogLik = max(logLik);
    lik = exp(logLik - maxLogLik);  % Numerically stable
    w = w .* lik;
    wSum = sum(w);
    if ~isfinite(wSum) || wSum <= 0
        w = ones(1, Np) / Np;
    else
        w = w / wSum;  % [1, Np]
    end

    % Estimate state BEFORE resampling to avoid sampling noise in estimate.
    xEst(:, t) = xPred * w';  % [2, 1]
    
    % === Resampling: ESS-based ===
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
    
    % === RMSE at this timestep ===
    rmseVec(t) = sqrt(mean((xEst(:, t) - gtPos(:, t)).^2));
    
    % Store for next iteration
end

end

function x_ls = lsPositionFromRanging(ranges, anchors)
% Least-squares position estimate from range measurements.
% ranges: [4, 1]
% anchors: [2, 4]

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

x_ls = A \ b;
end

function idx = resample_systematic(weights)
% Systematic resampling.
% weights: [1, Np]

Np = length(weights);
cumW = cumsum(weights);

% Generate uniform samples
u = (0:Np-1)' / Np + rand() / Np;  % [Np, 1]

% Find indices
idx = zeros(Np, 1);
j = 1;
for i = 1:Np
    while cumW(j) < u(i) && j < Np
        j = j + 1;
    end
    idx(i) = j;
end

end

function cfg = getNoiseConfig(noiseLabel)
% Map noise label to DNN checkpoint and PF covariance settings.
% Prefer the latest saved tuning summary if it exists.

noiseLabel = string(noiseLabel);
switch noiseLabel
    case "001"
        noiseVar = 0.01;
    case "01"
        noiseVar = 0.1;
    case "1"
        noiseVar = 1;
    case "10"
        noiseVar = 10;
    case "100"
        noiseVar = 100;
    otherwise
        error("Unsupported noise label: %s", noiseLabel);
end

bestQDiag = [];
tuningFile = "checkpoints/pf_qr_tuning_results.mat";
if isfile(tuningFile)
    tuningData = load(tuningFile, "summary");
    if isfield(tuningData, "summary") && ~isempty(tuningData.summary)
        idx = find(tuningData.summary.Noise == noiseLabel, 1, "first");
        if ~isempty(idx)
            bestQDiag = tuningData.summary.BestQAlpha(idx);
        end
    end
end

if isempty(bestQDiag)
    switch noiseLabel
        case "001"
            bestQDiag = 0.01;
        case "01"
            bestQDiag = 0.01;
        case "1"
            bestQDiag = 0.1;
        case "10"
            bestQDiag = 1;
        case "100"
            bestQDiag = 10;
    end
end

cfg.dnn1File = "checkpoints/dnn1_residual_single_" + noiseLabel + ".mat";
cfg.Q = eye(2) * bestQDiag;  % Use tuned Q values
cfg.R = eye(4) * noiseVar;   % R auto-estimated from DNN residuals
end

function [x_dnn, state] = preprocess_dnn1_predict(net, normalization, xRaw)
% Apply DNN1 preprocessing.
% xRaw: [4, T] or [T, 4] measurements
% x_dnn: [4, T] denoised measurements

if size(xRaw, 1) == 4 && size(xRaw, 2) ~= 4
    xRaw_T = xRaw';  % [T, 4]
    isTransposed = true;
else
    xRaw_T = xRaw;  % [T, 4] assumed
    isTransposed = false;
end

% Normalize
xNorm = (xRaw_T - normalization.muX) ./ normalization.sigmaX;
yNorm = predict(net, xNorm);  % [T, 4]
x_dnn_T = yNorm .* normalization.sigmaY + normalization.muY;  % [T, 4]

if isTransposed
    x_dnn = x_dnn_T';  % Back to [4, T]
else
    x_dnn = x_dnn_T;  % Keep [T, 4] or reshape?
end

% Ensure output is [4, T]
if size(x_dnn, 1) ~= 4
    x_dnn = x_dnn';
end

state = struct("inputShape", size(xRaw), "outputShape", size(x_dnn));

end
