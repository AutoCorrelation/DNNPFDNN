%% eval_dpf_performance.m
% Evaluate saved DPF checkpoints and visualize DPF-only performance.

clear; clc; rng(42);

%% Configuration
h5Path = "ranging_data_cv.h5";
noiseDatasets = ["/ranging_001", "/ranging_01", "/ranging_1", "/ranging_10", "/ranging_100"];
checkpointDir = "checkpoints";
checkpointPrefix = "dpf_supervised_";

trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;
seed = 42;
miniBatchSize = 16;
executionEnvironment = "gpu"; % "gpu" | "cpu" | "auto"

if ~isfile(h5Path)
    error("File not found: %s", h5Path);
end

results = table('Size', [numel(noiseDatasets), 5], ...
    'VariableTypes', ["string", "double", "double", "double", "string"], ...
    'VariableNames', ["noise", "testRMSE", "testLoss", "numTestSeq", "checkpoint"]);

stepRmseAll = cell(numel(noiseDatasets), 1);

useGPU = shouldUseGPU(executionEnvironment);
fprintf("=== DPF Evaluation (%s) ===\n", upper(string(executionEnvironment)));

for k = 1:numel(noiseDatasets)
    noiseDataset = string(noiseDatasets(k));
    noiseTag = erase(noiseDataset, "/ranging_");
    ckptPath = fullfile(checkpointDir, checkpointPrefix + noiseTag + ".mat");

    if ~isfile(ckptPath)
        warning("Checkpoint not found for %s: %s", noiseDataset, ckptPath);
        results.noise(k) = noiseDataset;
        results.testRMSE(k) = NaN;
        results.testLoss(k) = NaN;
        results.numTestSeq(k) = NaN;
        results.checkpoint(k) = string(ckptPath);
        continue;
    end

    S = load(ckptPath, "transitionNet", "measureNet", "normalization", "config");

    data = loadSequenceDatasetFromRanging(h5Path, noiseDataset, trainRatio, valRatio, testRatio, seed);

    ZTest = zscoreApply3D(data.ZTest, S.normalization.muZ, S.normalization.sigmaZ);
    XTestGT = zscoreApply3D(data.XTestGT, S.normalization.muX, S.normalization.sigmaX);

    numParticles = S.config.numParticles;
    processStd = S.config.processStd;
    softResampleLambda = S.config.softResampleLambda;
    essThresholdRatio = S.config.essThresholdRatio;

    testLoss = evaluateLoss(S.transitionNet, S.measureNet, ZTest, XTestGT, ...
        numParticles, processStd, softResampleLambda, essThresholdRatio, miniBatchSize, useGPU);

    [rmseMean, rmsePerStep] = evaluateRMSE(S.transitionNet, S.measureNet, ZTest, XTestGT, S.normalization, ...
        numParticles, processStd, softResampleLambda, essThresholdRatio, miniBatchSize, useGPU);

    results.noise(k) = noiseDataset;
    results.testRMSE(k) = rmseMean;
    results.testLoss(k) = testLoss;
    results.numTestSeq(k) = size(ZTest, 3);
    results.checkpoint(k) = string(ckptPath);
    stepRmseAll{k} = rmsePerStep;

    fprintf("%s | testRMSE=%.6f | testLoss=%.6f | Ntest=%d\n", ...
        noiseDataset, rmseMean, testLoss, size(ZTest, 3));
end

fprintf("\n=== DPF Performance Table ===\n");
disp(results);

% Plot 1: Noise-wise RMSE bar chart
figure('Name','DPF Test RMSE by Noise','Color','w');
bar(categorical(results.noise), results.testRMSE);
ylabel('Test RMSE (m)');
xlabel('Noise Dataset');
title('DPF Test RMSE by Noise');
grid on;

% Plot 2: Step-wise RMSE curves
figure('Name','DPF Step-wise RMSE','Color','w');
hold on;
for k = 1:numel(noiseDatasets)
    if isempty(stepRmseAll{k})
        continue;
    end
    plot(stepRmseAll{k}, 'LineWidth', 1.5, 'DisplayName', char(noiseDatasets(k)));
end
xlabel('Time Step');
ylabel('RMSE (m)');
title('DPF Step-wise RMSE (Test Set)');
grid on;
legend('Location','best');
hold off;

%% Local functions
function loss = evaluateLoss(transitionNet, measureNet, Z, XGT, numParticles, processStd, softResampleLambda, essThresholdRatio, miniBatchSize, useGPU)
numSeq = size(Z, 3);
numIters = max(1, ceil(numSeq / miniBatchSize));
lossAcc = 0;
for it = 1:numIters
    idxStart = (it - 1) * miniBatchSize + 1;
    idxEnd = min(it * miniBatchSize, numSeq);
    idx = idxStart:idxEnd;

    l = dlfeval(@dpfLoss, transitionNet, measureNet, Z(:, :, idx), XGT(:, :, idx), ...
        numParticles, processStd, softResampleLambda, essThresholdRatio, useGPU);
    lossAcc = lossAcc + double(gather(extractdata(l)));
end
loss = lossAcc / numIters;
end

function [rmseMean, rmsePerStep] = evaluateRMSE(transitionNet, measureNet, Z, XGT, normalization, numParticles, processStd, softResampleLambda, essThresholdRatio, miniBatchSize, useGPU)
numSeq = size(Z, 3);
T = size(Z, 2);
seStep = zeros(T, 1);
numCount = 0;

numIters = max(1, ceil(numSeq / miniBatchSize));
for it = 1:numIters
    idxStart = (it - 1) * miniBatchSize + 1;
    idxEnd = min(it * miniBatchSize, numSeq);
    idx = idxStart:idxEnd;

    [xHatZ, xGtZ] = inferBatch(transitionNet, measureNet, Z(:, :, idx), XGT(:, :, idx), ...
        numParticles, processStd, softResampleLambda, essThresholdRatio, useGPU);

    xHat = zscoreInverse3D(xHatZ, normalization.muX, normalization.sigmaX);
    xGt = zscoreInverse3D(xGtZ, normalization.muX, normalization.sigmaX);

    err2 = squeeze(sum((xHat - xGt).^2, 1));
    seStep = seStep + sum(err2, 2);
    numCount = numCount + size(err2, 2);
end

rmsePerStep = sqrt(seStep / max(numCount, 1));
rmseMean = mean(rmsePerStep);
end

function [xHatSeq, xGtSeq] = inferBatch(transitionNet, measureNet, Z, XGT, numParticles, processStd, softResampleLambda, essThresholdRatio, useGPU)
[~, T, B] = size(Z);
Z = toDevice(single(Z), useGPU);
XGT = toDevice(single(XGT), useGPU);

x0 = squeeze(XGT(:, 1, :));
x0 = reshape(x0, 2, 1, B);
particles = repmat(x0, 1, numParticles, 1) + processStd * randn(2, numParticles, B, "like", XGT);
logw = zeros(numParticles, B, "like", XGT);

xHatSeq = zeros(2, T, B, "like", XGT);
xGtSeq = XGT;

for t = 1:T
    zt = squeeze(Z(:, t, :));
    zRep = repmat(reshape(zt, 4, 1, B), 1, numParticles, 1);

    logwNorm = logw - max(logw, [], 1);
    Wprev = exp(logwNorm);
    Wprev = Wprev ./ max(sum(Wprev, 1), 1e-8);
    ess = 1 ./ max(sum(Wprev.^2, 1), 1e-8);
    essMin = essThresholdRatio * numParticles;

    particlesRes = zeros(size(particles), "like", particles);
    logCorr = zeros(numParticles, B, "like", particles);

    for b = 1:B
        wb = Wprev(:, b);
        if ess(b) < essMin
            wMix = softResampleLambda * wb + (1 - softResampleLambda) * (1 / numParticles);
            idxA = sampleFromWeights(wMix, numParticles);
            particlesRes(:, :, b) = particles(:, idxA, b);
            logCorr(:, b) = log(wb(idxA) + 1e-8) - log(wMix(idxA) + 1e-8);
        else
            particlesRes(:, :, b) = particles(:, :, b);
            logCorr(:, b) = log(wb + 1e-8);
        end
    end
    particles = particlesRes;

    xFlat = reshape(permute(particles, [1 3 2]), 2, []);
    zFlat = reshape(permute(zRep, [1 3 2]), 4, []);

    dx = forward(transitionNet, dlarray([xFlat; zFlat], "CB"));
    xPredFlat = xFlat + extractdata(dx) + processStd * randn(size(xFlat), "like", xFlat);

    logit = forward(measureNet, dlarray([xPredFlat; zFlat], "CB"));
    logitMat = reshape(extractdata(logit), numParticles, B);

    logw = logCorr + logitMat;
    logwNorm = logw - max(logw, [], 1);
    w = exp(logwNorm);
    w = w ./ max(sum(w, 1), 1e-8);

    particles = permute(reshape(xPredFlat, 2, B, numParticles), [1 3 2]);
    xHatSeq(:, t, :) = sum(particles .* reshape(w, 1, numParticles, B), 2);
end

xHatSeq = gather(xHatSeq);
xGtSeq = gather(xGtSeq);
end

function loss = dpfLoss(transitionNet, measureNet, Z, XGT, numParticles, processStd, softResampleLambda, essThresholdRatio, useGPU)
Z = toDevice(single(Z), useGPU);
XGT = toDevice(single(XGT), useGPU);
Z = dlarray(Z);
XGT = dlarray(XGT);

[~, T, B] = size(Z);

x0 = squeeze(XGT(:, 1, :));
x0 = reshape(x0, 2, 1, B);
particles = repmat(x0, 1, numParticles, 1) + processStd * dlarray(randn(2, numParticles, B, "like", extractdata(XGT)));
logw = dlarray(zeros(numParticles, B, "like", extractdata(XGT)));

lossAcc = dlarray(single(0));
for t = 1:T
    zt = squeeze(Z(:, t, :));
    zRep = repmat(reshape(zt, 4, 1, B), 1, numParticles, 1);

    logwNorm = logw - max(logw, [], 1);
    Wprev = exp(logwNorm);
    Wprev = Wprev ./ max(sum(Wprev, 1), 1e-8);

    ess = 1 ./ max(sum(Wprev.^2, 1), 1e-8);
    essMin = essThresholdRatio * numParticles;

    particlesNum = extractdata(particles);
    WprevNum = extractdata(Wprev);
    essNum = extractdata(ess);

    particlesRes = zeros(size(particlesNum), "single");
    logCorrNum = zeros(numParticles, B, "single");

    for b = 1:B
        wb = WprevNum(:, b);
        if essNum(b) < essMin
            wMix = softResampleLambda * wb + (1 - softResampleLambda) * (1 / numParticles);
            idxA = sampleFromWeights(wMix, numParticles);
            particlesRes(:, :, b) = particlesNum(:, idxA, b);
            logCorrNum(:, b) = log(wb(idxA) + 1e-8) - log(wMix(idxA) + 1e-8);
        else
            particlesRes(:, :, b) = particlesNum(:, :, b);
            logCorrNum(:, b) = log(wb + 1e-8);
        end
    end

    particles = dlarray(particlesRes);
    logCorr = dlarray(logCorrNum);

    xFlat = reshape(permute(particles, [1 3 2]), 2, []);
    zFlat = reshape(permute(zRep, [1 3 2]), 4, []);

    dx = forward(transitionNet, dlarray([xFlat; zFlat], "CB"));
    xPredFlat = xFlat + dx + processStd * dlarray(randn(size(xFlat), "like", extractdata(xFlat)));

    logit = forward(measureNet, dlarray([xPredFlat; zFlat], "CB"));
    logitMat = reshape(logit, numParticles, B);

    logw = logCorr + logitMat;
    logwNorm = logw - max(logw, [], 1);
    w = exp(logwNorm);
    w = w ./ max(sum(w, 1), 1e-8);

    particles = permute(reshape(xPredFlat, 2, B, numParticles), [1 3 2]);
    xHat = squeeze(sum(particles .* reshape(w, 1, numParticles, B), 2));
    xGtT = squeeze(XGT(:, t, :));

    lossAcc = lossAcc + mean((xHat - xGtT).^2, "all");
end

loss = lossAcc / T;
end

function idx = sampleFromWeights(w, N)
w = gather(extractdata(w(:)));
w = w / max(sum(w), 1e-8);
cdf = cumsum(w);
u = rand(N, 1);
idx = zeros(N, 1);
for i = 1:N
    idx(i) = find(cdf >= u(i), 1, "first");
end
end

function data = loadSequenceDatasetFromRanging(h5Path, noiseDataset, trainRatio, valRatio, testRatio, seed)
Z = h5read(h5Path, noiseDataset);      % [4,T,N]
gtPos = h5read(h5Path, "/gt_position"); % [2,T]

[nZ, T, N] = size(Z);
[nX, TX] = size(gtPos);
if nZ ~= 4 || nX ~= 2 || T ~= TX
    error("Unexpected dataset shape for %s.", noiseDataset);
end

XGT = repmat(gtPos, 1, 1, N);

rng(seed);
perm = randperm(N);
nTrain = floor(trainRatio * N);
nVal = floor(valRatio * N);

idxTrain = perm(1:nTrain);
idxVal = perm(nTrain + 1:nTrain + nVal);
idxTest = perm(nTrain + nVal + 1:end);

data.ZTest = single(Z(:, :, idxTest));
data.XTestGT = single(XGT(:, :, idxTest));
end

function Z = zscoreApply3D(A, mu, sigma)
Z = (A - reshape(mu, [], 1, 1)) ./ reshape(sigma, [], 1, 1);
end

function A = zscoreInverse3D(Z, mu, sigma)
A = Z .* reshape(sigma, [], 1, 1) + reshape(mu, [], 1, 1);
end

function tf = shouldUseGPU(mode)
mode = string(lower(mode));
if mode == "gpu"
    tf = canUseGPU();
elseif mode == "auto"
    tf = canUseGPU();
else
    tf = false;
end
end

function A = toDevice(A, useGPU)
if useGPU
    A = gpuArray(A);
end
end
