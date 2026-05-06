%% train_directDNN.m
% Direct DNN regression: ranging(4) -> position(2)
% Sequence:
% 1) load data
% 2) split train/val/test
% 3) z-score normalization (train stats only)
% 4) define direct regression model and initialize dlnetwork
% 5) train with trainnet and save per-noise checkpoints

clear; clc;
rng(42);

%% Configuration
h5Path = "ranging_data_cv.h5";
noiseDatasets = ["/ranging_001", "/ranging_01", "/ranging_1", "/ranging_10", "/ranging_100"];

trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;

maxEpochs = 80;
miniBatchSize = 512;
initialLearnRate = 1e-3;
optimizerName = "adam";
validationPatience = 10;
learnRateSchedule = "cosine";
seed = 42;

assert(abs(trainRatio + valRatio + testRatio - 1.0) < 1e-12, ...
    "Split ratios must sum to 1.");
if ~isfile(h5Path)
    error("File not found: %s", h5Path);
end

results = table('Size', [numel(noiseDatasets), 5], ...
    'VariableTypes', ["string", "double", "double", "double", "string"], ...
    'VariableNames', ["noise", "trainRMSE", "valRMSE", "testRMSE", "checkpoint"]);

for k = 1:numel(noiseDatasets)
    datasetName = string(noiseDatasets(k));

    [metrics, outFile] = trainOneDataset( ...
        h5Path, datasetName, trainRatio, valRatio, testRatio, ...
        optimizerName, maxEpochs, miniBatchSize, initialLearnRate, ...
        validationPatience, learnRateSchedule, seed);

    results.noise(k) = datasetName;
    results.trainRMSE(k) = metrics.trainRMSE;
    results.valRMSE(k) = metrics.valRMSE;
    results.testRMSE(k) = metrics.testRMSE;
    results.checkpoint(k) = string(outFile);

    fprintf("Saved: %s\n", outFile);
end

fprintf("\n=== Direct DNN Summary (All Trained Variances) ===\n");
disp(results);

%% Local functions
function [metrics, outFile] = trainOneDataset( ...
    h5Path, noiseDataset, trainRatio, valRatio, testRatio, ...
    optimizerName, maxEpochs, miniBatchSize, initialLearnRate, ...
    validationPatience, learnRateSchedule, seed)

noisy = h5read(h5Path, noiseDataset);  % [4, step, sample]
gtPos = h5read(h5Path, "/gt_position");     % [2, step]

[numAnchors, numSteps, numSamples] = size(noisy);
if numAnchors ~= 4
    error("Expected 4 anchors, but got %d.", numAnchors);
end
if size(gtPos, 1) ~= 2
    error("Expected gt_position with 2 channels [x;y], but got %d.", size(gtPos, 1));
end
if size(gtPos, 2) ~= numSteps
    error("gt_position step size mismatch: gt_position=%d, noisy=%d.", size(gtPos, 2), numSteps);
end

% Build target with same [step,sample] coverage as noisy
pos = repmat(gtPos, 1, 1, numSamples);  % [2, step, sample]

% Flatten into independent step-wise samples
X = reshape(noisy, 4, numSteps * numSamples)';  % [N,4]
Y = reshape(pos, 2, numSteps * numSamples)';    % [N,2]
N = size(X, 1);

rng(seed);
idx = randperm(N);
nTrain = floor(trainRatio * N);
nVal = floor(valRatio * N);
nTest = N - nTrain - nVal;

idxTrain = idx(1:nTrain);
idxVal = idx(nTrain + 1:nTrain + nVal);
idxTest = idx(nTrain + nVal + 1:end);

XTrain = X(idxTrain, :);
YTrain = Y(idxTrain, :);
XVal = X(idxVal, :);
YVal = Y(idxVal, :);
XTest = X(idxTest, :);
YTest = Y(idxTest, :);

% Z-score normalization (train statistics only)
[muX, sigmaX] = channelStats(XTrain);
[muY, sigmaY] = channelStats(YTrain);

XTrainZ = zscoreApply(XTrain, muX, sigmaX);
XValZ = zscoreApply(XVal, muX, sigmaX);
XTestZ = zscoreApply(XTest, muX, sigmaX);

YTrainZ = zscoreApply(YTrain, muY, sigmaY);
YValZ = zscoreApply(YVal, muY, sigmaY);

dlnetInit = createDirectModel();

opts = trainingOptions(optimizerName, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=initialLearnRate, ...
    Shuffle="every-epoch", ...
    ValidationData={XValZ, YValZ}, ...
    ValidationFrequency=max(1, floor(nTrain / miniBatchSize)), ...
    ValidationPatience=validationPatience, ...
    LearnRateSchedule=learnRateSchedule, ...
    Metrics="rmse", ...
    Verbose=true, ...
    Plots="training-progress");

net = trainnet(XTrainZ, YTrainZ, dlnetInit, "l1loss", opts);

YHatTrainZ = minibatchpredict(net, XTrainZ);
YHatValZ = minibatchpredict(net, XValZ);
YHatTestZ = minibatchpredict(net, XTestZ);

YHatTrain = zscoreInverse(YHatTrainZ, muY, sigmaY);
YHatVal = zscoreInverse(YHatValZ, muY, sigmaY);
YHatTest = zscoreInverse(YHatTestZ, muY, sigmaY);

metrics.trainRMSE = rmse(YHatTrain, YTrain);
metrics.valRMSE = rmse(YHatVal, YVal);
metrics.testRMSE = rmse(YHatTest, YTest);
metrics.trainMSE = mean((YHatTrain - YTrain).^2, "all");
metrics.valMSE = mean((YHatVal - YVal).^2, "all");
metrics.testMSE = mean((YHatTest - YTest).^2, "all");

fprintf("\n=== Training Result (%s) ===\n", noiseDataset);
fprintf("Train RMSE: %.6f | Val RMSE: %.6f | Test RMSE: %.6f\n", ...
    metrics.trainRMSE, metrics.valRMSE, metrics.testRMSE);

outDir = "checkpoints";
if ~isfolder(outDir)
    mkdir(outDir);
end

noiseTag = erase(noiseDataset, "/ranging_");
outFile = fullfile(outDir, "direct_dnn_pos_" + noiseTag + ".mat");

normalization.muX = muX;
normalization.sigmaX = sigmaX;
normalization.muY = muY;
normalization.sigmaY = sigmaY;

config.h5Path = h5Path;
config.noiseDataset = noiseDataset;
config.trainRatio = trainRatio;
config.valRatio = valRatio;
config.testRatio = testRatio;
config.nTrain = nTrain;
config.nVal = nVal;
config.nTest = nTest;
config.maxEpochs = maxEpochs;
config.miniBatchSize = miniBatchSize;
config.initialLearnRate = initialLearnRate;
config.optimizer = optimizerName;
config.validationPatience = validationPatience;
config.learnRateSchedule = learnRateSchedule;
config.loss = "mse";
config.seed = seed;
config.target = "[x,y]";
config.modelType = "direct-regression";

save(outFile, "net", "normalization", "metrics", "config", ...
    "idxTrain", "idxVal", "idxTest");
end

function net = createDirectModel()
layers = [
    featureInputLayer(4, Normalization="none", Name="input")
    fullyConnectedLayer(128, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(64, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(2, Name="out")
];

net = dlnetwork;
net = addLayers(net, layers);
end

function [mu, sigma] = channelStats(A)
mu = mean(A, 1);
sigma = std(A, 0, 1);
sigma(sigma < 1e-12) = 1;
end

function Z = zscoreApply(A, mu, sigma)
Z = (A - mu) ./ sigma;
end

function A = zscoreInverse(Z, mu, sigma)
A = Z .* sigma + mu;
end

function v = rmse(A, B)
v = sqrt(mean((A - B).^2, "all"));
end
