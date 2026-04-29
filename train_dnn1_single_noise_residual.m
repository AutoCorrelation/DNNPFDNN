%% train_dnn1_single_noise_residual.m
% DNN1: first-stage residual MLP denoising.
% Input  : 4x1 noisy ranging per step
% Target : 4x1 clean ranging per step
% Model  : residual MLP (y = x + f(x))

clear; clc;
rng(42);

%% Configuration
h5Path = "ranging_data_cv.h5";
noiseDataset = "/ranging_01";   % used only if you switch targets to single variance
noiseDatasets = ["/ranging_001", "/ranging_01", "/ranging_1", "/ranging_10", "/ranging_100"];
trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;

numHidden = 64;
maxEpochs = 60;
miniBatchSize = 512;
initialLearnRate = 1e-3;
optimizerName = "adam";
validationPatience = 10;
learnRateSchedule = "piecewise";
learnRateDropFactor = 0.1;
learnRateDropPeriod = 20;

trainOpts = struct();
trainOpts.optimizerName = optimizerName;
trainOpts.maxEpochs = maxEpochs;
trainOpts.miniBatchSize = miniBatchSize;
trainOpts.initialLearnRate = initialLearnRate;
trainOpts.validationPatience = validationPatience;
trainOpts.learnRateSchedule = learnRateSchedule;
trainOpts.learnRateDropFactor = learnRateDropFactor;
trainOpts.learnRateDropPeriod = learnRateDropPeriod;

%% Load HDF5
assert(abs(trainRatio + valRatio + testRatio - 1.0) < 1e-12, "Split ratios must sum to 1.");

if ~isfile(h5Path)
    error("File not found: %s", h5Path);
end

targets = noiseDatasets;
% For single variance training, use: targets = string(noiseDataset);

results = table('Size', [numel(targets), 4], ...
    'VariableTypes', ["string", "double", "double", "double"], ...
    'VariableNames', ["noise", "trainRMSE", "valRMSE", "testRMSE"]);

for k = 1:numel(targets)
    datasetName = string(targets(k));
    [metrics, outFile] = trainOneDataset( ...
        h5Path, datasetName, trainRatio, valRatio, testRatio, ...
        numHidden, trainOpts);

    results.noise(k) = datasetName;
    results.trainRMSE(k) = metrics.trainRMSE;
    results.valRMSE(k) = metrics.valRMSE;
    results.testRMSE(k) = metrics.testRMSE;

    fprintf("Saved: %s\n", outFile);
end

fprintf("\n=== Summary (All Trained Variances) ===\n");
disp(results);

%% Local functions
function [metrics, outFile] = trainOneDataset( ...
    h5Path, noiseDataset, trainRatio, valRatio, testRatio, ...
    numHidden, trainOpts)

noisy = h5read(h5Path, noiseDataset);      % [4, step, sample]
gtRanging = h5read(h5Path, "/gt_ranging"); % [4, step]

[numAnchors, numSteps, numSamples] = size(noisy);
if numAnchors ~= 4
    error("Expected 4 anchors, but got %d.", numAnchors);
end

% Build target with identical shape as noisy: [4, step, sample]
clean = repmat(gtRanging, 1, 1, numSamples);

% Flatten into independent step-wise samples.
% X, Y shape: [N, 4]
X = reshape(noisy, 4, numSteps * numSamples)';
Y = reshape(clean, 4, numSteps * numSamples)';
N = size(X, 1);

% Split indices (variance-independent split)
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

% Residual MLP: y = x + f(x)
layers = [
    featureInputLayer(4, Normalization="none", Name="input")
    fullyConnectedLayer(32, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(numHidden, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(32, Name="fc3")
    reluLayer(Name="relu3")
    fullyConnectedLayer(4, Name="delta")
    additionLayer(2, Name="skipadd")
    regressionLayer(Name="reg")
];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, "input", "skipadd/in2");

opts = trainingOptions(trainOpts.optimizerName, ...
    MaxEpochs=trainOpts.maxEpochs, ...
    MiniBatchSize=trainOpts.miniBatchSize, ...
    InitialLearnRate=trainOpts.initialLearnRate, ...
    Shuffle="every-epoch", ...
    ValidationData={XValZ, YValZ}, ...
    ValidationFrequency=max(1, floor(nTrain / trainOpts.miniBatchSize)), ...
    ValidationPatience=trainOpts.validationPatience, ...
    LearnRateSchedule=trainOpts.learnRateSchedule, ...
    LearnRateDropFactor=trainOpts.learnRateDropFactor, ...
    LearnRateDropPeriod=trainOpts.learnRateDropPeriod, ...
    Plots="training-progress");

net = trainNetwork(XTrainZ, YTrainZ, lgraph, opts);

YHatTrainZ = predict(net, XTrainZ);
YHatValZ = predict(net, XValZ);
YHatTestZ = predict(net, XTestZ);

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
outFile = fullfile(outDir, "dnn1_residual_single_" + noiseTag + ".mat");

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
config.numHidden = numHidden;
config.maxEpochs = trainOpts.maxEpochs;
config.miniBatchSize = trainOpts.miniBatchSize;
config.initialLearnRate = trainOpts.initialLearnRate;
config.optimizer = trainOpts.optimizerName;
config.validationPatience = trainOpts.validationPatience;
config.learnRateSchedule = trainOpts.learnRateSchedule;
config.learnRateDropFactor = trainOpts.learnRateDropFactor;
config.learnRateDropPeriod = trainOpts.learnRateDropPeriod;

save(outFile, "net", "normalization", "metrics", "config", "idxTrain", "idxVal", "idxTest");
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
