%% train_dnn1.m
% DNN1: first-stage residual MLP denoising (all noise variances).
% Sequence:
% 1) load data
% 2) split train/val/test
% 3) z-score normalization (train stats only)
% 4) define residual model and initialize dlnetwork
% 5) train with trainnet

clear; clc;
rng(42);

%% Configuration
h5Path = "ranging_data_cv.h5";
noiseDatasets = ["/ranging_001", "/ranging_01", "/ranging_1", "/ranging_10", "/ranging_100"];

trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;

maxEpochs = 60;
miniBatchSize = 512;
initialLearnRate = 1e-3;
optimizerName = "adam";
validationPatience = 10;
learnRateSchedule = "cosine";
% learnRateSchedule = "piecewise";

results = table('Size', [numel(noiseDatasets), 4], ...
    'VariableTypes', ["string", "double", "double", "double"], ...
    'VariableNames', ["noise", "trainRMSE", "valRMSE", "testRMSE"]);

for k = 1:numel(noiseDatasets)
    datasetName = string(noiseDatasets(k));

    [metrics, outFile] = trainOneDataset( ...
        h5Path, datasetName, trainRatio, valRatio, testRatio, ...
        optimizerName, maxEpochs, miniBatchSize, ...
        initialLearnRate, validationPatience, learnRateSchedule);

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
    optimizerName, maxEpochs, miniBatchSize, ...
    initialLearnRate, validationPatience, learnRateSchedule)

noisy = h5read(h5Path, noiseDataset);      % [4, step, sample]
gtRanging = h5read(h5Path, "/gt_ranging"); % [4, step]

[numAnchors, numSteps, numSamples] = size(noisy);
if numAnchors ~= 4
    error("Expected 4 anchors, but got %d.", numAnchors);
end

clean = repmat(gtRanging, 1, 1, numSamples); % [4, step, sample]

% Flatten to independent step-wise samples: [N, 4]
X = reshape(noisy, 4, numSteps * numSamples)';
Y = reshape(clean, 4, numSteps * numSamples)';
N = size(X, 1);

% Split indices
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

net = dlnetwork;
% Residual MLP: y = x + f(x)
layers = [
    featureInputLayer(4, Name="input")
    fullyConnectedLayer(32, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(16, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(32, Name="fc3")
    reluLayer(Name="relu3")
    fullyConnectedLayer(4, Name="delta")
    additionLayer(2, Name="skipadd")
];
net = addLayers(net, layers);
lgraph = connectLayers(net, "input", "skipadd/in2");

% Initialize dlnetwork then train with trainnet (R2025b style)

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

% trainnet returns trained dlnetwork for regression task.
net = trainnet(XTrainZ, YTrainZ, lgraph, "mse", opts);

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
outFile = fullfile(outDir, "dnn1_residual_trainnet_" + noiseTag + ".mat");

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
config.seed = 42;

save(outFile, "net", "normalization", "metrics", "config", ...
    "idxTrain", "idxVal", "idxTest");
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
