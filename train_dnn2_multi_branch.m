%% train_dnn2_multi_branch.m
% DNN2: postprocessor MLP training for both PF/LS branches in one file.
% Baseline model/options/flow are kept identical to y_train_dnn2.m.

clear; clc;
rng(42);

%% Configuration (baseline-aligned)
branches = struct( ...
    "name", {"pf", "ls"}, ...
    "h5Path", {"checkpoints/dnn2_dataset.h5", "checkpoints/dnn2_dataset_ls.h5"}, ...
    "checkpointPrefix", {"dnn2_postprocess_", "dnn2_postprocess_ls_"} ...
);

noiseLabels = ["001", "01", "1", "10", "100"];
trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;

maxEpochs = 60;
miniBatchSize = 512;
initialLearnRate = 1e-3;
optimizerName = "adam";
validationPatience = 10;
learnRateSchedule = "cosine";
seed = 42;

assert(abs(trainRatio + valRatio + testRatio - 1.0) < 1e-12, ...
    "Split ratios must sum to 1.");

outDir = "checkpoints";
if ~isfolder(outDir)
    mkdir(outDir);
end

numRows = numel(branches) * numel(noiseLabels);
results = table('Size', [numRows, 6], ...
    'VariableTypes', ["string", "string", "double", "double", "double", "string"], ...
    'VariableNames', ["branch", "noise", "trainRMSE", "valRMSE", "testRMSE", "checkpoint"]);

row = 1;
for b = 1:numel(branches)
    branch = branches(b);

    if ~isfile(branch.h5Path)
        fprintf("Dataset not found. Skip branch %s: %s\n", upper(branch.name), branch.h5Path);
        for k = 1:numel(noiseLabels)
            results.branch(row) = string(branch.name);
            results.noise(row) = string(noiseLabels(k));
            results.trainRMSE(row) = NaN;
            results.valRMSE(row) = NaN;
            results.testRMSE(row) = NaN;
            results.checkpoint(row) = "";
            row = row + 1;
        end
        continue;
    end

    fprintf("\n=== Branch: %s ===\n", upper(branch.name));
    for k = 1:numel(noiseLabels)
        noiseLabel = string(noiseLabels(k));

        [metrics, outFile] = trainOneNoise( ...
            branch.h5Path, branch.name, branch.checkpointPrefix, noiseLabel, ...
            trainRatio, valRatio, testRatio, seed, optimizerName, ...
            maxEpochs, miniBatchSize, initialLearnRate, ...
            validationPatience, learnRateSchedule);

        results.branch(row) = string(branch.name);
        results.noise(row) = noiseLabel;
        results.trainRMSE(row) = metrics.trainRMSE;
        results.valRMSE(row) = metrics.valRMSE;
        results.testRMSE(row) = metrics.testRMSE;
        results.checkpoint(row) = string(outFile);
        row = row + 1;

        fprintf("Saved: %s\n", outFile);
    end
end

fprintf("\n=== DNN2 Multi-Branch Training Summary ===\n");
disp(results);

%% Local functions
function [metrics, outFile] = trainOneNoise( ...
    h5Path, branchName, checkpointPrefix, noiseLabel, ...
    trainRatio, valRatio, testRatio, seed, optimizerName, ...
    maxEpochs, miniBatchSize, initialLearnRate, ...
    validationPatience, learnRateSchedule)

data = loadDnn2Dataset(h5Path, noiseLabel, trainRatio, valRatio, testRatio, seed);
dlnetInit = createDnn2ResidualModel();

opts = trainingOptions(optimizerName, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=initialLearnRate, ...
    Shuffle="every-epoch", ...
    ValidationData={data.XVal, data.YVal}, ...
    ValidationFrequency=max(1, floor(size(data.XTrain, 1) / miniBatchSize)), ...
    ValidationPatience=validationPatience, ...
    LearnRateSchedule=learnRateSchedule, ...
    Verbose=true, ...
    Plots="training-progress");

net = trainnet(data.XTrain, data.YTrain, dlnetInit, "l1loss", opts);

YHatTrain = minibatchpredict(net, data.XTrain);
YHatVal = minibatchpredict(net, data.XVal);
YHatTest = minibatchpredict(net, data.XTest);

metrics.trainRMSE = rmse(YHatTrain, data.YTrain);
metrics.valRMSE = rmse(YHatVal, data.YVal);
metrics.testRMSE = rmse(YHatTest, data.YTest);
metrics.trainMSE = mean((YHatTrain - data.YTrain).^2, "all");
metrics.valMSE = mean((YHatVal - data.YVal).^2, "all");
metrics.testMSE = mean((YHatTest - data.YTest).^2, "all");

fprintf("\n=== Training Result (%s, noise %s) ===\n", upper(branchName), noiseLabel);
fprintf("Train RMSE: %.6f | Val RMSE: %.6f | Test RMSE: %.6f\n", ...
    metrics.trainRMSE, metrics.valRMSE, metrics.testRMSE);

outDir = "checkpoints";
noiseTag = string(noiseLabel);
outFile = fullfile(outDir, checkpointPrefix + noiseTag + ".mat");

normalization.muX = data.muX;
normalization.sigmaX = data.sigmaX;

config.h5Path = h5Path;
config.branch = branchName;
config.noiseLabel = noiseLabel;
config.trainRatio = trainRatio;
config.valRatio = valRatio;
config.testRatio = testRatio;
config.nTrain = data.nTrain;
config.nVal = data.nVal;
config.nTest = data.nTest;
config.maxEpochs = maxEpochs;
config.miniBatchSize = miniBatchSize;
config.initialLearnRate = initialLearnRate;
config.optimizer = optimizerName;
config.validationPatience = validationPatience;
config.learnRateSchedule = learnRateSchedule;
config.loss = "mse";
config.seed = seed;

idxTrain = data.idxTrain;
idxVal = data.idxVal;
idxTest = data.idxTest;
save(outFile, "net", "normalization", "metrics", "config", ...
    "idxTrain", "idxVal", "idxTest");
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
nTest = numSamples - nTrain - nVal;

idxTrain = perm(1:nTrain);
idxVal = perm(nTrain + 1:nTrain + nVal);
idxTest = perm(nTrain + nVal + 1:end);

XTrain = packSamples(xPre, xPF, idxTrain);
YTrain = packTargets(gtPos, idxTrain);
XVal = packSamples(xPre, xPF, idxVal);
YVal = packTargets(gtPos, idxVal);
XTest = packSamples(xPre, xPF, idxTest);
YTest = packTargets(gtPos, idxTest);

data = struct();
data.groupPath = groupPath;
data.numSamples = numSamples;
data.numSteps = numSteps;
data.nTrain = nTrain;
data.nVal = nVal;
data.nTest = nTest;
data.idxTrain = idxTrain;
data.idxVal = idxVal;
data.idxTest = idxTest;
data.XTrain = XTrain;
data.YTrain = YTrain;
data.XVal = XVal;
data.YVal = YVal;
data.XTest = XTest;
data.YTest = YTest;
data.muX = [];
data.sigmaX = [];
end

function net = createDnn2Model()
layers = [
    featureInputLayer(6, Normalization="none", Name="input")
    fullyConnectedLayer(128, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(64, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(2, Name="out")
];
net = dlnetwork;
net = addLayers(net, layers);
end

function net = createDnn2ResidualModel()
% Residual model: y = xy_skip + delta
layers = [
    featureInputLayer(6, Normalization="none", Name="input")
    fullyConnectedLayer(128, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(64, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(2, Name="delta")
    additionLayer(2, Name="skipadd")
];

net = dlnetwork;
net = addLayers(net, layers);
net = addLayers(net, functionLayer(@selectXY, Name="xy_skip"));
net = connectLayers(net, "input", "xy_skip");
net = connectLayers(net, "xy_skip", "skipadd/in2");
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

function v = rmse(A, B)
v = sqrt(mean((A - B).^2, "all"));
end
