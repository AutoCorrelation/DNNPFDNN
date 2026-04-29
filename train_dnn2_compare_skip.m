%% train_dnn2_compare_skip.m
% Train DNN2 with two model variants for fair comparison:
%   1) residual skip (current dnn2_create_model)
%   2) no skip direct regression (dnn2_create_model_noskip)
%
% Branch:
%   - PF dataset only

clear; clc;
rng(42);

%% Configuration
noiseLabels = ["001", "01", "1", "10", "100"];
trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;
lossType = "mse";
optimizerName = "adam";
miniBatchSize = 512;
maxEpochs = 60;
initialLearnRate = 3e-3;
seed = 42;
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

branch = struct( ...
    "name", "pf", ...
    "h5Path", "checkpoints/dnn2_dataset.h5", ...
    "checkpointPrefix", "dnn2_postprocess_" ...
);

variants = struct( ...
    "name", {"res", "noskip"}, ...
    "tag", {"res", "noskip"} ...
);

outDir = "checkpoints";
if ~isfolder(outDir)
    mkdir(outDir);
end

numRows = numel(variants) * numel(noiseLabels);
results = table('Size', [numRows, 7], ...
    'VariableTypes', ["string", "string", "string", "double", "double", "double", "string"], ...
    'VariableNames', ["Branch", "Variant", "Noise", "TrainRMSE", "ValRMSE", "TestRMSE", "Checkpoint"]);

row = 1;
fprintf("\n=== Branch: %s ===\n", upper(branch.name));

if ~isfile(branch.h5Path)
    error("Dataset not found for branch %s: %s", upper(branch.name), branch.h5Path);
end

for v = 1:numel(variants)
    variant = variants(v);
    fprintf("\n--- Variant: %s ---\n", upper(variant.name));

    for k = 1:numel(noiseLabels)
        noiseLabel = string(noiseLabels(k));
        fprintf("Training (%s, %s) noise %s\n", upper(branch.name), upper(variant.name), noiseLabel);

        data = dnn2_load_dataset(branch.h5Path, noiseLabel, trainRatio, valRatio, testRatio, seed);
        [net, trainRMSE, valRMSE, testRMSE] = trainOneDataset( ...
            data, lossType, trainOpts, variant.name);

        checkpoint = fullfile(outDir, branch.checkpointPrefix + variant.tag + "_" + noiseLabel + ".mat");

        config = struct();
        config.pipeline = "DNN1-" + upper(branch.name) + "-DNN2";
        config.variant = variant.name;
        config.h5Path = branch.h5Path;
        config.noiseLabel = noiseLabel;
        config.trainRatio = trainRatio;
        config.valRatio = valRatio;
        config.testRatio = testRatio;
        config.lossType = lossType;
        config.optimizer = optimizerName;
        config.miniBatchSize = miniBatchSize;
        config.maxEpochs = maxEpochs;
        config.initialLearnRate = initialLearnRate;
        config.validationPatience = validationPatience;
        config.learnRateSchedule = learnRateSchedule;
        config.learnRateDropFactor = learnRateDropFactor;
        config.learnRateDropPeriod = learnRateDropPeriod;
        config.seed = seed;

        normalization = struct();
        normalization.muX = data.muX;
        normalization.sigmaX = data.sigmaX;

        save(checkpoint, "net", "config", "normalization", "trainRMSE", "valRMSE", "testRMSE");

        results.Branch(row) = string(branch.name);
        results.Variant(row) = string(variant.name);
        results.Noise(row) = noiseLabel;
        results.TrainRMSE(row) = trainRMSE;
        results.ValRMSE(row) = valRMSE;
        results.TestRMSE(row) = testRMSE;
        results.Checkpoint(row) = string(checkpoint);
        row = row + 1;

        fprintf("Saved: %s\n", checkpoint);
        fprintf("Train RMSE: %.6f | Val RMSE: %.6f | Test RMSE: %.6f\n", ...
            trainRMSE, valRMSE, testRMSE);
    end
end

results = results(1:row-1, :);
fprintf("\n=== DNN2 Variant Comparison Training Summary ===\n");
disp(results);


function [net, trainRMSE, valRMSE, testRMSE] = trainOneDataset(data, lossType, trainOpts, variantName)
if variantName == "res"
    lgraph = dnn2_create_model(lossType);
elseif variantName == "noskip"
    lgraph = dnn2_create_model_noskip();
else
    error("Unsupported variant: %s", variantName);
end

opts = trainingOptions(trainOpts.optimizerName, ...
    MaxEpochs=trainOpts.maxEpochs, ...
    MiniBatchSize=trainOpts.miniBatchSize, ...
    InitialLearnRate=trainOpts.initialLearnRate, ...
    Shuffle="every-epoch", ...
    ValidationData={data.XVal, data.YVal}, ...
    ValidationFrequency=max(1, floor(size(data.XTrain,1)/trainOpts.miniBatchSize)), ...
    ValidationPatience=trainOpts.validationPatience, ...
    LearnRateSchedule=trainOpts.learnRateSchedule, ...
    LearnRateDropFactor=trainOpts.learnRateDropFactor, ...
    LearnRateDropPeriod=trainOpts.learnRateDropPeriod, ...
    Plots="training-progress");

net = trainNetwork(data.XTrain, data.YTrain, lgraph, opts);

yHatTrain = predict(net, data.XTrain);
yHatVal = predict(net, data.XVal);
yHatTest = predict(net, data.XTest);

trainRMSE = rmse(yHatTrain, data.YTrain);
valRMSE = rmse(yHatVal, data.YVal);
testRMSE = rmse(yHatTest, data.YTest);
end

function v = rmse(yHat, y)
v = sqrt(mean((yHat - y).^2, "all"));
end
