%% train_dnn2_multi_branch.m
% Train DNN2 postprocessor models for both branches in one script:
%   - DNN1-PF-DNN2
%   - DNN1-LS-DNN2
%
% Existing scripts/files are kept unchanged.
% Loss is fixed to MSE.

clear; clc;
rng(42);

%% Configuration
noiseLabels = ["001", "01", "1", "10", "100"];
trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;
lossType = "mse";   % fixed by design
optimizerName = "adam"; % fixed by design
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

branches = struct( ...
    "name", {"pf", "ls"}, ...
    "h5Path", {"checkpoints/dnn2_dataset.h5", "checkpoints/dnn2_dataset_ls.h5"}, ...
    "checkpointPrefix", {"dnn2_postprocess_", "dnn2_postprocess_ls_"} ...
);

outDir = "checkpoints";
if ~isfolder(outDir)
    mkdir(outDir);
end

numRows = numel(branches) * numel(noiseLabels);
results = table('Size', [numRows, 6], ...
    'VariableTypes', ["string", "string", "double", "double", "double", "string"], ...
    'VariableNames', ["Branch", "Noise", "TrainRMSE", "ValRMSE", "TestRMSE", "Checkpoint"]);

row = 1;
for b = 1:numel(branches)
    branch = branches(b);
    fprintf("\n=== Branch: %s ===\n", upper(branch.name));

    if ~isfile(branch.h5Path)
        fprintf("Dataset not found. Skip branch %s: %s\n", upper(branch.name), branch.h5Path);
        for k = 1:numel(noiseLabels)
            results.Branch(row) = string(branch.name);
            results.Noise(row) = string(noiseLabels(k));
            results.TrainRMSE(row) = NaN;
            results.ValRMSE(row) = NaN;
            results.TestRMSE(row) = NaN;
            results.Checkpoint(row) = "";
            row = row + 1;
        end
        continue;
    end

    for k = 1:numel(noiseLabels)
        noiseLabel = string(noiseLabels(k));
        fprintf("\nTraining DNN2 (%s) for noise %s\n", upper(branch.name), noiseLabel);

        data = dnn2_load_dataset(branch.h5Path, noiseLabel, trainRatio, valRatio, testRatio, seed);
        [net, trainRMSE, valRMSE, testRMSE] = trainOneDataset( ...
            data, lossType, trainOpts);

        checkpoint = fullfile(outDir, branch.checkpointPrefix + noiseLabel + ".mat");

        config = struct();
        config.pipeline = "DNN1-" + upper(branch.name) + "-DNN2";
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

fprintf("\n=== DNN2 Multi-Branch Training Summary ===\n");
disp(results);


function [net, trainRMSE, valRMSE, testRMSE] = trainOneDataset(data, lossType, trainOpts)
lgraph = dnn2_create_model(lossType);

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
