%% train_dnn2.m
% Train DNN2 postprocessor models in MATLAB.
% Architecture:
%   Linear(6, 128) -> BatchNorm1d(128) -> ReLU
%   Linear(128, 64) -> BatchNorm1d(64) -> ReLU
%   Linear(64, 2)
%
% Input : [pf_x, pf_y, r1, r2, r3, r4]
% Target: [x, y]
%
% Noise-specific models are trained independently and saved to checkpoints.

clear; clc;
rng(42);

%% Configuration
h5Path = "checkpoints/dnn2_dataset.h5";
noiseLabels = ["001", "01", "1", "10", "100"];
trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;
lossType = "mse";
miniBatchSize = 512;
maxEpochs = 80;
initialLearnRate = 1e-3;
seed = 42;

if ~isfile(h5Path)
    error("DNN2 dataset not found: %s", h5Path);
end

outDir = "checkpoints";
if ~isfolder(outDir)
    mkdir(outDir);
end

results = table('Size', [numel(noiseLabels), 5], ...
    'VariableTypes', ["string", "double", "double", "double", "string"], ...
    'VariableNames', ["Noise", "TrainRMSE", "ValRMSE", "TestRMSE", "Checkpoint"]);

for k = 1:numel(noiseLabels)
    noiseLabel = string(noiseLabels(k));
    fprintf("\n=== Training DNN2 for noise %s ===\n", noiseLabel);

    data = dnn2_load_dataset(h5Path, noiseLabel, trainRatio, valRatio, testRatio, seed);

    lgraph = dnn2_create_model(lossType);
    opts = trainingOptions("adam", ...
        MaxEpochs=maxEpochs, ...
        MiniBatchSize=miniBatchSize, ...
        InitialLearnRate=initialLearnRate, ...
        Shuffle="every-epoch", ...
        ValidationData={data.XVal, data.YVal}, ...
        ValidationFrequency=max(1, floor(size(data.XTrain, 1) / miniBatchSize)), ...
        Verbose=true, ...
        Plots="training-progress");

    net = trainNetwork(data.XTrain, data.YTrain, lgraph, opts);

    YHatTrain = predict(net, data.XTrain);
    YHatVal = predict(net, data.XVal);
    YHatTest = predict(net, data.XTest);

    trainRMSE = rmse(YHatTrain, data.YTrain);
    valRMSE = rmse(YHatVal, data.YVal);
    testRMSE = rmse(YHatTest, data.YTest);

    checkpoint = fullfile(outDir, "dnn2_postprocess_" + noiseLabel + ".mat");

    config = struct();
    config.h5Path = h5Path;
    config.noiseLabel = noiseLabel;
    config.trainRatio = trainRatio;
    config.valRatio = valRatio;
    config.testRatio = testRatio;
    config.lossType = lossType;
    config.miniBatchSize = miniBatchSize;
    config.maxEpochs = maxEpochs;
    config.initialLearnRate = initialLearnRate;
    config.seed = seed;

    normalization = struct();
    normalization.muX = data.muX;
    normalization.sigmaX = data.sigmaX;

    save(checkpoint, "net", "config", "normalization", "trainRMSE", "valRMSE", "testRMSE");

    results.Noise(k) = noiseLabel;
    results.TrainRMSE(k) = trainRMSE;
    results.ValRMSE(k) = valRMSE;
    results.TestRMSE(k) = testRMSE;
    results.Checkpoint(k) = string(checkpoint);

    fprintf("Saved: %s\n", checkpoint);
    fprintf("Train RMSE: %.6f | Val RMSE: %.6f | Test RMSE: %.6f\n", trainRMSE, valRMSE, testRMSE);
end

fprintf("\n=== DNN2 Training Summary ===\n");
disp(results);


function v = rmse(YHat, Y)
v = sqrt(mean((YHat - Y).^2, "all"));
end