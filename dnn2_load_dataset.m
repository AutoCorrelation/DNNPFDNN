function data = dnn2_load_dataset(h5Path, noiseLabel, trainRatio, valRatio, testRatio, seed)
%DNN2_LOAD_DATASET Load one noise group's DNN2 data and split it by sample.
%   data = dnn2_load_dataset(h5Path, noiseLabel, trainRatio, valRatio, testRatio, seed)
%
% Returns a struct with fields:
%   XTrain, YTrain, XVal, YVal, XTest, YTest
%   muX, sigmaX, groupPath, noiseLabel, numSamples, numSteps
%
% Dataset layout expected in H5:
%   /<noiseLabel>/xPre   [4, T, N]
%   /<noiseLabel>/xPF    [2, T, N]
%   /<noiseLabel>/gtPos  [2, T, N]

arguments
    h5Path (1,1) string
    noiseLabel (1,1) string
    trainRatio (1,1) double = 0.80
    valRatio (1,1) double = 0.10
    testRatio (1,1) double = 0.10
    seed (1,1) double = 42
end

assert(abs(trainRatio + valRatio + testRatio - 1.0) < 1e-12, "Split ratios must sum to 1.");

groupPath = "/" + noiseLabel;
if ~isfile(h5Path)
    error("H5 file not found: %s", h5Path);
end

xPre = h5read(h5Path, groupPath + "/xPre");   % [4, T, N]
xPF = h5read(h5Path, groupPath + "/xPF");     % [2, T, N]
gtPos = h5read(h5Path, groupPath + "/gtPos");  % [2, T, N]

[nPre, numSteps, numSamples] = size(xPre);
[nPF, numStepsPF, numSamplesPF] = size(xPF);
[nGT, numStepsGT, numSamplesGT] = size(gtPos);

if nPre ~= 4 || nPF ~= 2 || nGT ~= 2 || numSteps ~= numStepsPF || numSteps ~= numStepsGT || ...
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

[muX, sigmaX] = channelStats(XTrain);
XTrain = zscoreApply(XTrain, muX, sigmaX);
XVal = zscoreApply(XVal, muX, sigmaX);
XTest = zscoreApply(XTest, muX, sigmaX);

data = struct();
data.h5Path = h5Path;
data.noiseLabel = noiseLabel;
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
data.muX = muX;
data.sigmaX = sigmaX;
end

function X = packSamples(xPre, xPF, indices)
xPreSel = xPre(:, :, indices);   % [4, T, N]
xPFSel = xPF(:, :, indices);     % [2, T, N]
X = [transposeToRows(xPFSel), transposeToRows(xPreSel)];
end

function Y = packTargets(gtPos, indices)
Y = transposeToRows(gtPos(:, :, indices));
end

function X = transposeToRows(A)
% Convert [C, T, N] -> [N*T, C]
X = permute(A, [3 2 1]);
X = reshape(X, [], size(A, 1));
end

function [mu, sigma] = channelStats(A)
mu = mean(A, 1);
sigma = std(A, 0, 1);
sigma(sigma < 1e-12) = 1;
end

function Z = zscoreApply(A, mu, sigma)
Z = (A - mu) ./ sigma;
end