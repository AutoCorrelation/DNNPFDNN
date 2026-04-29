%% filter_lls_cv.m
% No PF in this script.
% 1) Load noisy ranging data
% 2) Apply DNN1 preprocessing (denoised ranging)
% 3) Estimate position by LLS at each step
% 4) Print RMSE summary

clear; clc;
rng(42);

%% Configuration
h5Path = "ranging_data_cv.h5";
anchors = [0 10; 0 0; 10 0; 10 10]';  % [2, 4]
noiseDatasets = ["/ranging_001", "/ranging_01", "/ranging_1", "/ranging_10", "/ranging_100"];
noiseLabels = ["001", "01", "1", "10", "100"];

if ~isfile(h5Path)
	error("Data file not found: %s", h5Path);
end

gtPos = h5read(h5Path, "/gt_position");  % [2, step]
[nPos, nStep] = size(gtPos);
if nPos ~= 2
	error("Expected gt_position size [2, step], got [%d, %d].", nPos, nStep);
end

targets = noiseDatasets;
% For single variance evaluation, use: targets = string("/ranging_01");

summaryTable = table('Size', [numel(targets), 4], ...
	'VariableTypes', ["string", "double", "double", "double"], ...
	'VariableNames', ["Noise", "MeanStepRMSE", "StdStepRMSE", "MeanSampleRMSE"]);

outFile = "checkpoints/lls_dnn1_results.h5";
if isfile(outFile)
	delete(outFile);
end

for k = 1:numel(targets)
	datasetName = string(targets(k));
	noiseLabel = extractAfter(datasetName, "/ranging_");
	dnn1File = "checkpoints/dnn1_residual_single_" + noiseLabel + ".mat";

	if ~isfile(dnn1File)
		error("Checkpoint not found: %s", dnn1File);
	end

	dnnData = load(dnn1File, "net", "normalization");
	net = dnnData.net;
	normalization = dnnData.normalization;

	xRaw = h5read(h5Path, datasetName);  % [4, step, sample]
	[nFeat, nStepRaw, nSample] = size(xRaw);
	if nFeat ~= 4 || nStepRaw ~= nStep
		error("Unexpected measurement size for %s.", datasetName);
	end

	xPreStack = zeros(4, nStep, nSample);
	xLLSStack = zeros(2, nStep, nSample);
	rmseStep = zeros(nStep, nSample);
	rmseSample = zeros(nSample, 1);

	for s = 1:nSample
		xSample = xRaw(:, :, s);                    % [4, step]
		xPre = preprocess_dnn1_predict(net, normalization, xSample);  % [4, step]

		xLLS = zeros(2, nStep);
		for t = 1:nStep
			xLLS(:, t) = llsPositionFromRanging(xPre(:, t), anchors);
			rmseStep(t, s) = sqrt(mean((xLLS(:, t) - gtPos(:, t)).^2));
		end

		rmseSample(s) = mean(rmseStep(:, s));
		xPreStack(:, :, s) = xPre;
		xLLSStack(:, :, s) = xLLS;
	end

	meanStepRMSE = mean(rmseStep, "all");
	stdStepRMSE = std(rmseStep, 0, "all");
	meanSampleRMSE = mean(rmseSample);

	summaryTable.Noise(k) = noiseLabel;
	summaryTable.MeanStepRMSE(k) = meanStepRMSE;
	summaryTable.StdStepRMSE(k) = stdStepRMSE;
	summaryTable.MeanSampleRMSE(k) = meanSampleRMSE;

	grpPath = "/" + noiseLabel;
	h5create(outFile, grpPath + "/xPre", size(xPreStack));
	h5write(outFile, grpPath + "/xPre", xPreStack);
	h5create(outFile, grpPath + "/xLLS", size(xLLSStack));
	h5write(outFile, grpPath + "/xLLS", xLLSStack);
	h5create(outFile, grpPath + "/rmseStep", size(rmseStep));
	h5write(outFile, grpPath + "/rmseStep", rmseStep);
	h5create(outFile, grpPath + "/rmseSample", size(rmseSample));
	h5write(outFile, grpPath + "/rmseSample", rmseSample);
	h5create(outFile, grpPath + "/gtPos", size(gtPos));
	h5write(outFile, grpPath + "/gtPos", gtPos);

	fprintf("[%s] MeanStepRMSE=%.6f | StdStepRMSE=%.6f | MeanSampleRMSE=%.6f\n", ...
		noiseLabel, meanStepRMSE, stdStepRMSE, meanSampleRMSE);
end

fprintf("\n=== LLS + DNN1 Summary ===\n");
disp(summaryTable);
fprintf("Saved: %s\n", outFile);


function xLS = llsPositionFromRanging(ranges, anchors)
% Linearized least squares from range measurements.
% ranges: [4, 1], anchors: [2, 4]

ref = 1;
aRef = anchors(:, ref);
rRef = ranges(ref);

A = zeros(3, 2);
b = zeros(3, 1);

row = 1;
for i = 1:4
	if i == ref
		continue;
	end
	ai = anchors(:, i);
	ri = ranges(i);

	A(row, :) = 2 * (ai - aRef)';
	b(row) = rRef^2 - ri^2 - (aRef' * aRef) + (ai' * ai);
	row = row + 1;
end

xLS = A \ b;
end

