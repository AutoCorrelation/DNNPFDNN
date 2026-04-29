%% Extract best Q values from tuning results

clear; clc;

% Load tuning results
load("checkpoints/pf_qr_tuning_results.mat", "summary", "allTuneResults");

fprintf("=== EXTRACTED BEST Q VALUES ===\n\n");
fprintf("Noise Label | Best Q Alpha | Best Q Value (diagonal)\n");
fprintf("%-12s %-15s %-30s\n", "----------", "------------", "------------------------------");

% Extract and display best Q values per noise
bestQValues = zeros(1, height(summary));
noiseLabels = summary.Noise;

for k = 1:height(summary)
    bestQAlpha = summary.BestQAlpha(k);
    bestQValues(k) = bestQAlpha;
    fprintf("%-12s %-15.6g %-30.6g\n", noiseLabels(k), bestQAlpha, bestQAlpha);
end

fprintf("\n");
fprintf("For use in filter_pf_cv.m getNoiseConfig():\n");
fprintf("bestQDiag = [%.6g, %.6g, %.6g, %.6g, %.6g];\n", bestQValues(1), bestQValues(2), bestQValues(3), bestQValues(4), bestQValues(5));

fprintf("\nFull Summary Table:\n");
disp(summary);
