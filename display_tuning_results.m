%% Display Q tuning results from saved file

clear; clc;

matFile = "checkpoints/pf_qr_tuning_results.mat";
if ~isfile(matFile)
    error("Results file not found: %s", matFile);
end

load(matFile, "summary", "allTuneResults");

fprintf("=== Q TUNING RESULTS (Variance-Scaled Grid) ===\n\n");
disp(summary);

fprintf("\n=== Detailed Grid Search Results ===\n");
for k = 1:numel(allTuneResults)
    res = allTuneResults(k);
    fprintf("\n[Noise %s] (%d grid points tested)\n", res.noiseLabel, numel(res.qAlphaGrid));
    
    % Display Q grid and corresponding RMSE values
    fprintf("  Q Grid (×10⁻⁶): ");
    qGrid_scaled = res.qAlphaGrid * 1e6;
    fprintf("[");
    for i = 1:min(5, numel(qGrid_scaled))
        fprintf("%.2f", qGrid_scaled(i));
        if i < numel(qGrid_scaled), fprintf(", "); end
    end
    if numel(qGrid_scaled) > 5
        fprintf(", ... %.2f", qGrid_scaled(end));
    end
    fprintf("]\n");
    
    fprintf("  RMSE Values: [");
    for i = 1:min(5, numel(res.gridRMSE))
        fprintf("%.6f", res.gridRMSE(i));
        if i < numel(res.gridRMSE), fprintf(", "); end
    end
    if numel(res.gridRMSE) > 5
        fprintf(", ... %.6f", res.gridRMSE(end));
    end
    fprintf("]\n");
    
    fprintf("  Best Q: %.6g | Best RMSE: %.6f\n", res.bestQAlpha, res.bestRMSE);
end

fprintf("\n=== Summary ===\n");
fprintf("Script executed with variance-scaled Q grids (1000 samples).\n");
fprintf("Results saved to: checkpoints/pf_qr_tuning_results.mat\n");
