%% Final PF Evaluation with Tuned Q Values
% Run full PF with optimized Q values on all 10,000 samples per noise variance

clear; clc; rng(42);

fprintf("=== FINAL PF EVALUATION (Tuned Q Values) ===\n");
fprintf("Starting full evaluation on 10k samples per variance...\n");

if isfile("checkpoints/pf_qr_tuning_results.mat")
    tuningData = load("checkpoints/pf_qr_tuning_results.mat", "summary");
    if isfield(tuningData, "summary") && ~isempty(tuningData.summary)
        fprintf("Using tuned Q values from checkpoints/pf_qr_tuning_results.mat:\n");
        disp(tuningData.summary(:, ["Noise", "BestQAlpha"]));
    end
else
    fprintf("Tuning summary not found. filter_pf_cv.m will fall back to built-in Q values.\n");
end

fprintf("\n");

tic;

% Run optimized PF with tuned Q values
filter_pf_cv;

elapsed = toc;
fprintf("\n=== EVALUATION COMPLETE ===\n");
fprintf("Total Time: %.1f minutes\n", elapsed / 60);

% Load and compare results
fprintf("\nLoading results...\n");
if isfile("checkpoints/pf_results_cv.h5")
    firstPf = h5read("checkpoints/pf_results_cv.h5", "/" + noiseLabels(1) + "/rmseAll");
    pf_rmse_per_sample = zeros(numel(firstPf), numel(noiseLabels));
    pf_rmse_per_sample(:, 1) = firstPf;
    for k = 1:numel(noiseLabels)
        if k == 1
            continue;
        end
        pf_rmse_per_sample(:, k) = h5read("checkpoints/pf_results_cv.h5", "/" + noiseLabels(k) + "/rmseAll");
    end
    fprintf("PF Results loaded.\n");
end

if isfile("checkpoints/lls_dnn1_results.h5")
    firstLls = h5read("checkpoints/lls_dnn1_results.h5", "/" + noiseLabels(1) + "/rmseSample");
    lls_rmse_per_sample = zeros(numel(firstLls), numel(noiseLabels));
    lls_rmse_per_sample(:, 1) = firstLls;
    for k = 1:numel(noiseLabels)
        if k == 1
            continue;
        end
        lls_rmse_per_sample(:, k) = h5read("checkpoints/lls_dnn1_results.h5", "/" + noiseLabels(k) + "/rmseSample");
    end
    fprintf("LS Results loaded.\n");
end

% Compare performance
fprintf("\n");
fprintf("=== PERFORMANCE COMPARISON ===\n");
fprintf("%-8s %-15s %-15s %-12s\n", "Noise", "PF RMSE", "LS RMSE", "Improvement");
fprintf("%-8s %-15s %-15s %-12s\n", "-------", "-------", "-------", "-----------");

noiseLabels = ["001", "01", "1", "10", "100"];
for k = 1:numel(noiseLabels)
    pf_mean = mean(pf_rmse_per_sample(:, k), 'omitnan');
    lls_mean = mean(lls_rmse_per_sample(:, k), 'omitnan');
    improvement = (lls_mean - pf_mean) / lls_mean * 100;
    
    if pf_mean < lls_mean
        fprintf("%-8s %-15.6f %-15.6f ✓ %.1f%%\n", noiseLabels(k), pf_mean, lls_mean, improvement);
    else
        fprintf("%-8s %-15.6f %-15.6f ✗ -%.1f%%\n", noiseLabels(k), pf_mean, lls_mean, abs(improvement));
    end
end

fprintf("\n=== SUMMARY ===\n");
pf_overall = mean(pf_rmse_per_sample, 'all', 'omitnan');
lls_overall = mean(lls_rmse_per_sample, 'all', 'omitnan');
overall_improvement = (lls_overall - pf_overall) / lls_overall * 100;
fprintf("Overall PF RMSE: %.6f\n", pf_overall);
fprintf("Overall LS RMSE: %.6f\n", lls_overall);
if pf_overall < lls_overall
    fprintf("✓ PF outperforms LS by %.1f%%\n", overall_improvement);
else
    fprintf("✗ LS still better by %.1f%%\n", abs(overall_improvement));
end

fprintf("\nResults saved to:\n");
fprintf("  - checkpoints/pf_results_cv.h5 (PF with tuned Q)\n");
fprintf("  - checkpoints/lls_dnn1_results.h5 (LS baseline)\n");
