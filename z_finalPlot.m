noisevar = [0.01 0.1 1 10 100];
rmse = struct(); % 'sturct' -> 'struct' 오타 수정
rmse.LS = [0.0997
    0.3193
    1.0076
    3.252
    12.015
    ];
rmse.LKF = [0.0842
    0.2681
    0.8423
    2.5257
    6.9834
    ];
rmse.LKFdecayQ = [0.0782
    0.249
    0.7908
    2.4368
    6.5487
    ];
rmse.PF = [0.0803
    0.2562
    0.8052
    2.4826
    4.9665
    ];

rmse.DNN_LS = [0.0507
    0.1156
    0.3470
    1.0247
    2.2180
    ];

rmse.DNN_PF = [0.0392
    0.1018
    0.3119
    0.9616
    2.3213
    ];

rmse.DNN_PF_DNN = [0.0398
    0.0507
    0.3453
    0.9199
    1.7112
    ];

rmse.DNN_LS_DNN = [ 0.04469
    0.05944
    0.43969
    1.29814
    2.66879 ];

figure;

% 1. 전통적인 방법(Non-DNN)은 무채색(회색 계열) 및 얇은 선으로 시각적 대비를 낮춤
semilogx(noisevar, rmse.LS, 'o--', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'LS');
hold on;
semilogx(noisevar, rmse.LKF, 's--', 'Color', [0.55 0.55 0.55], 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'LKF');
semilogx(noisevar, rmse.LKFdecayQ, 'd:', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'LKF (decay Q)');
semilogx(noisevar, rmse.PF, '^-', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'PF');

% 2. DNN 기반 방법들은 강렬한 색상, 얇은 선, 안이 채워진 큰 마커로 눈에 띄게 강조
semilogx(noisevar, rmse.DNN_LS, 'v-', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.4470 0.7410], 'DisplayName', 'DNN-LS');
semilogx(noisevar, rmse.DNN_PF, 'p-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'DisplayName', 'DNN-PF');
semilogx(noisevar, rmse.DNN_PF_DNN, 'h-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5, 'MarkerSize', 10, 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'DisplayName', 'DNN-PF-DNN ');
semilogx(noisevar, rmse.DNN_LS_DNN, 'x-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5, 'MarkerSize', 10, 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'DisplayName', 'DNN-LS-DNN ');
hold off;

% 그래프 가독성을 높이기 위한 설정
title('Method Comparison: RMSE vs. Noise Variance', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Noise Variance (\sigma^2)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('RMSE (meters)', 'FontSize', 12, 'FontWeight', 'bold');
legend('show', 'Location', 'northwest', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 11, 'XMinorTick', 'on', 'YMinorTick', 'on');

figure;
semilogx(noisevar, rmse.DNN_LS, 'v-', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', [0 0.4470 0.7410], 'DisplayName', 'DNN-LS');
hold on;
semilogx(noisevar, rmse.DNN_PF, 'p-', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', [0.4660 0.6740 0.1880], 'DisplayName', 'DNN-PF');
semilogx(noisevar, rmse.DNN_PF_DNN, 'h-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5, 'MarkerSize', 10, 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'DisplayName', 'DNN-PF-DNN (Proposed)');
semilogx(noisevar, rmse.DNN_LS_DNN, 'x-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5, 'MarkerSize', 10, 'MarkerFaceColor', [0.8500 0.3250 0.0980], 'DisplayName', 'DNN-LS-DNN ');
hold off;
legend();
grid on;