function lgraph = dnn2_create_model(lossType)
%DNN2_CREATE_MODEL Create the requested DNN2 regression network.
%   Input : 6 features [pf_x, pf_y, r1, r2, r3, r4]
%   Output: 2 targets  [x, y]

arguments
    lossType (1,1) string = "mse"
end

% Residual MLP pattern aligned with DNN1:
% y = [pos_x, pos_y] + f(x)
lgraph = layerGraph();

mainLayers = [
    featureInputLayer(6, Normalization="none", Name="input")
    fullyConnectedLayer(32, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(64, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(32, Name="fc3")
    reluLayer(Name="relu3")
    fullyConnectedLayer(2, Name="delta")
];

skipLayers = [
    functionLayer(@selectStateXY, Name="skipxy", Formattable=true)
];

mergeLayers = [
    additionLayer(2, Name="skipadd")
];

lgraph = addLayers(lgraph, mainLayers);
lgraph = addLayers(lgraph, skipLayers);
lgraph = addLayers(lgraph, mergeLayers);

lgraph = connectLayers(lgraph, "delta", "skipadd/in1");
lgraph = connectLayers(lgraph, "input", "skipxy");
lgraph = connectLayers(lgraph, "skipxy", "skipadd/in2");

switch lower(lossType)
    case "mse"
        lgraph = addLayers(lgraph, regressionLayer(Name="regression"));
        lgraph = connectLayers(lgraph, "skipadd", "regression");
    case "mae"
        % MATLAB regressionLayer uses MSE by default. For MAE, use a custom
        % loss function in a custom training loop if needed later.
        lgraph = addLayers(lgraph, regressionLayer(Name="regression"));
        lgraph = connectLayers(lgraph, "skipadd", "regression");
    otherwise
        error("Unsupported loss type: %s", lossType);
end
end

function y = selectStateXY(x)
% Select only [pos_x, pos_y] from input [pos_x, pos_y, r1, r2, r3, r4].
y = x(1:2, :);
end
