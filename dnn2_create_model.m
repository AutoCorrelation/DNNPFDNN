function lgraph = dnn2_create_model()
%DNN2_CREATE_MODEL Create the requested DNN2 regression network.
%   Input : 6 features [pf_x, pf_y, r1, r2, r3, r4]
%   Output: 2 targets  [x, y]

%DNN2_CREATE_MODEL_NOSKIP Create DNN2 regression network without skip.
%   Input : 6 features [pos_x, pos_y, r1, r2, r3, r4]
%   Output: 2 targets  [x, y]

layers = [
    featureInputLayer(6, Normalization="none", Name="input")
    fullyConnectedLayer(16, Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(8, Name="fc2")
    reluLayer(Name="relu2")
    fullyConnectedLayer(4, Name="fc3")
    reluLayer(Name="relu3")
    fullyConnectedLayer(2, Name="out")
    regressionLayer(Name="regression")
];

lgraph = layerGraph(layers);
end
