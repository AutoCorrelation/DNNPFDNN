function [xPre, state] = preprocess_dnn1_predict(net, normalization, xRaw)
%PREPROCESS_DNN1_PREDICT Apply DNN1 preprocessing and return denoised ranging.
%   xRaw: [N, 4] or [4, N] noisy ranging input
%   xPre: [N, 4] denoised ranging output

if isempty(xRaw)
    xPre = xRaw;
    state = struct();
    return;
end

inputWasTransposed = false;
if size(xRaw, 1) == 4 && size(xRaw, 2) ~= 4
    xRaw = xRaw.';
    inputWasTransposed = true;
end

xNorm = (xRaw - normalization.muX) ./ normalization.sigmaX;
yNorm = predict(net, xNorm);
xPre = yNorm .* normalization.sigmaY + normalization.muY;

if inputWasTransposed
    xPre = xPre.';
end

state.inputWasTransposed = inputWasTransposed;
state.inputSize = size(xRaw);
state.outputSize = size(xPre);
end