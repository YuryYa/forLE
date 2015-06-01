function t_points = out_of_sample_est_kernel(points, X, mappedX, sigma)
%TRANSFORM_SAMPLE_EST Performs out-of-sample extension using estimation technique
%
%   t_points = out_of_sample_est(points, X, mappedX)
%
% Performs out-of-sample extension using estimation technique on datapoints
% points. You also need to specify the original dataset in X, and the
% reduced dataset in mappedX (the two datasets may also be PRTools datasets).
% The function returns the coordinates of the transformed points in t_points.

[X, normalizationParameters] = mapstd(X');
X = X';
points = mapstd('apply', points', normalizationParameters)';

t_points = zeros(size(points, 1), size(mappedX, 2));
kernel_function = @(x)(exp(-x.^2 / (2 * sigma^2)));
kernelMatrix = dist(points, X');
kernelMatrix = kernel_function(kernelMatrix);
kernelMatrix = diag(1 ./ sum(kernelMatrix, 2)) * kernelMatrix;
t_points = kernelMatrix * mappedX;
