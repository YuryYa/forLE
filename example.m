%
fun = @(u, v)[u.*cos(u), v, u.*sin(u)]; % swissroll
minT = [3*pi/2, 0];
rangeT = [3*pi, 20 * pi];
dim = 2;


%fun = @(t, h)[(-1).^(t>0).*t, h, t.*(t>0)]; % curved plane
%minT = [-1, -1];
%rangeT = [2, 2];
%dim = 2;

 %fun = @(t)[t.*cos(t),t.*sin(t), t];
 %minT = [3*pi/2];
 %rangeT = [3*pi];
 %dim = 1;

method = 'LaplacianEigenmapsAsymmetric';

trainSize = 1000;
testSize = 10000;
nns = 10;
sigma = 2;

seed = 0;

% get sample
rng(seed)
trainT = rand(trainSize, dim);
testT = rand(testSize, dim);
[col, idx] = sort(trainT(:,1));
trainT = trainT(idx, :) .* repmat(rangeT, trainSize, 1) + repmat(minT, trainSize, 1);
[colTest, idx] = sort(testT(:,1));
testT = testT(idx, :) .* repmat(rangeT, testSize, 1) + repmat(minT, testSize, 1);
    
trainX = fun(trainT(:,1),trainT(:,2));
testX = fun(testT(:,1),testT(:,2));

% embedding and reconstruction
[trainT, mapping] = compute_mapping(trainX, method, dim, nns, sigma);
recTrainX = out_of_sample_est_kernel(trainT, trainT, trainX, sigma);
trainT2 = out_of_sample_est(trainX, trainX, trainT);
recXTrain2 = out_of_sample_est_kernel(trainT2, trainT, trainX, sigma);
testT = out_of_sample_est(testX, trainX, trainT);
recXTest = out_of_sample_est_kernel(testT, trainT2, trainX, sigma);

%
figure();
scatter(trainT(:, 1), trainT(:, 2));
figure();
scatter(trainT2(:, 1), trainT2(:, 2));
figure();
scatter(testT(:, 1), testT(:, 2));
% scatter3(testX(:, 1), testX(:, 2), testX(:, 3));
% scatter3(testX(:, 1), testX(:, 2), testX(:, 3));
% scatter3(trainX(:, 1), trainX(:, 2), trainX(:, 3));
% scatter3(recXTest(:, 1), recXTest(:, 2), recXTest(:, 3));
% scatter3(recXTrain2(:, 1), recXTrain2(:, 2), recXTrain2(:, 3));