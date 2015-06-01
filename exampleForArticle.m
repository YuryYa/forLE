rng(0);
trainSizes = [250, 500, 1000, 2000];
testSize = 10000;
surfaceNames = {'cylinder', 'saddle', 'ellipsoid', 'cone'};
distanceToTrain = zeros(testSize, length(trainSizes), length(surfaceNames));
inputDimension = 3;
internalDimension = 2;
method = 'LaplacianEigenmapsAsymmetric';
nns = 10;
sigma = 1.5;
for surfaceIndex = 1%1:length(surfaceNames)
  disp(surfaceNames{surfaceIndex});
  surfaceName = surfaceNames{surfaceIndex};
  [testX, ~, parametrizationTest] = ...
       generateSampleOnSurface(testSize, surfaceName);
%   handle = figure();
%   scatter3(testX(:, 1), testX(:, 2), testX(:, 3), [], parametrizationTest(:, 1), 'filled');
%   saveas(handle, strcat(surfaceName, 'TestSample.png'));
%   close(handle);
  for trainSizeIndex = 1:2%1:length(trainSizes)
    disp(trainSizes(trainSizeIndex));
    trainSize = trainSizes(trainSizeIndex);
    if trainSizeIndex == 1
      [trainX, ~, ~] = ...
        generateSampleOnSurface(trainSize, surfaceName);
    else
      numberNewPoints = trainSize - trainSizes(trainSizeIndex - 1);
      [newTrainX, ~, ~] = ...
        generateSampleOnSurface(trainSize, surfaceName);
      trainX = [trainX; newTrainX];
    end
    % embedding and reconstruction
    [trainT, mapping] = compute_mapping(trainX, method, dim, nns, sigma);
    recTrainX = out_of_sample_est_kernel(trainT, trainT, trainX, sigma);
    trainT2 = out_of_sample_est(trainX, trainX, trainT);
    recXTrain2 = out_of_sample_est_kernel(trainT2, trainT, trainX, sigma);
    testT = out_of_sample_est(testX, trainX, trainT);
    recXTest = out_of_sample_est_kernel(testT, trainT2, trainX, sigma);
  end
end
% figure();
% scatter(trainT(:, 1), trainT(:, 2));
figure();
scatter(trainT2(:, 1), trainT2(:, 2));
figure();
scatter(testT(:, 1), testT(:, 2));
% figure();
% scatter3(testX(:, 1), testX(:, 2), testX(:, 3));
% scatter3(testX(:, 1), testX(:, 2), testX(:, 3));
% scatter3(trainX(:, 1), trainX(:, 2), trainX(:, 3));
% figure();
% scatter3(recXTest(:, 1), recXTest(:, 2), recXTest(:, 3));
% figure();
% scatter3(recXTrain2(:, 1), recXTrain2(:, 2), recXTrain2(:, 3));