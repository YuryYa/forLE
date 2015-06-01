rng(0);
trainSizes = [250, 500, 1000, 2000];
testSize = 10000;
surfaceNames = {'cylinder', 'saddle', 'ellipsoid', 'cone'};
distanceToTrain = zeros(testSize, length(trainSizes), length(surfaceNames));
inputDimension = 3;
internalDimension = 2;
for surfaceIndex = 1:length(surfaceNames)
  disp(surfaceNames{surfaceIndex});
  surfaceName = surfaceNames{surfaceIndex};
  [testX, ~, parametrizationTest] = ...
       generateSampleOnSurface(testSize, surfaceName);
%   handle = figure();
%   scatter3(testX(:, 1), testX(:, 2), testX(:, 3), [], parametrizationTest(:, 1), 'filled');
%   saveas(handle, strcat(surfaceName, 'TestSample.png'));
%   close(handle);
  for trainSizeIndex = 1:length(trainSizes)
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
    % one could try cycle to save memory
    distanceToTrain(:, trainSizeIndex, surfaceIndex) = min(dist(testX, trainX'), [], 2); 
  end
end

%% plot results
handle = figure();
colors = {'r', 'm', 'b', 'g'};
for surfaceIndex = 1:length(surfaceNames)
  subplot(4, 1, surfaceIndex);
  hold on
  for trainSizeIndex = 1:length(trainSizes)
    plot(sort(distanceToTrain(:, trainSizeIndex, surfaceIndex)), colors{trainSizeIndex});
    maxDistance(trainSizeIndex, surfaceIndex) = max(distanceToTrain(:, trainSizeIndex, surfaceIndex));
  end
  title(surfaceNames{surfaceIndex});
end

%% fit regression with 4 points)
XMatrix = [ones(4, 1), log10(trainSizes')];
for surfaceIndex = 1:length(surfaceNames)
  yVector = log10(maxDistance(:, surfaceIndex));
  theta(:, surfaceIndex) = (XMatrix' * XMatrix)^(-1) * XMatrix' * yVector;
  subplot(4, 1, surfaceIndex);
  scatter(log10(trainSizes'), XMatrix * theta(:, surfaceIndex), 'r');
  hold on
  plot(log10(trainSizes'), yVector, 'b');
  title(surfaceNames{surfaceIndex});
end

