%% 6 Layers CNN   [ C P C Full SoftMax ] For Classification
netconfig.imageDim = 28;
netconfig.numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
netconfig.filterDimC1 = 8;    % Filter size for conv layer
netconfig.numFiltersC1 = 6;   % Number of filters for conv layer
netconfig.poolDim = 3;      % Pooling dimension, (should divide imageDim-filterDim+1)
netconfig.filterDimC2=7;
netconfig.numFiltersC2=6;
netconfig.hiddenSize=200;
% Load MNIST Train
%%
images = loadMNISTImages('train-images.idx3-ubyte');
images = reshape(images,netconfig.imageDim,netconfig.imageDim,[]);
labels = loadMNISTLabels('train-labels.idx1-ubyte');
numImages=size(images,3);
labels=labels+1; 
labels = full(sparse(labels, 1:numImages, 1));
%%
% Initialize Parameters
theta = cnnInitParams(netconfig.imageDim,netconfig.filterDimC1,netconfig.numFiltersC1,netconfig.poolDim,netconfig.filterDimC2,netconfig.numFiltersC2,netconfig.hiddenSize,netconfig.numClasses);



%%======================================================================
%% Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG=true;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
%     db_numFilters = 2;
%     db_filterDim = 9;
%     db_poolDim = 5;
%     db_images = images(:,:,1:10);
%     db_labels = labels(1:10);
%     db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
%                 db_poolDim,numClasses);
    
    [cost grad] = cnnCost(theta,images(:,:,1:10),labels(:,1:10),netconfig);
    

    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,db_images,...
                                db_labels,numClasses,db_filterDim,...
                                db_numFilters,db_poolDim), db_theta);
 
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually 
    % less than 1e-9.
    disp(diff); 
    
end

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the model.



%%======================================================================
%% Test


testImages = loadMNISTImages('t10k-images.idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

[~,cost,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                filterDim,numFilters,poolDim,true);

acc = sum(preds==testLabels)/length(preds);

% Accuracy should be around 97.4% after 3 epochs
fprintf('Accuracy is %f\n',acc);
