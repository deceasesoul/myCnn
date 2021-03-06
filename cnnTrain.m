%% 6 Layers CNN   [ C S C Full SoftMax ] For Classification
netconfig.imageDim = 28;
netconfig.numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
netconfig.filterDimC1 = 8;    % Filter size for conv layer
netconfig.numFiltersC1 = 8;   % Number of filters for conv layer
netconfig.poolDim = 3;      % Pooling dimension, (should divide imageDim-filterDim+1)
netconfig.filterDimC2=7;
netconfig.numFiltersC2=20;
netconfig.hiddenSize=200;
% Load MNIST Train
%%
images = loadMNISTImages('train-images.idx3-ubyte');
images = reshape(images,netconfig.imageDim,netconfig.imageDim,[]);
labels = loadMNISTLabels('train-labels.idx1-ubyte');
% numImages=size(images,3);
% labels=labels+1;
% labels = full(sparse(labels, 1:numImages, 1));


%%
% Initialize Parameters
theta = cnnInitParams(netconfig.imageDim,netconfig.filterDimC1,netconfig.numFiltersC1,netconfig.poolDim,netconfig.filterDimC2,netconfig.numFiltersC2,netconfig.hiddenSize,netconfig.numClasses);



%%======================================================================
%% Gradient Check
%  Use the file computeNumericalGradient.m to check the gradient
%  calculation for your cnnCost.m function.  You may need to add the
%  appropriate path or copy the file to this directory.

DEBUG=false;  % set this to true to check gradient
if DEBUG
    % To speed up gradient checking, we will use a reduced network and
    % a debugging data set
    db.imageDim = 28;
    db.numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
    db.filterDimC1 = 8;    % Filter size for conv layer
    db.numFiltersC1 = 2;   % Number of filters for conv layer
    db.poolDim = 3;      % Pooling dimension, (should divide imageDim-filterDim+1)
    db.filterDimC2=7;
    db.numFiltersC2=3;
    db.hiddenSize=5;
    %     db_numFilters = 2;
    %     db_filterDim = 9;
    %     db_poolDim = 5;
    %     db_images = images(:,:,1:10);
    %     db_labels = labels(1:10);
    %     db_theta = cnnInitParams(imageDim,db_filterDim,db_numFilters,...
    %                 db_poolDim,numClasses);
    db_theta=cnnInitParams(db.imageDim,db.filterDimC1,db.numFiltersC1,db.poolDim,db.filterDimC2,db.numFiltersC2,db.hiddenSize,db.numClasses);
    [cost grad] = cnnCost(db_theta,images(:,:,1:10),labels(:,1:10),db);
    
    
    % Check gradients
    numGrad = computeNumericalGradient( @(x) cnnCost(x,images(:,:,1:10),labels(:,1:10),db), db_theta);
    
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually
    % less than 1e-9.
    disp(diff);
    
end

%%======================================================================
%% STEP 3: Learn Parameters
%  Implement minFuncSGD.m, then train the mode
batchSize=128;
alpha=0;
trainSize=50000;

trainData=images(:,:,1:trainSize);
trainLabel=labels(1:trainSize);
trainLabel=trainLabel+1;
trainLabel = full(sparse(trainLabel, 1:trainSize, 1));

validationSize=10000;

validationData=images(:,:,trainSize+1:end);
validationLabel=labels(trainSize+1:end);

hold on;
iter=0;
pre_iter=iter;
pre_acc=0;
plot(iter,1,'.');
acc=0;
while(true)
    step=1-acc;
    validationIndex=randi(validationSize,1,256);
    for i=1:20
        %% COST FUNCTION
        trainIndex=randi(trainSize,1,batchSize);
        alpha=rand*step;
        [cost, grad] = cnnCost(theta,trainData(:,:,trainIndex),trainLabel(:,trainIndex),netconfig);        
        theta=theta-alpha*grad;
        %         [cost, grad]=DCnnCostSoftMax_SGD_FixedMask(theta, netconfig,traindata,trainlabel,lamda,maskset2,batchSize);
        %         theta=theta-alpha*step*grad;
    end
%     step=step*0.9996;
    set(gca,'ygrid','on');
    pre=CnnPre(validationData(:,:,validationIndex),netconfig,theta);
    acc = mean(validationLabel(validationIndex) == pre(:));
    
    fprintf('acc: %0.3f%%\n', acc * 100);
    
    iter=iter+1;
    %     plot(iter,acc,'x');
    plot([pre_iter,iter],[pre_acc,acc],'-r');
    drawnow;
    pre_iter=iter;
    pre_acc=acc;
end


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
