function [cost, grad] = cnnCost(theta,images,labels,netconfig)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc1, Wc2, Wn, Ws, bc1,bc2, bn] = cnnParamsToStack(theta,netconfig);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc1_grad = zeros(size(Wc1));
Wc2_grad = zeros(size(Wc2));
Wn_grad = zeros(size(Wn));
Ws_grad = zeros(size(Ws));
bc1_grad = zeros(size(bc1));
bc2_grad = zeros(size(bc2));
bn_grad = zeros(size(bn));

%%======================================================================


%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim1 = netconfig.imageDim-netconfig.filterDimC1+1;

outputDim1 = (convDim1)/netconfig.poolDim; % dimension of subsampled output
convDim2 = outputDim1-netconfig.filterDimC2+1;


%%% YOUR CODE HERE %%%

activations1= cnnConvolve(netconfig.filterDimC1,netconfig.numFiltersC1,images,Wc1,bc1);

activationsPooled=cnnPool(netconfig.poolDim,activations1);

activations2= cnnConvolve(netconfig.filterDimC2,netconfig.numFiltersC2,reshape(activationsPooled,outputDim1,outputDim1,[]),Wc2,bc2);
%%  NN Layer
aout=reshape(activations2,[],numImages);
z=Wn*aout+repmat(bn,1,numImages);
anlayer=sigmoid(z);



%% Softmax Layer

M = Ws*anlayer;     % (numClasses,N)*(N,M)
M = bsxfun(@minus, M, max(M, [], 1));
h = exp(M);
probs =  bsxfun(@rdivide, h, sum(h));


%%======================================================================
%% Calculate Cost
cost = -1/numImages*sum(sum(labels.*log(probs)));
%%======================================================================
%%  Backpropagation

Ws_grad = -1/numImages*((labels-probs)*anlayer');

deltan=-Ws' * (labels - probs) .*anlayer .* (1-anlayer);
deltaC2=zeros(size(activations2));
%activations2=reshape(activations2,[],numImages);

deltaC1=zeros(size(activations1));
tempdelta2=reshape(Wn'*deltan,1,1,6,60);

for i=1:numImages    
    for j=1:netconfig.numFiltersC2
        deltaC2(:,:,j,i)=tempdelta2(:,:,j,i).*activations2(:,:,j,i).*(1-activations2(:,:,j,i));    
    end
end
dc2temp=reshape(deltaC2,[],numImages);
for i=1:numImages
    for j=1:netconfig.numFiltersC1
%        (1/poolDim^2) * kron(delta(:,:,j,i),ones(poolDim)).*activations(:,:,j,i).*(1-activations(:,:,j,i));
        deltatt=Wc2(:,:,j)*dc2temp(j,i);
        deltaC1(:,:,j,i)=   (1/netconfig.poolDim^2) * kron(deltatt(:,:,j,i),ones(netconfig.poolDim)) .*activations1(:,:,j,i).*(1-activations1(:,:,j,i));
    end
end



%%======================================================================
%% STEP 1d: Gradient Calculation

wc1dgradtemp=zeros(size(Wc1));
wc2dgradtemp=zeros(size(Wc2));
b1dgradtemp=zeros(numFilters,1);
b2dgradtemp=zeros(size(bc2));
b1dgradtemp=zeros(size(bc1));



for i=1:numImages    
    for j=1:netconfig.numFiltersC1
        deltak=rot90(deltaC1(:,:,j,i),2);
        wc1dgradtemp(:,:,j)=wc1dgradtemp(:,:,j)+conv2(images(:,:,i),deltak,'valid');
        b1dgradtemp(j,1)=b1dgradtemp(j,1)+sum(sum(deltaC1(:,:,j,i)));
    end
end
for i=1:numImages    
    for j=1:netconfig.numFiltersC2
        deltak=rot90(deltaC2(:,:,j,i),2);
        wc2dgradtemp(:,:,j)=wc2dgradtemp(:,:,j)+conv2(images(:,:,i),deltak,'valid');
        b2dgradtemp(j,1)=b2dgradtemp(j,1)+sum(sum(deltaC2(:,:,j,i)));
    end
end


Wc1_grad=1/numImages*wc1dgradtemp;
Wc2_grad=1/numImages*wc2dgradtemp;
bc_grad=1/numImages*bdgradtemp;
%% Unroll gradient into grad vector for minFunc
grad = [Wc1_grad(:) ;Wc2_grad(:); Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end
