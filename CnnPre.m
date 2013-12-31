function pre=CnnPre(images,netconfig,theta)
numImages=size(images,3);
[Wc1, Wc2, Wn, Ws, bc1,bc2, bn] = cnnParamsToStack(theta,netconfig);

convDim1 = netconfig.imageDim-netconfig.filterDimC1+1;

outputDim1 = (convDim1)/netconfig.poolDim; % dimension of subsampled output

activations1= cnnConvolve(netconfig.filterDimC1,netconfig.numFiltersC1,images,Wc1,bc1);

activationsPooled=cnnPool(netconfig.poolDim,activations1);

activations2= cnnConvolve(netconfig.filterDimC2,netconfig.numFiltersC2,reshape(activationsPooled,outputDim1,outputDim1,[]),Wc2,bc2);

aout=reshape(activations2,[],numImages);

z=Wn*aout+repmat(bn,1,numImages);

anlayer=sigmoid(z);

t = Ws*anlayer;     % (numClasses,N)*(N,M)

[~,pred]=max(t,[],1);
pred=pred';
pre=pred-1;
end
function sigm = sigmoid(x)
sigm = 1 ./ (1 + exp(-x));
end