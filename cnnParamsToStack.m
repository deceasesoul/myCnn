function [Wc1, Wc2, Wn, Ws, bc1,bc2, bn] = cnnParamsToStack(theta,netconfig)

outDim = netconfig.imageDim - netconfig.filterDimC1 + 1; % dimension of convolved image
outDim = outDim/netconfig.poolDim;

outDim = outDim-netconfig.filterDimC2+1;

hiddenOut=netconfig.numFiltersC1*netconfig.numFiltersC2*outDim;

%% Reshape theta
indS = 1;
indE = netconfig.filterDimC1^2*netconfig.numFiltersC1;
Wc1 = reshape(theta(indS:indE),netconfig.filterDimC1,netconfig.filterDimC1,netconfig.numFiltersC1);
indS = indE+1;
indE = indE+netconfig.filterDimC2^2*netconfig.numFiltersC2;
Wc2 = reshape(theta(indS:indE),netconfig.filterDimC2,netconfig.filterDimC2,netconfig.numFiltersC2);

indS = indE+1;
indE = indE+hiddenOut*netconfig.hiddenSize;
Wn = reshape(theta(indS:indE),netconfig.hiddenSize,hiddenOut);

indS = indE+1;
indE = indE+netconfig.hiddenSize*netconfig.numClasses;
Ws = reshape(theta(indS:indE),netconfig.numClasses,netconfig.hiddenSize);

indS = indE+1;
indE = indE+netconfig.numFiltersC1;
bc1= theta(indS:indE);

indS = indE+1;
indE = indE+netconfig.numFiltersC2;
bc2= theta(indS:indE);

indS = indE+1;
indE = indE+netconfig.hiddenSize;
bn= theta(indS:indE);

end