function theta = cnnInitParams(imageDim,filterDimC1,numFiltersC1,poolDim,filterDimC2,numFiltersC2,hiddenSize,numClasses)
%% Initialize parameters randomly based on layer sizes.


Wc1 = 1e-1*randn(filterDimC1,filterDimC1,numFiltersC1);
Wc2 = 1e-1*randn(filterDimC2,filterDimC2,numFiltersC2);

outDim = imageDim - filterDimC1 + 1; % dimension of convolved image
outDim = outDim/poolDim;

outDim = outDim-filterDimC2+1;

hiddenOut=numFiltersC1*numFiltersC2*outDim;



% we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(hiddenOut+hiddenSize+1);
Wn = rand(hiddenSize,hiddenOut) * 2 * r - r;
Ws = rand(numClasses, hiddenSize)*2*r-r;


bc1 = zeros(numFiltersC1, 1);
bc2 = zeros(numFiltersC2, 1);
bn = zeros(hiddenSize, 1);


% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [Wc1(:) ; Wc2(:); Wn(:) ; Ws(:); bc1(:) ;bc2(:); bn(:)];

end

