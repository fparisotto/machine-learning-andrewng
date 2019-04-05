function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
all_error = zeros(length(values) .^ 2, 1);
all_C = zeros(length(values) .^ 2, 1);
all_sigma = zeros(length(values) .^ 2, 1);

i = 1;
for C_index = 1:length(values)
  for sigma_index = 1:length(values)    
    _C = values(C_index);
    _sigma = values(sigma_index);
    model = svmTrain(X, y, _C, @(x1, x2) gaussianKernel(x1, x2, _sigma));
    predictions = svmPredict(model, Xval);
    all_error(i) = mean(double(predictions ~= yval));
    all_C(i) = _C;
    all_sigma(i) = _sigma;
    i += 1;
  end
end

[_, min_error_index] = min(all_error);

C = all_C(min_error_index);
sigma = all_sigma(min_error_index);


% =========================================================================

end
