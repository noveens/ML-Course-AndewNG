function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
J = 0;

for i=1:size(X, 1),
  temp = X(i, :);
  temp2 = theta' * temp';
  temp2 = temp2 - y(i);
  J = J + ((temp2 ^ 2) / (2*m));
 end

% =========================================================================

end
