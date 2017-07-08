function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

part1 = sigmoid(X * theta);
part2 = arrayfun(@func, 1 - part1);
part1 = arrayfun(@func, part1);
part1 = y' * part1;
part2 = (1 - y') * part2;
J = (part1 + part2) / -m;
J = J + ((sum(theta(2: length(theta)) .^ 2) * lambda) / (2 * m));

H = sigmoid(X * theta);

grad = (X' * (H - y)) / m;
temp = grad(1);
grad = grad .+ (theta .* (lambda/ m));
grad(1) = temp;

% =============================================================

end

function ret = func(z)
  ret = log(z);
end