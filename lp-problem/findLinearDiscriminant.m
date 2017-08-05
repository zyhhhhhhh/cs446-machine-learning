% This function finds a linear discriminant using LP
% The linear discriminant is represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [w,theta,delta] = findLinearDiscriminant(data)
%% setup linear program
[m, np1] = size(data);
n = np1-1;

% write your code here
c=[zeros(n+1,1);1];
b=[ones(m,1);0];
y =data(:,np1);
x = data(:,1:n);
yxt = zeros(m,n);
for i = 1:m
    yxt(i,1:n) = y(i)*x(i,1:n)
end
disp(yxt)
A=[yxt data(:,np1) ones(m,1); zeros(1,np1) 1];
disp(A)
%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);

end
