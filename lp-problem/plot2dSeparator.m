% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)

x = linspace(0,1);
y = -theta/w(2)-w(1)/w(2)*x;
plot(x,y);

end
