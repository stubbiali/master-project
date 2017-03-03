% driverLinearPoisson1d1pSolver Driver for solving the linear one-dimensional 
% Poisson equation $-u''(x) = f(x,\mu)$ on $[a,b]$ depending on the real
% parameter $\mu$. Linear or quadratic finite elements can be used.

clc
clear variables
clear variables -global
%close all

%
% User-defined settings
%

a = -1;  b = 1;
K = 100;
%mu = -0.8;  sigma = 0.2;  f = @(t) gaussian(t,mu,sigma);
%mu = 0.1;  f = @(t) mu * t .* ((-mu <= t) & (t <= mu));
%mu = 4;  f = @(t) (t-1).^mu;
%f = @(t) 50 * t .* cos(2*pi*t);
%mu = 0.25;  f = @(t) 2 * atan(mu * t/2);
mu = 0.5897;  nu = 0.8068;  f = @(t) 2*(t >= mu) - 1*(t < mu);
alpha = 1;
beta = 0;
fex = @(t) t.*cos(t) - 2*sin(t) + alpha*t + beta;
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = nu;

%
% Run
%

% Solve
[x,u] = LinearPoisson1dFEP1(a, b, K, f, BCLt, BCLv, BCRt, BCRv);

% Plot
%figure;
hold on;
plot(x,u);
title('Solution to Poisson equation')
xlabel('$x$')
ylabel('$u(x)$')
grid on
%axis equal
xlim([a b])