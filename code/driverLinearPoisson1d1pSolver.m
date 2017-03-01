% driverLinearPoisson1d1pSolver Driver for solving the linear one-dimensional 
% Poisson equation $-u''(x) = f(x,\mu)$ on $[a,b]$ depending on the real
% parameter $\mu$. Linear or quadratic finite elements can be used.

clc
clear variables
clear variables -global
close all

%
% User-defined settings
%

a = -1;  b = 1;
K = 100;
mu = -0.8;  f = @(t) 1/sqrt(2*pi) * exp(-(t-mu).^2);
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 1;

%
% Run
%

% Solve
[x,u] = LinearPoisson1dFEP1_f(a, b, K, f, BCLt, BCLv, BCRt, 1);
[x,u] = LinearPoisson1dFEP1_f(a, b, K, f, BCLt, BCLv, BCRt, 2);
%[x,u] = LinearPoisson1dFEP2(a, b, K, f, BCLt, BCLv, BCRt, BCRv);

% Plot
figure;
plot(x,u);
title('Solution to Poisson equation')
xlabel('$x$')
ylabel('$u(x)$')
grid on