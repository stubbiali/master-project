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

a = 0;  b = 2*pi;
K = 101;

%mu = -0.8;  sigma = 0.2;  f = @(t) gaussian(t,mu,sigma);
%mu = 0.1;  f = @(t) mu * t .* ((-mu <= t) & (t <= mu));
%mu = 4;  f = @(t) (t-1).^mu;
%f = @(t) 50 * t .* cos(2*pi*t);
%mu = 0.25;  f = @(t) 2 * atan(mu * t/2);
%v = @(t) 1 + 0*t;
%mu = 0.5;  f = @(t) 2*(t >= mu) - 1*(t < mu);

mu = 1;  %nu = 5;
%v = @(t) nu*(t < 0) + (nu + nu*t).*(t >= 0);
%v = @(t) 1*(t < nu) + 4*(t >= nu);
%v = @(t) gaussian(t,0,nu);
%v = @(t) 1 + (t+1).^nu;
%v = @(t) 1*(t < -0.5) + nu*(-0.5 <= t & t <= 0.5) + 1*(t > 0.5);
%f = @(t) - mu*(t < 0) + 2*mu*(t >= 0);
%f = @(t) -gaussian(t,-mu,0.8) + gaussian(t,mu,0.8);
%f = @(t) 2*(t >= mu) - 1*(t < mu);
%f = @(t) gaussian(t,mu,0.1);

v = @(t) exp(t);  dv = @(t) exp(t);
%v = @(t) 1 + 0*t;  dv = @(t) 0*t;
%v = @(t) t.^2;  dv = @(t) 2*t;
%v = @(t) t.^(-2);  dv = @(t) -2*t.^(-3);
%v = @(t) abs(t);  dv = @(t) -(t < 0) + (t >= 0);
%f = @(t) pi*pi*sin(pi*t);
f = @(x) exp(sin(x)) .* (sin(x) - cos(x).^2);

BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;

%
% Run
%

% Solve
%[x,u1] = LinearPoisson1dFEP1(a, b, K, f, BCLt, BCLv, BCRt, BCRv);
[x,u1] = NonLinearPoisson1dFEP1(a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv);

% Plot
%figure;
hold on;
plot(x,u1);
title('Solution to Poisson equation')
xlabel('$x$')
ylabel('$u(x)$')
grid on
%axis equal
xlim([a b])