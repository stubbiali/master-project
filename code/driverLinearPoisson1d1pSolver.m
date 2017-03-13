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

%mu = -0.8;  sigma = 0.2;  f = @(t) gaussian(t,mu,sigma);
%mu = 0.1;  f = @(t) mu * t .* ((-mu <= t) & (t <= mu));
%mu = 4;  f = @(t) (t-1).^mu;
%f = @(t) 50 * t .* cos(2*pi*t);
%mu = 0.25;  f = @(t) 2 * atan(mu * t/2);
%v = @(t) 1 + 0*t;
%mu = 0.5;  f = @(t) 2*(t >= mu) - 1*(t < mu);

%v = @(t) nu*(t < 0) + (nu + nu*t).*(t >= 0);
%v = @(t) 1*(t < nu) + 4*(t >= nu);
%v = @(t) gaussian(t,0,nu);
%v = @(t) 1 + (t+1).^nu;
%v = @(t) 1*(t < -0.5) + nu*(-0.5 <= t & t <= 0.5) + 1*(t > 0.5);
%f = @(t) - mu*(t < 0) + 2*mu*(t >= 0);
%f = @(t) -gaussian(t,-mu,0.8) + gaussian(t,mu,0.8);
%f = @(t) 2*(t >= mu) - 1*(t < mu);
%f = @(t) gaussian(t,mu,0.1);

%v = @(t) 1 + 0*t;  dv = @(t) 0*t;
%v = @(t) t.^2;  dv = @(t) 2*t;
%v = @(t) t.^(-2);  dv = @(t) -2*t.^(-3);
%v = @(t) abs(t);  dv = @(t) -(t < 0) + (t >= 0);
%f = @(t) pi*pi*sin(pi*t);

%a = 0;  b = 2*pi;
%v = @(t) exp(t);  dv = @(t) exp(t);
%f = @(x) exp(sin(x)) .* (sin(x) - cos(x).^2);

%a = 0;  b= 2*pi;
%v = @(t) log(t);  dv = @(t) 1./t;
%f = @(t) -(cos(t).^2)./(2+sin(t)) + log(2+sin(t)).*sin(t);

%a = -1;  b = 1;
%v = @(t) t.^(-2);  dv = @(t) -2*(t.^(-3));
%f = @(t) exp(t.^2) .* (2+4*t.^2);
%BCLt = 'D';  BCLv = exp(-1);
%BCRt = 'P';  BCRv = 0;

a = -1;  b = 1;  K = 100;
v = @(u) 1./(u.^2);  dv = @(u) - 2./(u.^3);
%mu = 0.5;  nu = 1;  f = @(t) -gaussian(t,mu,0.1) + nu*gaussian(t,-mu,0.1);
mu = 2;  f = @(t) 1;
BCLt = 'D';  BCLv = -1;
BCRt = 'D';  BCRv = -1;

%load(['../datasets/NonLinearPoisson1d1pSVD/NonLinearPoisson1d1p_FEP1Newton_' ...
%    'SVDunif_a-1.00_b1.00_D1.00_D1.00_mu1-1.00_mu21.00_K100_N50_L25_Nte100.mat']);

%
% Run
%

% Solve
%[x,u1] = LinearPoisson1dFEP1(a, b, K, f, BCLt, BCLv, BCRt, BCRv);
[x,u] = NonLinearPoisson1dFEP1Newton(a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv);
%[x,alpha] = NonLinearPoisson1dFEP1ReducedNewton(a, b, K, v, dv, f, BCLt, BCLv, ...
%    BCRt, BCRv, VL);
%ur = VL*alpha;

% Plot
%figure;
hold on;
plot(x,u);
%plot(x,ur);
title('Solution to Poisson equation')
xlabel('$x$')
ylabel('$u(x)$')
grid on
%axis equal
xlim([a b])
%legend('Full','Reduced')