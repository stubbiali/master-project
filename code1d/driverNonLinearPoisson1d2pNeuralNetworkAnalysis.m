% driverNonLinearPoisson1d2pNeuralNetworkAnalysis 
% Consider the one-dimensional nonlinear Poisson equation 
% $-(v(u) u'(x))' = f(x,\mu,\nu)$ on $[a,b]$ in the unknown $u = u(x)$, with
% $\mu \in [\mu_1,\mu_2]$ and $\nu \in [\nu_1,\nu_2]$ real parameters. 
% Given a reduced basis $U^l$ of rank $l$ and the associated reduced 
% solution $\boldsymbol{\alpha} \in \mathbb{R}^l$, this script aims at 
% analyzing the results for the approximation of the map
% $[\mu,\nu]^T \mapto \boldsymbol{\alpha}$ through a Neural Network.

clc
clear variables
clear variables -global
close all

%
% User-defined settings:
% a         left boundary of the domain
% b         right boundary of the domain
% K         number of grid points
% v         viscosity $v = v(u)$ as handle function
% dv        derivative of the viscosity as handle function
% f         force field $f = f(x,\mu,\nu)$ as handle function
% mu1       lower bound for $\mu$
% mu2       upper bound for $\mu$
% nu1       lower bound for $\nu$
% nu2       upper bound for $\nu$
% suffix    suffix for data file name
% Nmu       number of different samples for $\mu$ used for computing the
%           snapshots
% Nnu       number of different samples for $\nu$ used for computing the
%           snapshots
% BCLt      kind of left boundary condition
%           - 'D': Dirichlet, 
%           - 'N': Neumann, 
%           - 'P': periodic
% BCLv      left boundary condition as handle function in mu and nu
% BCRt      kind of right boundary condition
%           - 'D': Dirichlet, 
%           - 'N': Neumann, 
%           - 'P': periodic
% BCRv      right boundary condition as handle function in mu and nu
% solver    solver
%           - 'FEP1': linear finite elements; the resulting nonlinear
%           system is solved through the built-in Matlab function fsolve()
%           - 'FEP1Newton': linear finite elements; the resulting nonlinear
%           syste, is solved through Newton's method
% reducer   method to compute the reduced basis
%           - 'SVD': Single Value Decomposition, i.e. Proper
%					 Orthogonal Decomposition (POD)
% sampler   how the values for $\mu$ and $\nu$ should have been picked for
%           the computation of the snapshots in the reduction step:
%           - 'unif': sampled values form a Cartesian grid
%           - 'rand': sampled values drawn from a uniform random distribution
% L         rank of reduced basis
% Nte_r     number of testing samples stored in the dataset for reduction       
% root      path to folder where storing the output dataset

a = -pi;  b = pi;  K = 100;
v = @(u) u.^2;  dv = @(u) 2*u;
f = @(t,mu,nu) nu.*nu.*mu.*mu.*(2+sin(mu.*t)).*(-2*nu.*cos(mu.*t).^2 + ...
    2*nu.*sin(mu.*t) + nu.*sin(mu.*t).^2);
mu1 = 1;  mu2 = 3;  nu1 = 1;  nu2 = 3;  suffix = '';
BCLt = 'D';  BCLv = @(mu,nu) nu.*(2+sin(mu*a));
BCRt = 'D';  BCRv = @(mu,nu) nu.*(2+sin(mu*b));
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
Nmu = 25;  Nnu = 25;  L = 8;  Nte_r = 100;
root = '../datasets';

% Total number of sampled points for computing the reduced basis
N = Nmu*Nnu;

%% Plot reduced solution as given by direct method and Neural Network for 
% three values of $\mu$ and $\nu$. Both uniform and random sampling for
% training patterns are tested.

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$
% Nnu_tr        number of training values for $\nu$
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network
% h_opt         number of hidden neurons
% train_opt     training algorithm:
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization 

Nmu_tr = 20;  Nnu_tr = 20;  valPercentage = 0.3;  Nte_nn = 200;
h_opt = 19;  train_opt = 'trainlm';

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @NonLinearPoisson1dFEP1;
    rsolverFcn = @NonLinearPoisson1dFEP1Reduced;
elseif strcmp(solver,'FEP1Newton')
    solverFcn = @NonLinearPoisson1dFEP1Newton;
    rsolverFcn = @NonLinearPoisson1dFEP1ReducedNewton;
end

% Get total number of training and validating patterns
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Load data for uniform sampling
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Load dataset storing reduced basis
load(datafile);

% Select three values for $\mu$ and $\nu$
%mu = mu1 + (mu2 - mu1) * rand(3,1);
%nu = nu1 + (nu2 - nu1) * rand(3,1);
mu = mu_te([5 50 43]);  nu = nu_te([5 50 43]);

% Evaluate forcing term for the just set values for $\mu$
g = cell(3,1);
for i = 1:3
    g{i} = @(t) f(t,mu(i),nu(i));
end

% Compute reduced solution through direct method
[x, alpha1] = rsolverFcn(a, b, K, v, dv, g{1}, BCLt, BCLv(mu(1),nu(1)), ...
    BCRt, BCRv(mu(1),nu(1)), VL);
ur1 = VL * alpha1;
[x, alpha2] = rsolverFcn(a, b, K, v, dv, g{2}, BCLt, BCLv(mu(2),nu(2)), ...
    BCRt, BCRv(mu(2),nu(2)), VL);
ur2 = VL * alpha2;
[x, alpha3] = rsolverFcn(a, b, K, v, dv, g{3}, BCLt, BCLv(mu(3),nu(3)), ...
    BCRt, BCRv(mu(3),nu(3)), VL);
ur3 = VL * alpha3;

% Find position in error metrix associated with the desired
% network topology and training algorithm
row = find(H == h_opt);
col = 0;
for k = 1:length(trainFcn)
    if strcmp(trainFcn{k},train_opt)
        col = k;
    end
end
if isempty(row) || (col == 0)
    error('Specified number of hidden layers and/or training algorithm not found.')
end

% Extract desired Neural Network and compute reduced solution through it
net = net_opt_local{row,col};
alpha1_u = net([mu(1); nu(1)]);  ur1_u = VL*alpha1_u;
alpha2_u = net([mu(2); nu(2)]);  ur2_u = VL*alpha2_u;
alpha3_u = net([mu(3); nu(3)]);  ur3_u = VL*alpha3_u;

%{
% Load data and get reduced solutions for random sampling
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NNrand_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, mu1, mu2, nu1, nu2, ...
    K, Nmu, Nnu, N, L, Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);
net = net_opt_local{row_opt,col_opt};
alpha1_r = net([mu(1); nu(1)]);  ur1_r = VL*alpha1_r;
alpha2_r = net([mu(2); nu(2)]);  ur2_r = VL*alpha2_r;
alpha3_r = net([mu(3); nu(3)]);  ur3_r = VL*alpha3_r;

%
% Plot distribution of sampling values for $\mu$ and $\nu$ when drawn from 
% a uniform distribution
%

% Open a new window
figure(1);

% Plot distribution for $\mu$
bin = 20;
subplot(1,2,1);
hold off
histogram(mu_tr,bin);
hold on
plot([mu1 mu2], Nmu/bin * [1 1], 'g')
plot(mu, zeros(size(mu)), 'rx', 'Markersize', 10);
title('Distribution of $\mu$')
xlabel('$\mu$')
legend('Random sampling', 'Uniform sampling', 'Test values', 'location', 'best')
grid on
xlim([mu1 mu2])

% Plot distribution for $\nu$
bin = 20;
subplot(1,2,2);
hold off
histogram(nu_tr,bin);
hold on
plot([nu1 nu2], Nnu/bin * [1 1], 'g')
plot(nu, zeros(size(nu)), 'rx', 'Markersize', 10);
title('Distribution of $\nu$')
xlabel('$\nu$')
legend('Random sampling', 'Uniform sampling', 'Test values', 'location', 'best')
grid on
xlim([nu1 nu2])
%}

%
% Compare solutions for three values of $\mu$
%

% Open a new window
figure(2);
hold off

% Plot and set the legend
plot(x(1:1:end), ur1(1:1:end), 'b')
hold on
plot(x(1:1:end), ur1_u(1:1:end), 'b--', 'Linewidth', 2)
%plot(x(1:1:end), ur1_r(1:1:end), 'b:', 'Linewidth', 2)
plot(x(1:1:end), ur2(1:1:end), 'r')
plot(x(1:1:end), ur2_u(1:1:end), 'r--', 'Linewidth', 2)
%plot(x(1:1:end), ur2_r(1:1:end), 'r:', 'Linewidth', 2)
plot(x(1:1:end), ur3(1:1:end), 'g')
plot(x(1:1:end), ur3_u(1:1:end), 'g--', 'Linewidth', 2)
%plot(x(1:1:end), ur3_r(1:1:end), 'g:', 'Linewidth', 2)

% Define plot settings
str = sprintf('Reduced solution to Poisson equation ($k = %i$, $n_{tr} = %i$)', ...
    K, Ntr);
title(str)
xlabel('$x$')
ylabel('$u$')
legend(sprintf('$\\mu = %f$, $\\nu = %f$, reduced', mu(1), nu(1)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN', mu(1), nu(1)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(1), nu(1)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, reduced', mu(2), nu(2)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN', mu(2), nu(2)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(2), nu(2)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, reduced', mu(3), nu(3)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN', mu(3), nu(3)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(3), nu(3)), ...
    'location', 'best')
grid on

%% Plot full solution and reduced solution given by Neural Network for 
% three values of $\mu$ and $\nu$. Both uniform and random sampling for
% training patterns are tested.

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$
% Nnu_tr        number of training values for $\nu$
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network
% h_opt         number of hidden neurons
% train_opt     training algorithm:
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization 

Nmu_tr = 20;  Nnu_tr = 20;  valPercentage = 0.3;  Nte_nn = 200;
h_opt = 15;  train_opt = 'trainlm';

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @NonLinearPoisson1dFEP1;
    rsolverFcn = @NonLinearPoisson1dFEP1Reduced;
elseif strcmp(solver,'FEP1Newton')
    solverFcn = @NonLinearPoisson1dFEP1Newton;
    rsolverFcn = @NonLinearPoisson1dFEP1ReducedNewton;
end

% Get total number of training and validating patterns
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Load data for uniform sampling
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Load dataset storing reduced basis
load(datafile);

% Select three values for $\mu$ and $\nu$
%mu = mu1 + (mu2 - mu1) * rand(3,1);
%nu = nu1 + (nu2 - nu1) * rand(3,1);
mu = mu_te([5 50 43]);  nu = nu_te([5 50 43]);

% Evaluate forcing term for the just set values for $\mu$
g = cell(3,1);
for i = 1:3
    g{i} = @(t) f(t,mu(i),nu(i));
end

% Compute full solution
[x, ur1] = solverFcn(a, b, K, v, dv, g{1}, BCLt, BCLv(mu(1),nu(1)), ...
    BCRt, BCRv(mu(1),nu(1)));
[x, ur2] = solverFcn(a, b, K, v, dv, g{2}, BCLt, BCLv(mu(2),nu(2)), ...
    BCRt, BCRv(mu(2),nu(2)));
[x, ur3] = solverFcn(a, b, K, v, dv, g{3}, BCLt, BCLv(mu(3),nu(3)), ...
    BCRt, BCRv(mu(3),nu(3)));

% Find position in error metrix associated with the desired
% network topology and training algorithm
row = find(H == h_opt);
col = 0;
for k = 1:length(trainFcn)
    if strcmp(trainFcn{k},train_opt)
        col = k;
    end
end
if isempty(row) || (col == 0)
    error('Specified number of hidden layers and/or training algorithm not found.')
end

% Extract desired Neural Network and compute reduced solution through it
net = net_opt_local{row,col};
alpha1_u = net([mu(1); nu(1)]);  ur1_u = VL*alpha1_u;
alpha2_u = net([mu(2); nu(2)]);  ur2_u = VL*alpha2_u;
alpha3_u = net([mu(3); nu(3)]);  ur3_u = VL*alpha3_u;

%{
% Load data and get reduced solutions for random sampling
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NNrand_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, mu1, mu2, nu1, nu2, ...
    K, Nmu, Nnu, N, L, Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);
net = net_opt_local{row_opt,col_opt};
alpha1_r = net([mu(1); nu(1)]);  ur1_r = VL*alpha1_r;
alpha2_r = net([mu(2); nu(2)]);  ur2_r = VL*alpha2_r;
alpha3_r = net([mu(3); nu(3)]);  ur3_r = VL*alpha3_r;

%
% Plot distribution of sampling values for $\mu$ and $\nu$ when drawn from 
% a uniform distribution
%

% Open a new window
figure(3);

% Plot distribution for $\mu$
bin = 20;
subplot(1,2,1);
hold off
histogram(mu_tr,bin);
hold on
plot([mu1 mu2], Nmu/bin * [1 1], 'g')
plot(mu, zeros(size(mu)), 'rx', 'Markersize', 10);
title('Distribution of $\mu$')
xlabel('$\mu$')
legend('Random sampling', 'Uniform sampling', 'Test values', 'location', 'best')
grid on
xlim([mu1 mu2])

% Plot distribution for $\nu$
bin = 20;
subplot(1,2,2);
hold off
histogram(nu_tr,bin);
hold on
plot([nu1 nu2], Nnu/bin * [1 1], 'g')
plot(nu, zeros(size(nu)), 'rx', 'Markersize', 10);
title('Distribution of $\nu$')
xlabel('$\nu$')
legend('Random sampling', 'Uniform sampling', 'Test values', 'location', 'best')
grid on
xlim([nu1 nu2])
%}

%
% Compare solutions for three values of $\mu$
%

% Open a new window
figure(4);
hold off

% Plot and set the legend
plot(x(1:1:end), ur1(1:1:end), 'b')
hold on
plot(x(1:1:end), ur1_u(1:1:end), 'b--', 'Linewidth', 2)
%plot(x(1:1:end), ur1_r(1:1:end), 'b:', 'Linewidth', 2)
plot(x(1:1:end), ur2(1:1:end), 'r')
plot(x(1:1:end), ur2_u(1:1:end), 'r--', 'Linewidth', 2)
%plot(x(1:1:end), ur2_r(1:1:end), 'r:', 'Linewidth', 2)
plot(x(1:1:end), ur3(1:1:end), 'g')
plot(x(1:1:end), ur3_u(1:1:end), 'g--', 'Linewidth', 2)
%plot(x(1:1:end), ur3_r(1:1:end), 'g:', 'Linewidth', 2)

% Define plot settings
str = sprintf('Solution to Poisson equation ($K = %i$, $N_{tr} = %i$)', ...
    K, Ntr);
title(str)
xlabel('$x$')
ylabel('$u$')
legend(sprintf('$\\mu = %f$, $\\nu = %f$', mu(1), nu(1)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN', mu(1), nu(1)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(1), nu(1)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$', mu(2), nu(2)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN', mu(2), nu(2)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(2), nu(2)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$', mu(3), nu(3)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN', mu(3), nu(3)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(3), nu(3)), ...
    'location', 'best')
grid on
        
%% Fix the number of training samples and plot the average error on test data set
% versus the number of hidden neurons

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (row vector, 
%               no more than four values)
% Nnu_tr        number of training values for $\nu$ (row vector, 
%               same length as Nmu_tr)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network
% train_opt     training algorithm:
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization
% h_opt         number of hidden neurons

Nmu_tr = [5 15 25 50];  Nnu_tr = [5 15 25 50];
valPercentage = 0.3;  Nte_nn = 200;  train_opt = 'trainlm';

%
% Run
% 

% Grid spacing
dx = (b-a) / (K-1);

% Extract reference error, i.e. average distance between full solution and 
% its projection on the reduced space
datafile = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
    'NonLinearPoisson1d2p_%s_%s%s_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nte_r, suffix);
load(datafile);
err_ref_median = median(err_svd_abs);
err_ref_mean = mean(err_svd_abs);

for n = 1:length(Nmu_tr)
    % Total number of training and validating samples
    Ntr = Nmu_tr(n)*Nnu_tr(n);  Nva = ceil(valPercentage*Ntr);

    % Load data for uniform sampling; one hidden layer
    filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
        'NonLinearPoisson1d2p_%s_%s%s_NNunif_' ...
        'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
        'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, ...
        BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
        Nmu_tr(n), Nnu_tr(n), Ntr, Nva, Nte_nn, suffix);
    load(filename);
    
    % Find position in error matrix associated with the desired
    % training algorithm
    col = 0;
    for k = 1:length(trainFcn)
        if strcmp(trainFcn{k},train_opt)
            col = k;
        end
    end
    if isempty(col == 0)
        error('Specified training algorithm not found.')
    end
    
    % Extract data
    if (n == 1)
        h_u = cell(length(Nmu_tr),1);
        hmin = min(H);  hmax = max(H);
        err_u = zeros(length(H),length(Nmu_tr));
    end
    h_u{n} = H;
    err_u(:,n) = sqrt(dx)*err_opt_local(:,col);
    
    % Details for plotting
    if min(H) < hmin
        hmin = min(H);
    end
    if max(H) > hmax
        hmax = max(H);
    end
    
    % Load data for uniform sampling; two hidden layers
    filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
        'NonLinearPoisson1d2p_%s_%s%s_NNunif_' ...
        'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
        'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i_2L%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, ...
        BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
        Nmu_tr(n), Nnu_tr(n), Ntr, Nva, Nte_nn, suffix);
    load(filename);
    
    % Find position in error matrix associated with the desired
    % training algorithm
    col = 0;
    for k = 1:length(trainFcn)
        if strcmp(trainFcn{k},train_opt)
            col = k;
        end
    end
    if isempty(col == 0)
        error('Specified training algorithm not found.')
    end
    
    % Extract data
    if (n == 1)
        h_2l_u = cell(length(Nmu_tr),1);
        err_2l_u = zeros(length(H),length(Nmu_tr));
    end
    h_2l_u{n} = H;
    err_2l_u(:,n) = sqrt(dx)*err_opt_local(:,col);
    
    % Details for plotting
    if min(H) < hmin
        hmin = min(H);
    end
    if max(H) > hmax
        hmax = max(H);
    end
    
    %{
    % Load data for random sampling
    filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
        'NonLinearPoisson1d2p_%s_%s%s_NNrand_' ...
        'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
        'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, ...
        BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
        Nmu_tr(n), Nnu_tr(n), Ntr, Nva, Nte_nn, suffix);
    load(filename);
    
    % Find position in error matrix associated with the desired
    % training algorithm
    col = 0;
    for k = 1:length(trainFcn)
        if strcmp(trainFcn{k},train_opt)
            col = k;
        end
    end
    if isempty(col == 0)
        error('Specified training algorithm not found.')
    end
    
    % Extract data
    if (n == 1)
        h_r = cell{length(Nmu_tr),1};
        err_r = zeros(length(H),length(Nmu_tr));
    end
    h_r{n} = H;
    err_r(:,n) = sqrt(dx)*err_opt_local(:,col);
    
    % Details for plotting
    if min(H) < hmin
        hmin = min(H);
    end
    if max(H) > hmax
        hmax = max(H);
    end
    %}
end

% Open new window
figure(5);
hold off;

% Plot and dynamically update the legend
marker_u = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_2l_u = {'bo--', 'rs--', 'g^--', 'mv--'};
%marker_r = {'bo:', 'rs:', 'g^:', 'mv:'};
str_leg = 'legend(''location'',''best''';
for i = 1:length(Nmu_tr)
    semilogy(h_u{i}, err_u(:,i), marker_u{i}, 'linewidth', 1.2);
    hold on
    semilogy(h_2l_u{i}, err_2l_u(:,i), marker_2l_u{i}, 'linewidth', 1.2);
    %semilogy(h_r{i}, err_r(:,i), marker_r{i}, 'linewidth', 1.2);
    
    str_u = sprintf('''$N_{tr} = %i$''', Nmu_tr(i)*Nnu_tr(i));
    str_2l_u = sprintf('''$N_{tr} = %i$, 2-layer''', Nmu_tr(i)*Nnu_tr(i));
    %str_r = sprintf('''$N_{tr} = %i$, random''', Nmu_tr(i)*Nnu_tr(i));
    str_leg = strcat(str_leg,', ',str_u);
    str_leg = strcat(str_leg,', ',str_2l_u);
    %str_leg = strcat(str_leg,', ',str_r);
end
semilogy([hmin-1 hmax+1], [err_ref_median err_ref_median], 'k-')
str_leg = strcat(str_leg,', ''DM (median)''');
semilogy([hmin-1 hmax+1], [err_ref_mean err_ref_mean], 'k--')
str_leg = strcat(str_leg,', ''DM (mean)'')');

% Define plot settings
title('Average error in $L^2_h$-norm on test data set')
xlabel('$H$')
ylabel('$||u - u^l||_{L^2_h}$')
grid on
eval(str_leg)
xlim([hmin-1 hmax+1])

%% Boxplot for the errors yielded by Direct Method (DM) and Neural Network (NN)

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ 
% Nnu_tr        number of training values for $\nu$ 
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network
% train_opt     training algorithm:
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization
% h_opt         number of hidden neurons

Nmu_tr = 25;  Nnu_tr = 25;  valPercentage = 0.3;  Nte_nn = 200;  
train_opt = 'trainlm';  h_opt = 21;

%
% Run
% 

% Grid spacing
dx = (b-a) / (K-1);

% Extract error yielded by Direct Method
datafile = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
    'NonLinearPoisson1d2p_%s_%s%s_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nte_r, suffix);
load(datafile);
mu = mu_te;  nu = nu_te;

% Extract error yielded by Neural Network
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i_2L%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Nmu_tr*Nnu_tr, ceil(valPercentage*Nmu_tr*Nnu_tr), Nte_nn, suffix);
load(filename);

% Find position in error metrix associated with the desired
% network topology and training algorithm
row = find(H == h_opt);
col = 0;
for k = 1:length(trainFcn)
    if strcmp(trainFcn{k},train_opt)
        col = k;
    end
end
if isempty(row) || (col == 0)
    error('Specified number of hidden layers and/or training algorithm not found.')
end

% Compute errors
net = net_opt_local{row,col};
err_nn_abs = zeros(numel(mu),1);
for i = 1:numel(mu)
    y = net([mu(i) nu(i)]');
    err_nn_abs(i) = sqrt(dx) * norm(u_te(:,i)-VL*y);
end

% Plot boxplot
figure(6)
boxplot([err_svd_abs err_nn_abs],'Labels',{'DM','NN'})

%% For the optimal network, plot error on training, validation and test data 
% set versus epochs

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (row vector)
% Nnu_tr        number of training values for $\nu$ (no more than four values)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network
% train_opt     training algorithm:
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization   
% h_opt         number of hidden neurons

Nmu_tr = 25;  Nnu_tr = 25;  valPercentage = 0.3;  Nte_nn = 200;
train_opt = 'trainlm';  h_opt = 21;

% 
% Run
%

% Determine number of training and validating patterns
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Load data
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i_2L%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Find position in error metrix associated with the desired
% network topology and training algorithm
row = find(H == h_opt);
col = 0;
for k = 1:length(trainFcn)
    if strcmp(trainFcn{k},train_opt)
        col = k;
    end
end
if isempty(row) || (col == 0)
    error('Specified number of hidden layers and/or training algorithm not found.')
end

% Open a new plot window
figure(6);
hold off

% Extract optimal network
tr_opt = tr_opt_local{row,col};

% Plot and define settings
semilogy(tr_opt.epoch,tr_opt.perf,'b', tr_opt.epoch,tr_opt.vperf,'r', ...
    tr_opt.epoch,tr_opt.tperf,'g')

str = sprintf('Learning curves ($H = %i$, $N_{\\mu,tr} = %i$, $N_{\\nu,tr} = %i$)', ...
    h_opt, Nmu_tr, Nnu_tr);
title(str)
xlabel('$t$')
ylabel('$MSE$')
grid on
legend('Train', 'Validation', 'Test', 'location', 'best')

%% For test data, compute regression line of current output versus associated
% teaching input for all output neurons

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (row vector)
% Nnu_tr        number of training values for $\nu$ (no more than four values)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network
% train_opt     training algorithm:
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization   
% h_opt         number of hidden neurons

Nmu_tr = 25;  Nnu_tr = 25;  valPercentage = 0.3;  Nte_nn = 200;
train_opt = 'trainlm';  h_opt = 21;

%
% Run
%

% Get total number of training and validating samples
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Load data
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i_2L%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Find position in error metrix associated with the desired
% network topology and training algorithm
row = find(H == h_opt);
col = 0;
for k = 1:length(trainFcn)
    if strcmp(trainFcn{k},train_opt)
        col = k;
    end
end
if isempty(row) || (col == 0)
    error('Specified number of hidden layers and/or training algorithm not found.')
end

% Load dataset storing reduced basis
load(datafile);
alpha_te = VL'*u_te;

% Extract optimal network and compute outputs
net_opt = net_opt_local{row,col};
y = net_opt([mu_te'; nu_te']);

% Compute regression for each component of the output, then plot
for i = 1:size(y,1)
    [r,m,q] = regression(alpha_te(i,:),y(i,:));
    figure(6+i);
    hold off
    plot(alpha_te(i,:),y(i,:),'bo', alpha_te(i,:),alpha_te(i,:),'r', ...
        [min(alpha_te(i,:)) max(alpha_te(i,:))],m*[min(alpha_te(i,:)) max(alpha_te(i,:))]+q,'r--');
    str = sprintf('Current output versus exact output for output neuron $\\Omega = %i$ ($N_{tr} = %i$, $H = %i$)', ...
        i, Ntr, h_opt);
    title(str)
    xlabel('$t_{\Omega}$')
    ylabel('$y_{\Omega}$')
    grid on
    legend('Output', 'Perfect fitting', 'Regression line', 'location', 'best')
    yl = get(gca,'xlim');
    ylim(yl);
end

%% Plot pointwise error

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ 
% Nnu_tr        number of training values for $\nu$ 
% sampler_tr    how the training values for $\mu$ and $\nu$ should have been 
%               sampled:
%               - 'unif': samples form a Cartesian grid
%               - 'rand': drawn from random uniform distribution
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network
% train_opt     training algorithm:
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization   
% h_opt         number of hidden neurons

Nmu_tr = 25;  Nnu_tr = 25;  sampler_tr = 'unif';
valPercentage = 0.3;  Nte_nn = 200;  train_opt = 'trainlm';  h_opt = 15;

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @NonLinearPoisson1dFEP1;
    rsolverFcn = @NonLinearPoisson1dFEP1Reduced;
elseif strcmp(solver,'FEP1Newton')
    solverFcn = @NonLinearPoisson1dFEP1Newton;
    rsolverFcn = @NonLinearPoisson1dFEP1ReducedNewton;
end

% Determine total number of training and validating patterns
if strcmp(sampler,'unif')
    Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);
elseif strcmp(sampler,'rand')
    Ntr = Nmu_tr;  Nva = ceil(valPercentage*Ntr);
end

% Load data
filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
    'NonLinearPoisson1d2p_%s_%s%s_NN%s_a%2.2f_b%2.2f_' ...
    '%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
    'Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCRt, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Load dataset storing reduced basis
load(datafile);

% Find position in error matrix associated with the desired
% network topology and training algorithm
row = find(H == h_opt);
col = 0;
for k = 1:length(trainFcn)
    if strcmp(trainFcn{k},train_opt)
        col = k;
    end
end
if isempty(row) || (col == 0)
    error('Specified number of hidden layers and/or training algorithm not found.')
end

% Extract optimal network
net = net_opt_local{row,col};

% Determine testing samples
Nte_nn = 1000;
mu_te = mu1 + (mu2-mu1) * rand(Nte_nn,1);
nu_te = nu1 + (nu2-nu1) * rand(Nte_nn,1);

err_abs = zeros(Nte_nn,1);
err_rel = zeros(Nte_nn,1);
for i = 1:Nte_nn    
    % Evaluate forcing term
    g = @(t) f(t,mu_te(i),nu_te(i));
    
    % Compute reduced solution through direct method
    [x,u] = solverFcn(a, b, K, v, g, BCLt, BCLv(mu_te(i),nu_te(i)), ...
        BCRt, BCRv(mu_te(i),nu_te(i)));
    
    % Compute reduced solution through Neural Network
    alpha_nn = net([mu_te(i); nu_te(i)]);
    
    % Compute error
    err_abs(i) = sqrt((x(2)-x(1)) * norm(u - VL*alpha_nn)^2);
    err_rel(i) = norm(u - VL*alpha_nn) / norm(u);
end

% Generate structured data out of scatter data (test samples randomly picked)
mu_g = linspace(mu1,mu2,1000);  nu_g = linspace(nu1,nu2,1000);
[MU,NU] = meshgrid(mu_g,nu_g);
Ea = griddata(mu_te, nu_te, err_abs, MU, NU);
Er = griddata(mu_te, nu_te, err_rel, MU, NU);

% Plot absolute error
figure(7+L);
contourf(MU,NU,Ea, 'LineStyle','none')
colorbar
str = sprintf('Absolute error ($n_{\\mu,tr} = %i$, $n_{\\nu,tr} = %i$)', ...
    Nmu_tr, Nnu_tr);
title(str);
xlabel('$\mu$')
ylabel('$\nu$')

% Plot relative error
figure(8+L);
contourf(MU,NU,Er, 'LineStyle','none')
colorbar
str = sprintf('Relative error ($n_{\\mu,tr} = %i$, $n_{\\nu,tr} = %i$)', ...
    Nmu_tr, Nnu_tr);
title(str);
xlabel('$\mu$')
ylabel('$\nu$')