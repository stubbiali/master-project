% driverHeterogeneousViscosityLinearPoisson1d2pNeuralNetworkAnalysis 
% Consider the one-dimensional linear Poisson equation 
% $-(v(x,\nu)u'(x))' = f(x,\mu)$ on $[a,b]$ in the unknown $u = u(x)$, with
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
% v         viscosity $v = v(t,\nu)$
% f         force field $f = f(t,\mu)$ as handle function
% mu1       lower bound for $\mu$
% mu2       upper bound for $\mu$
% Nmu       number of different samples for $\mu$ used for computing the
%           snapshots
% nu1       lower bound for $\nu$
% nu2       upper bound for $\nu$
% Nnu       number of different samples for $\nu$ used for computing the
%           snapshots
% BCLt      kind of left boundary condition
%           - 'D': Dirichlet, 
%           - 'N': Neumann, 
%           - 'P': periodic
% BCLv      value of left boundary condition
% BCRt      kind of right boundary condition
%           - 'D': Dirichlet, 
%           - 'N': Neumann, 
%           - 'P': periodic
% BCRv      value of right boundary condition
% solver    solver
%           - 'FEP1': linear finite elements
%           - 'FEP2': quadratic finite elements
% reducer   method to compute the reduced basis
%           - 'SVD': Single Value Decomposition, i.e. Proper
%					 Orthogonal Decomposition (POD)
% sampler   how the values for $\mu$ and $\nu$ should have been picked for
%           the computation of the snapshots in the reduction step:
%           - 'unif': sampled values form a Cartesian grid
%           - 'rand': sampled values drawn from a uniform random distribution
% L         rank of reduced basis
% Nte       number of testing samples stored in the dataset for reduction       
% root      path to folder where storing the output dataset

a = -1;  b = 1;  K = 100;
v = @(t,nu) 2 + sin(nu*pi*t);
f = @(t,mu) -(t < mu) + 2*(t >= mu);  
suffix = '_ter';
mu1 = -1;  mu2 = 1;  Nmu = 50;
nu1 = 1;   nu2 = 3;  Nnu = 10;
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
L = 10;  Nte = 50;
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

Nmu_tr = 50;  Nnu_tr = 50;  valPercentage = 0.3;  Nte_nn = 200;

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @HeterogeneousViscosityLinearPoisson1dFEP1_f;
elseif strcmp(solver,'FEP2')
    solverFcn = @HeterogeneousViscosityLinearPoisson1dFEP2_f;
end

% Get total number of training and validating patterns
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Select three values for $\mu$ and $\nu$
%mu = mu1 + (mu2 - mu1) * rand(3,1);
%mu = [-0.455759 -0.455759 -0.455759]; 
%nu = nu1 + (nu2 - nu1) * rand(3,1);
%nu = [0.03478 0.5 0.953269]; 

% Evaluate viscosity for the just set values for $\nu$
vis = cell(3,1);
for i = 1:3
    vis{i} = @(t) v(t,nu(i));
end

% Evaluate forcing term for the just set values for $\mu$
g = cell(3,1);
for i = 1:3
    g{i} = @(t) f(t,mu(i));
end

% Load data for uniform sampling
filename = sprintf(['%s/HeterogeneousViscosityLinearPoisson1d2pNN/' ...
    'HeterogeneousViscosityLinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Load dataset storing reduced basis
load(datafile);

% Compute reduced solution to direct method
[x, alpha1] = solverFcn(a, b, K, vis{1}, g{1}, BCLt, BCLv, BCRt, BCRv, UL);
ur1 = UL * alpha1;
[x, alpha2] = solverFcn(a, b, K, vis{2}, g{2}, BCLt, BCLv, BCRt, BCRv, UL);
ur2 = UL * alpha2;
[x, alpha3] = solverFcn(a, b, K, vis{3}, g{3}, BCLt, BCLv, BCRt, BCRv, UL);
ur3 = UL * alpha3;

% Extract optimal Neural Network and compute reduced solution through it
net = net_opt_local{row_opt,col_opt};
alpha1_u = net([mu(1); nu(1)]);  ur1_u = UL*alpha1_u;
alpha2_u = net([mu(2); nu(2)]);  ur2_u = UL*alpha2_u;
alpha3_u = net([mu(3); nu(3)]);  ur3_u = UL*alpha3_u;

%{
% Load data and get reduced solutions for random sampling
filename = sprintf(['%s/HeterogeneousViscosityLinearPoisson1d2pNN/' ...
    'HeterogeneousViscosityLinearPoisson1d2p_%s_%s%s_NNrand_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, BCRv, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);
net = net_opt_local{row_opt,col_opt};
alpha1_r = net([mu(1); nu(1)]);  ur1_r = UL*alpha1_r;
alpha2_r = net([mu(2); nu(2)]);  ur2_r = UL*alpha2_r;
alpha3_r = net([mu(3); nu(3)]);  ur3_r = UL*alpha3_r;

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
    sprintf('$\\mu = %f$, $\\nu = %f$, NN (uniform)', mu(1), nu(1)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(1), nu(1)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, reduced', mu(2), nu(2)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN (uniform)', mu(2), nu(2)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(2), nu(2)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, reduced', mu(3), nu(3)), ...
    sprintf('$\\mu = %f$, $\\nu = %f$, NN (uniform)', mu(3), nu(3)), ...
    ... %sprintf('$\\mu = %f$, $\\nu = %f$, NN (random)', mu(3), nu(3)), ...
    'location', 'best')
grid on

%% Senisitivity analysis on the number of hidden neurons: plot both number of 
% hidden layers and accumulated error on test data set versus the number of
% samples for $\mu$ and $\nu$. pcolor plots are used. For ease of
% implementation, only uniform sampling of training patterns is considered.

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (row vector)
% Nnu_tr        number of training values for $\nu$ (row vector)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network

Nmu_tr = [5 10 15 20 30 40 50];  Nnu_tr = [5 10 15 20 30 40 50];
valPercentage = 0.3;  Nte_nn = 200; 

%
% Run
%

% Allocate memory for number of layers and error
h_opt = zeros(length(Nmu_tr),length(Nnu_tr));
err_opt = zeros(length(Nmu_tr),length(Nnu_tr));

for i = 1:length(Nmu_tr)
    for j = 1:length(Nnu_tr)
        % Determine total number of training and validating patterns
        Ntr = Nmu_tr(i)*Nnu_tr(j);  Nva = ceil(valPercentage*Ntr);
        
        % Load data
        filename = sprintf(['%s/HeterogeneousViscosityLinearPoisson1d2pNN/' ...
            'HeterogeneousViscosityLinearPoisson1d2p_%s_%s%s_NNunif_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
            mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Nmu_tr(i), Nnu_tr(j), Ntr, Nva, Nte_nn, suffix);
        load(filename);
        
        % Get values
        h_opt(i,j) = H(row_opt);
        err_opt(i,j) = err_opt_local(row_opt,col_opt);
    end
end

% Adjust Nmu_tr and Nnu_tr for pcolor
Nmu_tr = repmat(Nmu_tr', [1,length(Nnu_tr)]);
Nnu_tr = repmat(Nnu_tr, [size(Nmu_tr,1),1]);

% Plot number of hidden neurons
figure(3);
pcolor(Nmu_tr,Nnu_tr,h_opt)
colorbar
title('Number of hidden neurons')
xlabel('$n_{\mu,tr}$')
ylabel('$n_{\nu,tr}$')

% Plot error
figure(4);
pcolor(Nmu_tr,Nnu_tr,err_opt)
colorbar
title('Accumulated error')
xlabel('$n_{\mu,tr}$')
ylabel('$n_{\nu,tr}$')
        

%% Given the optimal network, plot the accumulated error on testing dataset
% as a function of the number of training samples for $\nu$

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (no more than four values)
% Nnu_tr        number of training values for $\nu$ (row vector)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network

Nmu_tr = [10 20 30 50];  Nnu_tr = [1 2 5 10 20 30 40 50];
valPercentage = 0.3;  Nte_nn = 200;

%
% Run
% 

% Allocate space for error
err_u = zeros(length(Nmu_tr),length(Nnu_tr));
%err_r = zeros(length(Nmu_tr),length(Nnu_tr));

for i = 1:length(Nmu_tr)
    for j = 1:length(Nnu_tr)
        % Total number of training and validating samples
        Ntr = Nmu_tr(i)*Nnu_tr(j);  Nva = ceil(valPercentage*Ntr);
        
        % Load data for uniform sampling
        filename = sprintf(['%s/LinearPoisson1d2p/LinearPoisson1d2pNN/' ...
            'LinearPoisson1d2p_%s_%s%s_NNunif_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, ...
            BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Nmu_tr(i), Nnu_tr(j), Ntr, Nva, Nte_nn, suffix);
        load(filename);
        err_u(i,j) = err_opt_local(row_opt,col_opt);
        
        %{
        % Load data for random sampling
        filename = sprintf(['%s/LinearPoisson1d2p/LinearPoisson1d2pNN/' ...
            'LinearPoisson1d2p_%s_%s%s_NNrand_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, ...
            BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Ntr, Ntr, Ntr, Nva, Nte_nn, suffix);
        load(filename);
        err_r(i,j) = err_opt_local(row_opt,col_opt);
        %}
    end
end

% Open new window
figure(3);
hold off;

% Plot and dynamically update the legend
marker_u = {'b', 'r', 'g', 'm'};
%marker_r = {'b--', 'r--', 'g--', 'm--'};
str_leg = 'legend(''location'',''best''';
for i = 1:length(Nmu_tr)
    semilogy(Nnu_tr, err_u(i,:), marker_u{i});
    hold on
    %semilogy(Nnu_tr, err_r(i,:), marker_r{i});
    
    str_u = sprintf('''$N_{\\mu} = %i$''', Nmu_tr(i));
    %str_r = sprintf('''N_{mu} = %i, random''', Nmu_tr(i));
    str_leg = strcat(str_leg,', ',str_u);
    %str_leg = strcat(str_leg,', ',str_u,', ',str_r);
end
str_leg = strcat(str_leg,')');

% Define plot settings
title('Accumulated error $\epsilon$ on testing dataset')
xlabel('$N_{\nu}$')
ylabel('$\epsilon$')
grid on
eval(str_leg)

%% Given the optimal network, plot the accumulated error on testing dataset
% as a function of the number of training samples for $\mu$

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (row vector)
% Nnu_tr        number of training values for $\nu$ (no more than four values)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network

Nmu_tr = [10 20 30 40 50];  Nnu_tr = [5 10 20 50];
valPercentage = 0.3;  Nte_nn = 200;

%
% Run
% 

% Allocate space for error
err_u = zeros(length(Nnu_tr),length(Nmu_tr));
%err_r = zeros(length(Nmu_tr),length(Nnu_tr));

for i = 1:length(Nnu_tr)
    for j = 1:length(Nmu_tr)
        % Total number of training and validating samples
        Ntr = Nmu_tr(j)*Nnu_tr(i);  Nva = ceil(valPercentage*Ntr);
        
        % Load data for uniform sampling
        filename = sprintf(['%s/LinearPoisson1d2p/LinearPoisson1d2pNN/' ...
            'LinearPoisson1d2p_%s_%s%s_NNunif_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, ...
            BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Nmu_tr(j), Nnu_tr(i), Ntr, Nva, Nte_nn, suffix);
        load(filename);
        err_u(i,j) = err_opt_local(row_opt,col_opt);
        
        %{
        % Load data for random sampling
        filename = sprintf(['%s/LinearPoisson1d2p/LinearPoisson1d2pNN/' ...
            'LinearPoisson1d2p_%s_%s%s_NNrand_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, ...
            BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Ntr, Ntr, Ntr, Nva, Nte_nn, suffix);
        load(filename);
        err_r(i,j) = err_opt_local(row_opt,col_opt);
        %}
    end
end

% Open new window
figure(4);
hold off;

% Plot and dynamically update the legend
marker_u = {'b', 'r', 'g', 'm'};
marker_r = {'b--', 'r--', 'g--', 'm--'};
str_leg = 'legend(''location'',''best''';
for i = 1:length(Nnu_tr)
    semilogy(Nmu_tr, err_u(i,:), marker_u{i});
    hold on
    %semilogy(Nmu_tr, err_r(i,:), marker_r{i});
    
    str_u = sprintf('''$N_{\\nu} = %i$''', Nnu_tr(i));
    %str_r = sprintf('''N_{nu} = %i, random''', Nnu_tr(i));
    str_leg = strcat(str_leg,', ',str_u);
    %str_leg = strcat(str_leg,', ',str_u,', ',str_r);
end
str_leg = strcat(str_leg,')');

% Define plot settings
title('Accumulated error $\epsilon$ on testing dataset')
xlabel('$N_{\mu}$')
ylabel('$\epsilon$')
grid on
eval(str_leg)

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

Nmu_tr = 50;  Nnu_tr = 50;  sampler_tr = 'unif';
valPercentage = 0.3;  Nte_nn = 200;

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @LinearPoisson1dFEP1;
elseif strcmp(solver,'FEP2')
    solverFcn = @LinearPoisson1dFEP2;
end

% Determine total number of training and validating patterns
if strcmp(sampler,'unif')
    Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);
elseif strcmp(sampler,'rand')
    Ntr = Nmu_tr;  Nva = ceil(valPercentage*Ntr);
end

% Load data
filename = sprintf(['%s/LinearPoisson1d2p/LinearPoisson1d2pNN/' ...
    'LinearPoisson1d2p_%s_%s%s_NN%s_a%2.2f_b%2.2f_' ...
    '%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
    'Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, BCRt, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);
load(datafile);

% Extract optimal network
net = net_opt_local{row_opt,col_opt};

% Determine testing samples
Nte_nn = 1000;
mu_te = mu1 + (mu2-mu1) * rand(Nte_nn,1);
nu_te = nu1 + (nu2-nu1) * rand(Nte_nn,1);

err_abs = zeros(Nte_nn,1);
err_rel = zeros(Nte_nn,1);
for i = 1:Nte_nn
    % Evaluate forcing term
    g = @(t) f(t,mu_te(i));
    
    % Compute reduced solution through direct method
    [x,alpha] = solverFcn(a, b, K, g, BCLt, BCLv, BCRt, nu_te(i), UL);
    
    % Compute reduced solution through Neural Network
    alpha_nn = net([mu_te(i); nu_te(i)]);
    
    % Compute error
    err_abs(i) = norm(alpha_nn - alpha);
    err_rel(i) = norm(alpha_nn - alpha) / norm(alpha);
end

% Generate structured data out of scatter data (test samples randomly picked)
mu_g = linspace(mu1,mu2,1000);  nu_g = linspace(nu1,nu2,1000);
[MU,NU] = meshgrid(mu_g,nu_g);
Ea = griddata(mu_te, nu_te, err_abs, MU, NU);
Er = griddata(mu_te, nu_te, err_rel, MU, NU);

% Plot absolute error
figure(5);
contourf(MU,NU,Ea, 'LineStyle','none')
colorbar
str = sprintf('Absolute error ($n_{\\mu,tr} = %i$, $n_{\\nu,tr} = %i$)', ...
    Nmu_tr, Nnu_tr);
title(str);
xlabel('$\mu$')
ylabel('$\nu$')

% Plot relative error
figure(6);
contourf(MU,NU,Er, 'LineStyle','none')
colorbar
str = sprintf('Relative error ($n_{\\mu,tr} = %i$, $n_{\\nu,tr} = %i$)', ...
    Nmu_tr, Nnu_tr);
title(str);
xlabel('$\mu$')
ylabel('$\nu$')