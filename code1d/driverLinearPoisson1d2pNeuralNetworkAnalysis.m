% driverLinearPoisson1d2pNeuralNetworkAnalysis Consider the one-dimensional 
% linear Poisson equation $-u''(x) = f(x,\mu)$ on $[a,b]$ depending on the 
% real parameters $\mu$ and $\nu$, with the latter incorporated in the right
% boundary condition. Given a reduced basis $U^l$ of rank $l$ and the
% associated reduced solution $\boldsymbol{\alpha} \in \mathbb{R}^l$,
% this script aims at analyzing the results for the approximation of the map
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
% f         force field $f = f(t,\mu)$ as handle function
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
% BCLv      value of left boundary condition
% BCRt      kind of right boundary condition
%           - 'D': Dirichlet, 
%           - 'N': Neumann, 
%           - 'P': periodic
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
% Nte_r     number of testing samples stored in the dataset for reduction       
% root      path to folder where storing the output dataset

a = -1;  b = 1;  K = 100;

% Suffix ''
%f = @(t,mu) gaussian(t,mu,0.2);  
%mu1 = -1;  mu2 = 1;  nu1 = 0;  nu2 = 0.5;  suffix = '';
% Suffix '_ter'
f = @(t,mu) -(t < mu) + 2*(t >= mu);  
mu1 = -1;  mu2 = 1;  nu1 = 0;  nu2 = 1;  suffix = '_ter';

Nmu = 50;  Nnu = 10;
BCLt = 'D';  BCLv = 0;
BCRt = 'D';
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
L = 10;  Nte_r = 50;
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
    solverFcn = @LinearPoisson1dFEP1;
elseif strcmp(solver,'FEP2')
    solverFcn = @LinearPoisson1dFEP2;
end

% Get total number of training and validating patterns
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Select three values for $\mu$ and $\nu$
mu = mu1 + (mu2 - mu1) * rand(3,1);
%mu = [-0.455759 -0.455759 -0.455759]; 
nu = nu1 + (nu2 - nu1) * rand(3,1);
%nu = [0.03478 0.5 0.953269];  

% Evaluate forcing term for the just set values for $\mu$
g = cell(3,1);
for i = 1:3
    g{i} = @(t) f(t,mu(i));
end

% Load data for uniform sampling
filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
    'LinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Load dataset storing reduced basis
datafile = sprintf(['%s/LinearPoisson1d2pSVD/' ...
    'LinearPoisson1d2p_%s_%s%s_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nte_r, suffix);
load(datafile);

% Compute reduced solution to direct method
[x, alpha1] = solverFcn(a, b, K, g{1}, BCLt, BCLv, BCRt, nu(1), UL);
ur1 = UL * alpha1;
[x, alpha2] = solverFcn(a, b, K, g{2}, BCLt, BCLv, BCRt, nu(2), UL);
ur2 = UL * alpha2;
[x, alpha3] = solverFcn(a, b, K, g{3}, BCLt, BCLv, BCRt, nu(3), UL);
ur3 = UL * alpha3;

% Extract optimal Neural Network and compute reduced solution through it
net = net_opt_local{row_opt,col_opt};
alpha1_u = net([mu(1); nu(1)]);  ur1_u = UL*alpha1_u;
alpha2_u = net([mu(2); nu(2)]);  ur2_u = UL*alpha2_u;
alpha3_u = net([mu(3); nu(3)]);  ur3_u = UL*alpha3_u;

%{
% Load data and get reduced solutions for random sampling
filename = sprintf(['%s/LinearPoisson1d2p/LinearPoisson1d2pNN/' ...
    'LinearPoisson1d2p_%s_%s%s_NNrand_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
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

%% Sensitivity analysis on the number of hidden neurons: fix the number of 
% samples for $\nu$, then plot average error versus number of samples for
% $\mu$ for different number of hidden neurons

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (no more than four values)
% Nnu_tr        number of training values for $\nu$ (row vector)
% hv            number of hidden neurons (no more than four values)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network

Nmu_tr = [5 10 15 20 25 50];  Nnu_tr = [5 15 25];  hv = [5 10 15 20];
valPercentage = 0.3;  Nte_nn = 200;

%
% Run
% 

% Grid spacing
dx = (b-a) / (K-1);

% Get reference error, i.e. error yielded by SVD
datafile = sprintf(['%s/LinearPoisson1d2pSVD/' ...
    'LinearPoisson1d2p_%s_%s%s_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nte_r, suffix);
load(datafile);
err_ref = sqrt(dx)*mean(err_svd_abs);

for n = 1:length(Nnu_tr)
    % Allocate space for error
    err_u = zeros(length(Nmu_tr),length(hv));
    %err_r = zeros(length(Nmu_tr),length(Nnu_tr));

    for i = 1:length(Nmu_tr)
        % Total number of training and validating samples
        Ntr = Nmu_tr(i)*Nnu_tr(n);  Nva = ceil(valPercentage*Ntr);

        % Load data for uniform sampling
        filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
            'LinearPoisson1d2p_%s_%s%s_NNunif_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, ...
            BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Nmu_tr(i), Nnu_tr(n), Ntr, Nva, Nte_nn, suffix);
        load(filename);
        
        for j = 1:length(hv)
            % Find index for hv(j) and extract error
            idx = find(H == hv(j));
            err_u(i,j) = sqrt(dx)*err_opt_local(idx,col_opt);
        end
        
        %{
        % Load data for random sampling
        filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
            'LinearPoisson1d2p_%s_%s%s_NNrand_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, ...
            BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Nmu_tr(i), Nnu_tr(j), Ntr, Nva, Nte_nn, suffix);
        load(filename);
        
        for j = 1:length(hv)
            % Find index for hv(j) and extract error
            idx = find(H == hv(j));
            err_r(i,j) = sqrt(dx)*err_opt_local(idx,col_opt);
        end
        %}
    end
            
    % Open new window
    figure(2+n);
    hold off;

    % Plot and dynamically update the legend
    marker_u = {'bo-', 'rs-', 'g^-', 'mv-'};
    %marker_r = {'bo--', 'r--', 'g--', 'm--'};
    str_leg = 'legend(''location'',''best''';
    for j = 1:length(hv)
        semilogy(Nmu_tr, err_u(:,j), marker_u{j}, 'linewidth', 1.2);
        hold on
        %semilogy(Nnu_tr, err_r(i,:), marker_r{i});

        str_u = sprintf('''$H = %i$''', hv(j));
        %str_r = sprintf('''$H = %i$, random''', hv(j));
        str_leg = strcat(str_leg,', ',str_u);
        %str_leg = strcat(str_leg,', ',str_u,', ',str_r);
    end
    semilogy(Nmu_tr([1 end]), [err_ref err_ref], 'k--')
    str_leg = strcat(str_leg,', ''DM'')');

    % Define plot settings
    str = sprintf('Average error in $L^2_h$-norm on test data set ($N_{\\nu,tr} = %i$)', ...
        Nnu_tr(n));
    title(str)
    xlabel('$N_{\mu,tr}$')
    ylabel('$||u - u^l||_{L^2_h}$')
    grid on
    eval(str_leg)
end

%% Given the optimal network, plot the accumulated error on testing dataset
% as a function of the number of training samples for $\nu$

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (no more than four values)
% Nnu_tr        number of training values for $\nu$ (row vector)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network

Nmu_tr = [5 15 25 50];  Nnu_tr = [5 10 15 20 25 50];
valPercentage = 0.3;  Nte_nn = 200;

%
% Run
% 

% Allocate space for error
err_u = zeros(length(Nmu_tr),length(Nnu_tr));
%err_r = zeros(length(Nmu_tr),length(Nnu_tr));

dx = (b-a) / (K-1);
for i = 1:length(Nmu_tr)
    for j = 1:length(Nnu_tr)
        % Total number of training and validating samples
        Ntr = Nmu_tr(i)*Nnu_tr(j);  Nva = ceil(valPercentage*Ntr);
        
        % Load data for uniform sampling
        filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
            'LinearPoisson1d2p_%s_%s%s_NNunif_' ...
            'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, ...
            BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
            Nmu_tr(i), Nnu_tr(j), Ntr, Nva, Nte_nn, suffix);
        load(filename);
        err_u(i,j) = sqrt(dx)*err_opt_local(row_opt,col_opt);
        
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
marker_u = {'bo-', 'rs-', 'g^-', 'mv-'};
%marker_r = {'bo--', 'r--', 'g--', 'm--'};
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
        filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
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

%% For the optimal network, plot error on training, validation and test data 
% set versus epochs

%
% User defined settings:
% Nmu_tr        number of training values for $\mu$ (row vector)
% Nnu_tr        number of training values for $\nu$ (no more than four values)
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% Nte_nn        number of testing samples for Neural Network

Nmu_tr = 50;  Nnu_tr = 50;  
valPercentage = 0.3;  Nte_nn = 200;

% 
% Run
%

% Determine number of training and validating patterns
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Load data
filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
    'LinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Open a new plot window
figure(5);
hold off

% Extract optimal network
tr_opt = tr_opt_local{row_opt,col_opt};

% Plot and define settings
semilogy(tr_opt.epoch,tr_opt.perf,'b', tr_opt.epoch,tr_opt.vperf,'r', ...
    tr_opt.epoch,tr_opt.tperf,'g')

str = sprintf('Learning curves ($h = %i$, $n_{\\mu,tr} = %i$, $n_{\\nu,tr} = %i$ $n_{va} = %i$, $n_{te} = %i$)', ...
    H(row_opt), Nmu_tr, Nnu_tr, Nva, Nte);
title(str)
xlabel('$t$')
ylabel('$\epsilon$')
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

Nmu_tr = 50;  Nnu_tr = 50;  valPercentage = 0.3;  Nte_nn = 200;

%
% Run
%

% Get total number of training and validating samples
Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);

% Load data
filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
    'LinearPoisson1d2p_%s_%s%s_NNunif_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
    Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Load dataset storing reduced basis
datafile = sprintf(['%s/LinearPoisson1d2pSVD/' ...
    'LinearPoisson1d2p_%s_%s%s_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nte_r, suffix);
load(datafile);

% Extract optimal network and compute outputs
net_opt = net_opt_local{row_opt,col_opt};
y = net_opt([mu_te'; nu_te']);

% Compute regression for each component of the output, then plot
for i = 1:size(y,1)
    [r,m,q] = regression(alpha_te(i,:),y(i,:));
    figure(5+i);
    hold off
    plot(alpha_te(i,:),y(i,:),'bo', alpha_te(i,:),alpha_te(i,:),'r', ...
        [min(alpha_te(i,:)) max(alpha_te(i,:))],m*[min(alpha_te(i,:)) max(alpha_te(i,:))]+q,'r--');
    str = sprintf('Current output versus exact output for output neuron $\\Omega = %i$ ($n_{tr} = %i$, $h = %i$)', ...
        i, Ntr, H(row_opt));
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
filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
    'LinearPoisson1d2p_%s_%s%s_NN%s_a%2.2f_b%2.2f_' ...
    '%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
    'Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, BCRt, ...
    mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nmu_tr, Nnu_tr, Ntr, Nva, Nte_nn, suffix);
load(filename);

% Load dataset storing reduced basis
datafile = sprintf(['%s/LinearPoisson1d2pSVD/' ...
    'LinearPoisson1d2p_%s_%s%s_' ...
    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, ...
    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nte_r, suffix);
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

