% driverLinearPoisson1d1pNeuralNetworkAnalysis Consider the one-dimensional 
% linear Poisson equation $-u''(x) = f(x,\mu)$ on $[a,b]$ depending on the 
% real parameter $\mu$ and an associated reduced basis $U^l$ of rank $l$.
% Denoting by $\boldsymbol{\alpha}$ the reduced solution. this script aims 
% at analyzing the results for the approximation of the map
% $\mu \mapto \boldsymbol{\alpha}$ through a Neural Network.

clc
clear variables
clear variables -global
close all

%
% User-defined settings:
% a             left boundary of the domain
% b             right boundary of the domain
% K             number of grid points
% f             forcing term $f = f(x,\mu)$ as handle function
% mu1           lower bound for $\mu$
% mu2           upper bound for $\mu$
% suffix        suffix for data file name
% BCLt          kind of left boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% BCLv          value of left boundary condition
% BCRt          kind of right boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% BCRv          value of right boundary condition
% solver        solver
%               - 'FEP1': linear finite elements
%               - 'FEP2': quadratic finite elements
% reducer       method to compute the reduced basis
%               - 'SVD': Single Value Decomposition 
% sampler       how the values for $\mu$ used to compute the solution
%               should have been selected;
%               - 'unif': uniform distribution over $\mu_1,\mu_2$
%               - 'rand': drawn from a uniform random distribution over $\mu_1,\mu_2$
% N             number of snapshots
% L             rank of reduced basis
% Nte           number of testing values for $\mu$
% root          path to folder where storing the output dataset

a = -1;  b = 1;  K = 100;

% Suffix ''
f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  suffix = '';
% Suffix '_bis'
%f = @(t,mu) 50 * t .* cos(mu*pi*t);  mu1 = 1;  mu2 = 3;  suffix = '_bis';
% Suffix '_ter'
%f = @(t,mu) -(t < mu) + 2*(t >= mu);  mu1 = -1;  mu2 = 1;  suffix = '_ter';

BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
N = 50;  L = 10;  Nte = 50;
root = '../datasets';

%% For each training algorithm and for each sampling method for the training
% patterns, plot the minimum error versus the number of training patterns. 
% Useful to detect the best algorithm and which is best between uniform and
% random sampling.

% 
% User-defined settings:
% Ntr_v  number of training patterns (row vector)
% Nva_v  number of validation patterns (row vector, same length as Ntr_v)

Ntr_v = 10:10:80;  Nva_v = ceil(0.3 * Ntr_v);

%
% Run
%

% Grid spacing
dx = (b-a) / (K-1);

% Get training algorithms which has been tested
filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
    'LinearPoisson1d1p_%s_%s%s_NNunif_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, K, N, L, Ntr_v(1), Nva_v(1), Nte, suffix);
load(filename);

% Get optimal number of hidden layers and minimum error for each number of 
% training patterns, each training algorithm and each sampling method
H_opt_unif = zeros(length(Ntr_v),length(trainFcn));
H_opt_rand = zeros(length(Ntr_v),length(trainFcn));
err_opt_unif = zeros(length(Ntr_v),length(trainFcn));
err_opt_rand = zeros(length(Ntr_v),length(trainFcn));

for i = 1:length(Ntr_v)
    % Uniform sampling
    filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
        'LinearPoisson1d1p_%s_%s%s_NNunif_a%2.2f_b%2.2f_' ...
        '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N, L, Ntr_v(i), Nva_v(i), Nte, suffix);
    load(filename); 
    [err_opt_unif(i,:),I] = min(sqrt(dx)*err_opt_local);
    H_opt_unif(i,:) = H(I);
    
    % Random sampling
    filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
        'LinearPoisson1d1p_%s_%s%s_NNrand_a%2.2f_b%2.2f_' ...
        '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N, L, Ntr_v(i), Nva_v(i), Nte, suffix);
    load(filename); 
    [err_opt_rand(i,:),I] = min(sqrt(dx)*err_opt_local);
    H_opt_rand(i,:) = H(I);
end

%
% Plot number of hidden neurons versus number of training patterns
%

% Open a new window
figure(1);
hold off

% Load data, plot and dynamically update legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_rand = {'bo--', 'rs--', 'g^--', 'mv--'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(trainFcn)
    plot(Ntr_v', H_opt_unif(:,i), marker_unif{i});
    hold on
    plot(Ntr_v', H_opt_rand(:,i), marker_rand{i});
    str_unif = sprintf('''%s, uniform''', trainFcn{i});
    str_rand = sprintf('''%s, random''', trainFcn{i});
    str_leg = strcat(str_leg, ', ', str_unif, ', ', str_rand);
end
str_leg = strcat(str_leg,')');

% Define plot settings
str = sprintf('Optimal number of hidden layers ($k = %i$, $n = %i$, $n_{te} = %i$)', ...
    K, N, Nte);
title(str)
xlabel('$n_{tr}$')
ylabel('$H_{opt}$')
grid on
eval(str_leg)

%
% Plot minimum error versus number of training patterns
%

% Open a new window
figure(2);
hold off

% Load data and plot
for i = 1:length(trainFcn)
    semilogy(Ntr_v', err_opt_unif(:,i), marker_unif{i});
    hold on
    semilogy(Ntr_v', err_opt_rand(:,i), marker_rand{i});
end

% Define plot settings
str = sprintf('Average error in $L^2_h$-norm on test data set ($k = %i$, $n = %i$, $n_{te} = %i$)', ...
    K, N, Nte);
title(str)
xlabel('$n_{tr}$')
ylabel('$||u - u^l||_{L^2_h}$')
grid on
eval(str_leg)    

%% Fix the number of hidden neurons, than for each sampling method and each 
% training algorithm plot the average error on test dataset as function
% of the number of training samples

% 
% User-defined settings:
% Ntr_v  number of training patterns (row vector)
% Nva_v  number of validation patterns (row vector, same length as Ntr_v)
% h_opt  number of hidden neurons

Ntr_v = 10:10:80;  Nva_v = ceil(0.3 * Ntr_v);  h_opt = 10;

%
% Run
%

% Grid spacing
dx = (b-a)/(K-1);

% Get training algorithms which has been tested
filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
    'LinearPoisson1d1p_%s_%s%s_NNunif_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, K, N, L, Ntr_v(1), Nva_v(1), Nte, suffix);
load(filename);

% Get for each number of training patterns, each training algorithm and 
% each sampling method
err_unif = zeros(length(Ntr_v),length(trainFcn));
err_rand = zeros(length(Ntr_v),length(trainFcn));

for i = 1:length(Ntr_v)
    % Uniform sampling
    filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
        'LinearPoisson1d1p_%s_%s%s_NNunif_a%2.2f_b%2.2f_' ...
        '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N, L, Ntr_v(i), Nva_v(i), Nte, suffix);
    load(filename); 
    idx = find(H == h_opt);  err_unif(i,:) = sqrt(dx)*err_opt_local(idx,:);
        
    % Random sampling
    filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
        'LinearPoisson1d1p_%s_%s%s_NNrand_a%2.2f_b%2.2f_' ...
        '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N, L, Ntr_v(i), Nva_v(i), Nte, suffix);
    load(filename); 
    idx = find(H == h_opt);  err_rand(i,:) = sqrt(dx)*err_opt_local(idx,:);
end

%
% Plot number of hidden neurons versus number of training patterns
%

% Open a new window
figure(3);
hold off

% Load data, plot and dynamically update legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_rand = {'bo--', 'rs--', 'g^--', 'mv--'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(trainFcn)
    semilogy(Ntr_v', err_unif(:,i), marker_unif{i});
    hold on
    semilogy(Ntr_v', err_rand(:,i), marker_rand{i});
    str_unif = sprintf('''%s, uniform''', trainFcn{i});
    str_rand = sprintf('''%s, random''', trainFcn{i});
    str_leg = strcat(str_leg, ', ', str_unif, ', ', str_rand);
end
str_leg = strcat(str_leg,')');

% Define plot settings
str = sprintf('Average error in $L^2_h$-norm on test data set ($h = %i$, $n_{te} = %i$)', ...
    h_opt, Nte);
title(str)
xlabel('$n_{tr}$')
ylabel('$||u - u^l||_{L^2_h}$')
grid on
eval(str_leg)

%% For different number of hidden neurons, fix sampling method and 
% training algorithm plot the accumulated error on test dataset as function
% of the number of training samples

% 
% User-defined settings:
% Ntr_v         number of training patterns (row vector)
% Nva_v         number of validation patterns (row vector, same length as Ntr_v)
% h             number of hidden neurons (row vector, no more than four values)
% sampler_tr    how the training values for $\mu$ should be selected:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on 
%                         $[\mu_1,\mu_2]$
% train         training algorithm
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method

Ntr_v = 10:10:80;  Nva_v = ceil(0.3 * Ntr_v);  h = [5 10 15 20];
sampler_tr = 'unif';  train = 'trainlm';

%
% Run
%

% Grid size
dx = (b-a)/(K-1);

% Extract reference value for the error, i.e. the error got with SVD
datafile = sprintf(['%s/LinearPoisson1d1pSVD/' ...
    'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_' ...
    '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, ...
    BCRt, BCRv, mu1, mu2, K, N, L, Nte, suffix);
load(datafile);
err_ref = mean(sqrt(dx)*err_svd_abs);
    
err = zeros(length(Ntr_v),length(h));
for i = 1:length(Ntr_v)
    % Load data
    filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
        'LinearPoisson1d1p_%s_%s%s_NN%s_a%2.2f_b%2.2f_' ...
        '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, ...
        BCRt, BCRv, mu1, mu2, K, N, L, Ntr_v(i), Nva_v(i), Nte, suffix);
    load(filename);
    
    % Extract column index associated with the specified training algorithm
    jopt = 0;
    for j = 1:length(trainFcn)
        if strcmp(trainFcn{j},train)
            jopt = j;
        end
    end
    
    for j = 1:length(h)
        % Find row index associated with the specified number of neurons
        iopt = find(H == h(j));
        
        % Get the error
        err(i,j) = sqrt(dx)*err_opt_local(iopt,jopt);
    end
end

%
% Plot
%

% Open a new window
figure(4);
hold off

% Load data, plot and dynamically update legend
marker = {'bo-', 'rs-', 'g^-', 'mv-'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(h)
    semilogy(Ntr_v', err(:,i), marker{i}, 'Linewidth', 1.2);
    hold on
    str = sprintf('''$H = %i$''', h(i));
    str_leg = strcat(str_leg, ', ', str);
end
semilogy(Ntr_v([1 end]), [err_ref err_ref], 'k--')
str_leg = strcat(str_leg,', ''DM'')');

% Define plot settings
title('Average error in $L^2_h$-norm on test data set')
xlabel('$N_{tr}$')
ylabel('$||u - u^l||_{L^2_h}$')
grid on
eval(str_leg)

%% Fixed the number of training patterns, for each training algorithm and 
% way of sampling, plot the error versus the number of hidden neurons

% 
% User-defined settings:
% Ntr   number of training patterns
% Nva   number of validation patterns

Ntr = 30;  Nva = ceil(0.3*Ntr);

%
% Run
%

% Grid spacing
dx = (b-a) / (K-1);

% Load data for uniform sampling
filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
    'LinearPoisson1d1p_%s_%s%s_NNunif_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
    K, N, L, Ntr, Nva, Nte, suffix);
load(filename);
err_opt_local_unif = sqrt(dx)*err_opt_local;

% Load data for random sampling
filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
    'LinearPoisson1d1p_%s_%s%s_NNrand_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
    K, N, L, Ntr, Nva, Nte, suffix);
load(filename);
err_opt_local_rand = sqrt(dx)*err_opt_local;

% Open a new plot window
figure(3);
hold off

% Load data, plot and dynamically update the legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_rand = {'bo--', 'rs--', 'g^--', 'mv--'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(trainFcn)
    semilogy(H, err_opt_local_unif(:,i), marker_unif{i});
    hold on
    semilogy(H, err_opt_local_rand(:,i), marker_rand{i});
    str_unif = sprintf('''%s, uniform''', trainFcn{i});
    str_rand = sprintf('''%s, random''', trainFcn{i});
    str_leg = strcat(str_leg, ', ', str_unif, ', ', str_rand);
end
str_leg = strcat(str_leg,')');

% Define plot settings
str = sprintf('Average error in $L^2_h$-norm on test data set ($n_{tr} = %i$, $n_{te} = %i$)', ...
    Ntr, Nte);
title(str)
xlabel('$h$')
ylabel('$||u - u^l||_{L^2_h}$')
grid on
eval(str_leg)

%% For the optimal network, plot error on training, validation and test data 
% set versus epochs

% 
% User-defined settings:
% Ntr           number of training patterns
% Nva           number of validation patterns
% sampler_tr    how the training values for $\mu$ should be selected:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

Ntr = 30;  Nva = ceil(0.3*Ntr);  sampler_tr = 'unif';

% Load data
filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
    'LinearPoisson1d1p_%s_%s%s_NN%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, K, N, L, Ntr, Nva, Nte, suffix);
load(filename);
% Open a new plot window
figure(4);
hold off

% Extract optimal network
tr_opt = tr_opt_local{row_opt,col_opt};

% Plot and define settings
semilogy(tr_opt.epoch,tr_opt.perf,'b', tr_opt.epoch,tr_opt.vperf,'r', ...
    tr_opt.epoch,tr_opt.tperf,'g')

str = sprintf('Learning curves ($h = %i$, $n_{tr} = %i$, $n_{va} = %i$, $n_{te} = %i$)', ...
    H(row_opt), Ntr, Nva, Nte);
title(str)
xlabel('$t$')
ylabel('$\epsilon$')
grid on
legend('Train', 'Validation', 'Test', 'location', 'best')

%% For test data, compute regression line of current output versus associated
% teaching input for all output neurons

% 
% User-defined settings:
% Ntr   number of training patterns
% Nva           number of validation patterns
% sampler_tr    how the training values for $\mu$ should be selected:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

Ntr = 50;  Nva = ceil(0.3*Ntr);  sampler_tr = 'unif';

% Load data
filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
    'LinearPoisson1d1p_%s_%s%s_NN%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, K, N, L, Ntr, Nva, Nte, suffix);
load(filename);

% Load training and testing data
datafile = sprintf(['%s/LinearPoisson1d1pSVD/' ...
    'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, K, N, L, Nte, suffix);
load(datafile);

% Extract optimal network and compute outputs
net_opt = net_opt_local{row_opt,col_opt};
y = net_opt(mu_te');

% Compute regression for each component of the output, then plot
for i = 1:size(y,1)
    [r,m,q] = regression(alpha_te(i,:),y(i,:));
    figure(4+i);
    plot(alpha_te(i,:),y(i,:),'bo', alpha_te(i,:),alpha_te(i,:),'r', alpha_te(i,:),r,'r:');
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

%% Comparison between full and reduced solution as given by the neural network

% 
% User-defined settings:
% Ntr   number of training patterns
% Nva           number of validation patterns
% sampler_tr    how the training values for $\mu$ should be selected:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

Ntr = 50;  Nva = ceil(0.3*Ntr);  sampler_tr = 'unif';

%
% Run
%

% Load data
filename = sprintf(['%s/LinearPoisson1d1pNN/' ...
    'LinearPoisson1d1p_%s_%s%s_NN%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, K, N, L, Ntr, Nva, Nte, suffix);
load(filename);

% Load training and testing data
datafile = sprintf(['%s/LinearPoisson1d1pSVD/' ...
    'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
    mu1, mu2, K, N, L, Nte, suffix);
load(datafile);

% Extract optimal network and compute outputs
net_opt = net_opt_local{row_opt,col_opt};
y = net_opt(mu_te');
Y = UL * y;

% Randomly selected three test samples for $\mu$
idx = randi(size(y,2), [3,1]);

% Open a new plot window
figure(L+5);
hold off

% Plot
plot(x, ur_te(:,idx(1)), 'b');
hold on
plot(x, Y(:,idx(1)), 'b--');
plot(x, ur_te(:,idx(2)), 'r');
plot(x, Y(:,idx(2)), 'r--');^
plot(x, ur_te(:,idx(3)), 'g');
plot(x, Y(:,idx(3)), 'g--');

% Define plot settings
str1 = sprintf('Comparison between reduced solution obtained through direct method');
str2 = sprintf('and Neural Network ($k = %i$, $n = %i$, $l = %i$, $h = %i$, $n_{tr} = %i$)', ...
    K, N, L, H(row_opt), Ntr);
title({strcat('\makebox[5in][c]{', str1, '}'), strcat('\makebox[5in][c]{', str2, '}')},...
    'Interpreter','latex')
xlabel('$x$')
ylabel('$ul$')
grid on
legend(sprintf('$\\mu = %f$', mu_te(idx(1))), ...
    sprintf('$\\mu = %f$, Neural Network', mu_te(idx(1))), ...
    sprintf('$\\mu = %f$', mu_te(idx(2))), ...
    sprintf('$\\mu = %f$, Neural Network', mu_te(idx(2))), ...
    sprintf('$\\mu = %f$', mu_te(idx(3))), ...
    sprintf('$\\mu = %f$, Neural Network', mu_te(idx(3))), ...
    'location', 'best')

