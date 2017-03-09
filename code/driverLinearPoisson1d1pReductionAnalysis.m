% driverLinearPoisson1d1pReductionAnalysis Some post-processing for the
% reduction modeling applied to one-dimensional linear Poisson equation 
% $-u''(x) = f(x,\mu)$ on $[a,b]$ depending on the real parameter $\mu$.
% The reduced basis has been obtained through, e.g., SVD and the reduced solution
% has been computed through the direct method, i.e. solving the reduced model.

clc
clear variables
clear variables -global
close all

%
% User-defined settings:
% a         left boundary of the domain
% b         right boundary of the domain
% f         force field $f = f(t,\mu)$ as handle function
% mu1       lower bound for $\mu$
% mu2       upper bound for $\mu$
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
%           - 'SVD': Single Value Decomposition 
% root      path to folder where storing the output dataset

a = -1;  b = 1;  
%f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  suffix = '';
%f = @(t,mu) 50*t.*cos(mu*pi*t);  mu1 = 1;  mu2 = 3;  suffix = '_bis';
f = @(t,mu) -(t < mu) + 2*(t >= mu);  mu1 = -1;  mu2 = 1;  suffix = '_ter';
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
root = '../datasets';

%% Plot full and reduced solution for three training values of $\mu$
% Both uniform and random distribution of $\mu$ over $[\mu_1,\mu_2]$ are
% considered. This is useful to have some insights into which is the best 
% sampling method.

%
% User defined settings:
% K         number of grid points
% N         number of shapshots
% L         rank of reduced basis
% Nte       number of testing samples

K = 100;  N = 50;  L = 10;  Nte = 50;

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @LinearPoisson1dFEP1;
elseif strcmp(solver,'FEP2')
    solverFcn = @LinearPoisson1dFEP2;
end

% Select three values for $\mu$
%mu = mu1 + (mu2 - mu1) * rand(3,1);

% Evaluate forcing term for the just set values for $\mu$
g = cell(3,1);
for i = 1:3
    g{i} = @(t) f(t,mu(i));
end

% Load data associated with uniform distribution of $\mu$ and get full and 
% reduced solutions
filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
    'LinearPoisson1d1p_%s_%sunif_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N, L, ...
    Nte, suffix);
load(filename);
[x, u, alpha_unif] = solverFcn(a, b, K, g, BCLt, BCLv, BCRt, BCRv, UL);
ur_unif = UL * alpha_unif;

% Load data associated with random distribution of $\mu$ and get
% reduced solutions
filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
    'LinearPoisson1d1p_%s_%srand_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N, L, ...
    Nte, suffix);
load(filename);
[x, alpha_rand] = solverFcn(a, b, K, g, BCLt, BCLv, BCRt, BCRv, UL);
ur_rand = UL * alpha_rand;

%
% Plot distribution of sampling values for $\mu$ drawn from a uniform
% distribution
%

% Open a new window
figure(1);
hold off

% Plot
bin = 20;
histogram(mu_tr,bin);
hold on
plot([mu1 mu2], N/bin * [1 1], 'g')
plot(mu, zeros(size(mu)), 'rx', 'Markersize', 10);

% Define plot settings
title('Distribution of randomly sampled values for $\mu$')
xlabel('$\mu$')
legend('Random sampling', 'Uniform sampling', 'Test values', 'location', 'best')
grid on

%
% Compare solutions for three values of $\mu$
%

% Open a new window
figure(2);
hold off

% Plot and set the legend
plot(x(1:1:end), u(1:1:end,1), 'b')
hold on
plot(x(1:1:end), ur_unif(1:1:end,1), 'b--', 'Linewidth', 2)
plot(x(1:1:end), ur_rand(1:1:end,1), 'b:', 'Linewidth', 2)
plot(x(1:1:end), u(1:1:end,2), 'r')
plot(x(1:1:end), ur_unif(1:1:end,2), 'r--', 'Linewidth', 2)
plot(x(1:1:end), ur_rand(1:1:end,2), 'r:', 'Linewidth', 2)
plot(x(1:1:end), u(1:1:end,3), 'g')
plot(x(1:1:end), ur_unif(1:1:end,3), 'g--', 'Linewidth', 2)
plot(x(1:1:end), ur_rand(1:1:end,3), 'g:', 'Linewidth', 2)

% Define plot settings
str_leg = sprintf(['Full and reduced solution to Poisson equation ($k = %i$,' ...
    ' $n = %i$, $l = %i$)'], K, N, L);
title(str_leg)
xlabel('$x$')
ylabel('$u$')
legend(sprintf('$\\mu = %f$, full', mu(1)), sprintf('$\\mu = %f$, reduced (uniform)', mu(1)), ...
    sprintf('$\\mu = %f$, reduced (random)', mu(1)), ...
    sprintf('$\\mu = %f$, full', mu(2)), sprintf('$\\mu = %f$, reduced (uniform)', mu(2)), ...
    sprintf('$\\mu = %f$, reduced (random)', mu(2)), ...
    sprintf('$\\mu = %f$, full', mu(3)), sprintf('$\\mu = %f$, reduced (uniform)', mu(3)), ...
    sprintf('$\\mu = %f$, reduced (random)', mu(3)), ...
    'location', 'best')
grid on

%% To determine which is best between uniform and random sampling,
% plot the maximum and average error versus the number of training samples

%
% User defined settings:
% K         number of grid points
% N         number of shapshots (row vector)
% L         rank of reduced basis (row vector, no more than four values)
% Nte       number of testing samples

K = 100;  N = [10 25 50 75 100];  L = [5 10 15];  Nte = 50;

%
% Run
%

% Get accumulated error for any combination of N and L
err_max_unif = zeros(length(L),length(N));
err_avg_unif = zeros(length(L),length(N));
err_max_rand = zeros(length(L),length(N));
err_avg_rand = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        % Uniform sampling
        filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
            'LinearPoisson1d1p_%s_%sunif_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_unif(i,j) = max(err_svd_abs);
        err_avg_unif(i,j) = sum(err_svd_abs)/Nte;
        
        % Random sampling
        filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
            'LinearPoisson1d1p_%s_%srand_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_rand(i,j) = max(err_svd_abs);
        err_avg_rand(i,j) = sum(err_svd_abs)/Nte;
    end
end

% 
% Plot maximum error
%

% Open a new window
figure(3);
hold off

% Plot and dynamically update the legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_rand = {'bo--', 'rs--', 'g^--', 'mv--'};
str_leg = sprintf('legend(''location'', ''best''');
for i = 1:length(L)
    semilogy(N, err_max_unif(i,:), marker_unif{i});
    hold on
    semilogy(N, err_max_rand(i,:), marker_rand{i});
    str_unif = sprintf('''L = %i, uniform''', L(i));
    str_rand = sprintf('''L = %i, random''', L(i));
    str_leg = strcat(str_leg, ', ', str_unif, ', ', str_rand);
end
str_leg = strcat(str_leg, ')');

% Define plot settings
str = sprintf('Maximum error $\\epsilon_{max}$ ($k = %i$, $n_{te} = %i$)', ...
    K, Nte);
title(str)
xlabel('$n$')
ylabel('$\epsilon_{max}$')
grid on
eval(str_leg);

% 
% Plot average error
%

% Open a new window
figure(4);
hold off

% Plot and dynamically update the legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_rand = {'bo--', 'rs--', 'g^--', 'mv--'};
str_leg = sprintf('legend(''location'', ''best''');
for i = 1:length(L)
    semilogy(N, err_avg_unif(i,:), marker_unif{i});
    hold on
    semilogy(N, err_avg_rand(i,:), marker_rand{i});
    str_unif = sprintf('''L = %i, uniform''', L(i));
    str_rand = sprintf('''L = %i, random''', L(i));
    str_leg = strcat(str_leg, ', ', str_unif, ', ', str_rand);
end
str_leg = strcat(str_leg, ')');

% Define plot settings
str = sprintf('Average error $\\epsilon_{avg}$ ($k = %i$, $n_{te} = %i$)', ...
    K, Nte);
title(str)
xlabel('$n$')
ylabel('$\epsilon_{avg}$')
grid on
eval(str_leg);

%% A complete sensitivity analysis on the sampling method, the number of
% snapshots and the rank of the reduced basis: plot the maximum and average
% error versus number of basis functions for different number of snapshots.

%
% User defined settings:
% K     number of grid points
% N     number of shapshots (no more than four values)
% L     rank of reduced basis

K = 100;  N = [10 25 50 100];  L = 1:25;  Nte = 50;

%
% Run
% 

% Get error accumulated error for all values of L and for even distribution 
% of snapshot values of $\mu$
err_max_unif = zeros(length(L),length(N));
err_avg_unif = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
            'LinearPoisson1d1p_%s_%sunif_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_unif(i,j) = max(err_svd_rel);
        err_avg_unif(i,j) = sum(err_svd_rel)/Nte;
    end
end

% Get error accumulated error for all values of L and for random distribution 
% of shapshot values for $\mu$
err_max_rand = zeros(length(L),length(N));
err_avg_rand = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
            'LinearPoisson1d1p_%s_%srand_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_rand(i,j) = max(err_svd_rel);
        err_avg_rand(i,j) = sum(err_svd_rel)/Nte;
    end
end

%
% Plot maximum error
%

% Open a new window
figure(5);
hold off

% Plot data and dynamically update legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
str_leg = 'legend(''location'', ''best''';
for j = 1:length(N)
    semilogy(L', err_max_unif(:,j), marker_unif{j});
    hold on
    semilogy(L', err_max_rand(:,j), marker_rand{j});
    str_unif = sprintf('''$n = %i$, uniform''', N(j));
    str_rand = sprintf('''$n = %i$, random''', N(j));
    str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
end
str_leg = sprintf('%s)', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf('Maximum error $\\epsilon_{max}$ ($k = %i$, $n_{te} = %i$)', ...
    K, Nte);
title(str_leg)
xlabel('$l$')
ylabel('$\epsilon_{max}$')
grid on    
xlim([min(L)-1 max(L)+1])

%
% Plot average error
%

% Open a new window
figure(6);
hold off

% Plot data and dynamically update legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
str_leg = 'legend(''location'', ''best''';
for j = 1:length(N)
    semilogy(L', err_avg_unif(:,j), marker_unif{j});
    hold on
    semilogy(L', err_avg_rand(:,j), marker_rand{j});
    str_unif = sprintf('''$n = %i$, uniform''', N(j));
    str_rand = sprintf('''$n = %i$, random''', N(j));
    str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
end
str_leg = sprintf('%s)', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf('Average error $\\epsilon_{avg}$ ($k = %i$, $n_{te} = %i$)', ...
    K, Nte);
title(str_leg)
xlabel('$l$')
ylabel('$\epsilon_{avg}$')
grid on    
xlim([min(L)-1 max(L)+1])

%% Fix the sampling method and plot full and reduced solution for three 
% testing values of $\mu$ (This is actually really similar to the first
% setcion, yet here only one sampling method is considered)

%
% User defined settings:
% K         number of grid points
% N         number of shapshots
% L         rank of reduced basis
% Nte       number of testing samples
% sampler   how the shapshot values for $\mu$ should be sampled:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

K = 100;  N = 10;  L = 10;  Nte = 50;
sampler = 'unif';

%
% Run
%

% Load data
filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
    'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
    K, N, L, Nte, suffix);
load(filename);

% Select the three solutions to plot
idx = randi(Nte,3,1);

% Open a new window
figure(7);
hold off

% Plot and set the legend
plot(x(1:1:end), u_te(1:1:end,idx(1)), 'b')
hold on
plot(x(1:1:end), ur_te(1:1:end,idx(1)), 'b:', 'Linewidth', 2)
plot(x(1:1:end), u_te(1:1:end,idx(2)), 'r')
plot(x(1:1:end), ur_te(1:1:end,idx(2)), 'r:', 'Linewidth', 2)
plot(x(1:1:end), u_te(1:1:end,idx(3)), 'g')
plot(x(1:1:end), ur_te(1:1:end,idx(3)), 'g:', 'Linewidth', 2)

% Define plot settings
str_leg = sprintf(['Full and reduced solution to Poisson equation ($k = %i$,' ...
    ' $n = %i$, $l = %i$)'], K, N, L);
title(str_leg)
xlabel('$x$')
ylabel('$u$')
legend(sprintf('$\\mu = %f$, full', mu_te(idx(1))), sprintf('$\\mu = %f$, reduced', mu_te(idx(1))), ...
    sprintf('$\\mu = %f$, full', mu_te(idx(2))), sprintf('$\\mu = %f$, reduced', mu_te(idx(2))), ...
    sprintf('$\\mu = %f$, full', mu_te(idx(3))), sprintf('$\\mu = %f$, reduced', mu_te(idx(3))), ...
    'location', 'best')
grid on

%% Fix the sampling method and the number of snapshots and perform a sensitivity 
% analysis on the pointwise error as the rank of the reduced basis varies

%
% User defined settings:
% K         number of grid points
% N         number of shapshots
% L         rank of reduced basis
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

K = 100;  N = 50;  L = [3 5 10];  Nte = 50;
sampler = 'unif';

%
% Plot a specific solution
%

% Select the solution to plot
idx = randi(Nte,1);

% Open a new plot window
figure(6);
hold off

% Load data, plot and update legend
str_leg = 'legend(''location'', ''best''';
for i = 1:length(L)
    filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
        'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
            mu1, mu2, K, N, L(i), Nte, suffix);
    load(filename);
    plot(x(1:1:end), ur_te(1:1:end,idx));
    hold on
    stri = sprintf('''$l = %i$''', L(i));
    str_leg = sprintf('%s, %s', str_leg, stri);
end
plot(x(1:1:end), u_te(1:1:end,idx));
str_leg = sprintf('%s, ''Full'')', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf(['Full and reduced solution to Poisson equation ($k = %i$,' ...
    ' $n = %i$, $\\mu = %f$)'], K, N, mu_te(idx));
title(str_leg)
xlabel('$x$')
ylabel('$u$')
grid on


%
% Plot error versus $\mu$
%

% Open a new plot window
figure(7);
hold off

% Load data, plot and update legend
marker = {'o-', 's-', '^-', 'x-'};
str_leg = 'legend(''location'', ''best''';
for i = 1:3 %i = 1:length(L)
    filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
        'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N, L(i), Nte, suffix);
    load(filename);
    [mu_te,I] = sort(mu_te);
    err_svd_rel = err_svd_rel(I);
    plot(mu_te, err_svd_rel, marker{mod(i,4)+1});
    hold on
    stri = sprintf('''$l = %i$''', L(i));
    str_leg = sprintf('%s, %s', str_leg, stri);
end
str_leg = sprintf('%s)', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf(['Relative error between full and reduced solution to ' ...
    'Poisson equation ($k = %i$, $n = %i$)'], K, N);
title(str_leg)
xlabel('$\mu$')
ylabel('$\left\Vert u - u^l \right\Vert / \left\Vert u \right\Vert$')
grid on

%% Fix the sampling method and the rank of the reduced basis and perform a 
% sensitivity analysis on the pointwise error as the number of snapshots varies

%
% User defined settings:
% K         number of grid points
% N         number of shapshots
% L         rank of reduced basis
% Nte       number of testing samples
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

K = 100;  N = [10 25 50 100];  L = 5;  Nte = 50;
sampler = 'unif';

%
% Plot a specific solution
%

% Select the solution to plot
idx = randi(3,1);

% Open a new plot window
figure(8);
hold off

% Load data, plot and update legend
str_leg = 'legend(''location'', ''best''';
for i = 1:length(N)
    filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
        'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N(i), L, Nte, suffix);
    load(filename);
    plot(x(1:1:end), ur_te(1:1:end,idx));
    hold on
    stri = sprintf('''$n = %i$''', N(i));
    str_leg = sprintf('%s, %s', str_leg, stri);
end
plot(x(1:1:end), u_te(1:1:end,idx));
str_leg = sprintf('%s, ''Full'')', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf(['Full and reduced solution to Poisson equation ($k = %i$,' ...
    ' $l = %i$, $\\mu = %f$)'], K, L, mu_te(idx));
title(str_leg)
xlabel('$x$')
ylabel('$u$')
grid on

%
% Plot error versus $\mu$
%

% Open a new plot window
figure(9);
hold off

% Load data, plot and update legend
marker = {'o-', 's-', '^-', 'x-'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(N)
    filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
        'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N(i), L, Nte, suffix);
    load(filename);
    [mu_te,I] = sort(mu_te);
    err_svd_rel = err_svd_rel(I);
    plot(mu_te, err_svd_rel, marker{mod(i,4)+1});
    hold on
    stri = sprintf('''$n = %i$''', N(i));
    str_leg = sprintf('%s, %s', str_leg, stri);
end
str_leg = sprintf('%s)', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf(['Relative error between full and reduced solution to ' ...
    'Poisson equation ($k = %i$, $l = %i$)'], K, L);
title(str_leg)
xlabel('$\mu$')
ylabel('$\left\Vert u - u^l \right\Vert / \left\Vert u \right\Vert$')
grid on

%% Plot basis function

%
% User defined settings:
% K         number of grid points
% N         number of shapshots
% L         rank of reduced basis
% Nte       number of testing samples
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

K = 100;  N = 50;  L = 5;  Nte = 50;
sampler = 'unif';

%
% Run
%  

% Load data
filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
    'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
    mu2, K, N, L, Nte, suffix);
load(filename);

% Plot basis functions
for l = 1:L
    figure(9+l);
    plot(x,UL(:,l),'b')
    title('Basis function')
    xlabel('$x$')
    str = sprintf('$\\psi^%i$', l);
    ylabel(str);
    grid on
end

