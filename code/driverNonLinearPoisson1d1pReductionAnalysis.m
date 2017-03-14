% driverNonLinearPoisson1d1pReductionAnalysis Some post-processing for 
% reduction modeling applied to one-dimensional nonlinear Poisson equation 
% $-(v(u) u'(x))' = f(x,\mu)$ in the unknown $u = u(x)$, $x \in [a,b]$,
% depending on the real parameter $\mu \in [\mu_1,\mu_2]$.
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
% v         viscosity $v = v(u)$ as handle function
% dv        derivative of the viscosity as handle function
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
%           - 'FEP1Newton': linear finite elements
% reducer   method to compute the reduced basis
%           - 'SVD': Single Value Decomposition 
% root      path to folder where storing the output dataset

a = -1;  b = 1;  
v = @(t) 1 ./ (t.^2);  dv = @(t) - 2 ./ (t.^3);
f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  suffix = '_bis';
BCLt = 'D';  BCLv = 1;
BCRt = 'D';  BCRv = 1;
solver = 'FEP1Newton';
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

K = 100;  N = 20;  L = 6;  Nte = 100;

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1Newton')
    solverFcn = @NonLinearPoisson1dFEP1Newton;
    rsolverFcn = @NonLinearPoisson1dFEP1ReducedNewton;
end

% Select three values for $\mu$
mu = mu1 + (mu2 - mu1) * rand(3,1);

% Evaluate forcing term for the just set values for $\mu$
g = cell(3,1);
for i = 1:3
    g{i} = @(t) f(t,mu(i));
end

% Load data associated with uniform distribution of $\mu$ and get full and 
% reduced solutions
filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
    'NonLinearPoisson1d1p_%s_%sunif_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N, L, ...
    Nte, suffix);
load(filename);
[x,u1] = solverFcn(a, b, K, v, dv, g{1}, BCLt, BCLv, BCRt, BCRv);
[x,alpha1_unif] = rsolverFcn(a, b, K, v, dv, g{1}, BCLt, BCLv, BCRt, BCRv, VL);
ur1_unif = VL * alpha1_unif;
[x,u2] = solverFcn(a, b, K, v, dv, g{2}, BCLt, BCLv, BCRt, BCRv);
[x,alpha2_unif] = rsolverFcn(a, b, K, v, dv, g{2}, BCLt, BCLv, BCRt, BCRv, VL);
ur2_unif = VL * alpha2_unif;
[x,u3] = solverFcn(a, b, K, v, dv, g{3}, BCLt, BCLv, BCRt, BCRv);
[x,alpha3_unif] = rsolverFcn(a, b, K, v, dv, g{3}, BCLt, BCLv, BCRt, BCRv, VL);
ur3_unif = VL * alpha3_unif;

%{
% Load data associated with random distribution of $\mu$ and get
% reduced solutions
filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
    'NonLinearPoisson1d1p_%s_%srand_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N, L, ...
    Nte, suffix);
load(filename);
[x,alpha1_rand] = rsolverFcn(a, b, K, v, dv, g{1}, BCLt, BCLv, BCRt, BCRv, VL);
ur1_rand = VL * alpha1_rand;
[x,alpha2_rand] = rsolverFcn(a, b, K, v, dv, g{2}, BCLt, BCLv, BCRt, BCRv, VL);
ur2_rand = VL * alpha2_rand;
[x,alpha3_rand] = rsolverFcn(a, b, K, v, dv, g{3}, BCLt, BCLv, BCRt, BCRv, VL);
ur3_rand = VL * alpha3_rand;

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
%}

%
% Compare solutions for three values of $\mu$
%

% Open a new window
figure(2);
hold off

% Plot and set the legend
plot(x(1:1:end), u1(1:1:end), 'b')
hold on
plot(x(1:1:end), ur1_unif(1:1:end), 'b--', 'Linewidth', 2)
%plot(x(1:1:end), ur1_rand(1:1:end), 'b:', 'Linewidth', 2)
plot(x(1:1:end), u2(1:1:end), 'r')
plot(x(1:1:end), ur2_unif(1:1:end), 'r--', 'Linewidth', 2)
%plot(x(1:1:end), ur2_rand(1:1:end), 'r:', 'Linewidth', 2)
plot(x(1:1:end), u3(1:1:end), 'g')
plot(x(1:1:end), ur3_unif(1:1:end), 'g--', 'Linewidth', 2)
%plot(x(1:1:end), ur3_rand(1:1:end), 'g:', 'Linewidth', 2)

% Define plot settings
str_leg = sprintf(['Full and reduced solution to Poisson equation ($k = %i$,' ...
    ' $n = %i$, $l = %i$)'], K, N, L);
title(str_leg)
xlabel('$x$')
ylabel('$u$')
legend(sprintf('$\\mu = %f$, full', mu(1)), ...
    sprintf('$\\mu = %f$, reduced (uniform)', mu(1)), ...
    ... %sprintf('$\\mu = %f$, reduced (random)', mu(1)), ...
    sprintf('$\\mu = %f$, full', mu(2)), ...
    sprintf('$\\mu = %f$, reduced (uniform)', mu(2)), ...
    ... %sprintf('$\\mu = %f$, reduced (random)', mu(2)), ...
    sprintf('$\\mu = %f$, full', mu(3)), ...
    sprintf('$\\mu = %f$, reduced (uniform)', mu(3)), ...
    ... %sprintf('$\\mu = %f$, reduced (random)', mu(3)), ...
    'location', 'best')
grid on

%% To determine which is best between uniform and random sampling,
% plot the maximum and average error versus the number of training samples
% for different values of L

%
% User defined settings:
% K         number of grid points
% N         number of shapshots (row vector)
% L         rank of reduced basis (row vector, no more than four values)
% Nte       number of testing samples

K = 100;  N = [5 10 15 20 25 50 75 100];  L = [10 15 18 20];  Nte = 100;

%
% Run
%

% Get accumulated error for any combination of N and L
err_max_unif = zeros(length(L),length(N));
err_avg_unif = zeros(length(L),length(N));
%err_max_rand = zeros(length(L),length(N));
%err_avg_rand = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        % Uniform sampling
        filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
            'NonLinearPoisson1d1p_%s_%sunif_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_unif(i,j) = max(err_svd_rel);
        err_avg_unif(i,j) = sum(err_svd_rel)/Nte;
        
        %{
        % Random sampling
        filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
            'NonLinearPoisson1d1p_%s_%srand_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_rand(i,j) = max(err_svd_abs);
        err_avg_rand(i,j) = sum(err_svd_abs)/Nte;
        %}
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
%marker_rand = {'bo--', 'rs--', 'g^--', 'mv--'};
str_leg = sprintf('legend(''location'', ''best''');
for i = 1:length(L)
    semilogy(N, err_max_unif(i,:), marker_unif{i});
    hold on
    %semilogy(N, err_max_rand(i,:), marker_rand{i});
    str_unif = sprintf('''L = %i''', L(i));
    %str_rand = sprintf('''L = %i, random''', L(i));
    str_leg = strcat(str_leg, ', ', str_unif);
    %str_leg = strcat(str_leg, ', ', str_unif, ', ', str_rand);
end
str_leg = strcat(str_leg, ')');

% Define plot settings
str = sprintf('Maximum relative error $\\epsilon_{max}$ ($k = %i$, $n_{te} = %i$)', ...
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
%marker_rand = {'bo--', 'rs--', 'g^--', 'mv--'};
str_leg = sprintf('legend(''location'', ''best''');
for i = 1:length(L)
    semilogy(N, err_avg_unif(i,:), marker_unif{i});
    hold on
    %semilogy(N, err_avg_rand(i,:), marker_rand{i});
    str_unif = sprintf('''L = %i''', L(i));
    %str_rand = sprintf('''L = %i, random''', L(i));
    str_leg = strcat(str_leg, ', ', str_unif);
    %str_leg = strcat(str_leg, ', ', str_unif, ', ', str_rand);
end
str_leg = strcat(str_leg, ')');

% Define plot settings
str = sprintf('Average relative error $\\epsilon_{avg}$ ($k = %i$, $n_{te} = %i$)', ...
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

K = 100;  N = 50;  L = 1:25;  Nte = 50;

%
% Run
% 

% Get error accumulated error for all values of L and for even distribution 
% of snapshot values of $\mu$
err_max_unif = zeros(length(L),length(N));
err_avg_unif = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
            'NonLinearPoisson1d1p_%s_%sunif_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_unif(i,j) = max(err_svd_rel);
        err_avg_unif(i,j) = sum(err_svd_rel)/Nte;
    end
end

%{
% Get error accumulated error for all values of L and for random distribution 
% of shapshot values for $\mu$
err_max_rand = zeros(length(L),length(N));
err_avg_rand = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
            'NonLinearPoisson1d1p_%s_%srand_a%2.2f_b%2.2f_' ...
            '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
            K, N(j), L(i), Nte, suffix);
        load(filename);
        err_max_rand(i,j) = max(err_svd_rel);
        err_avg_rand(i,j) = sum(err_svd_rel)/Nte;
    end
end
%}

%
% Plot maximum error
%

% Open a new window
figure(5);
hold off

% Plot data and dynamically update legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
%marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
str_leg = 'legend(''location'', ''best''';
for j = 1:length(N)
    semilogy(L', err_max_unif(:,j), marker_unif{j});
    hold on
    %semilogy(L', err_max_rand(:,j), marker_rand{j});
    str_unif = sprintf('''$n = %i$''', N(j));
    %str_rand = sprintf('''$n = %i$, random''', N(j));
    str_leg = sprintf('%s, %s', str_leg, str_unif);
    %str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
end
semilogy(L,s,'ko--')
str_leg = sprintf('%s, ''Singular values'')', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf('Maximum relative error $\\epsilon_{max}$ ($k = %i$, $n_{te} = %i$)', ...
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
%marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
str_leg = 'legend(''location'', ''best''';
for j = 1:length(N)
    semilogy(L', err_avg_unif(:,j), marker_unif{j});
    hold on
    %semilogy(L', err_avg_rand(:,j), marker_rand{j});
    str_unif = sprintf('''$n = %i$''', N(j));
    %str_rand = sprintf('''$n = %i$, random''', N(j));
    str_leg = sprintf('%s, %s', str_leg, str_unif);
    %str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
end
semilogy(L,s,'ko--')
str_leg = sprintf('%s, ''Singular values'')', str_leg);
eval(str_leg)

% Define plot settings
str_leg = sprintf('Average relative error $\\epsilon_{avg}$ ($k = %i$, $n_{te} = %i$)', ...
    K, Nte);
title(str_leg)
xlabel('$l$')
ylabel('$\epsilon_{avg}$')
grid on    
xlim([min(L)-1 max(L)+1])

%% Fix the number of snapshots (for simplicity) and plot the error
% $|| U - V^l A^l||^2$, where the $i$-th column of $A^l$ is $\boldsymbol{\alpha}_i$.
% The error should behave as $\sum_{i = 1}^l \sigma_i^2$, $\sigma_i$
% being the $i$-th singular value. Compare $A^l$ as computed through SVD
% and through reduced solver.

%
% User defined settings:
% K     number of grid points
% N     number of shapshots
% L     rank of reduced basis
% Nte   number of testing samples

K = 100;  N = 50;  L = 1:25;  Nte = 50;

%
% Run
% 

% Get error accumulated error for all values of L
ref = zeros(length(L),1);
err_svd = zeros(length(L),1);
err_ls = zeros(length(L),1);
err_newton = zeros(length(L),1);
for i = 1:length(L)
    % Load data
    filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
        'NonLinearPoisson1d1p_%s_%sunif_a%2.2f_b%2.2f_' ...
        '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
        root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
        K, N, L(i), Nte, suffix);
    load(filename);
        
    % Compute singular values and get reference value for the error
    [V,S,W] = svd(u_tr);  s = diag(S);  ref(i) = sum(s(L(i)+1:end).^2);
    
    % Compute error for SVD
    B = S*W';  err_svd(i) = norm(u_tr - VL*B(1:L(i),:),'fro')^2;
    
    % Compute error for least-squares
    alpha_ls = VL \ u_tr;  err_ls(i) = norm(u_tr - VL*alpha_ls,'fro')^2;
    
    % Compute error for Newton
    err_newton(i) = norm(u_tr - VL*alpha_tr,'fro')^2;
end

%
% Plot
%

figure(7);
semilogy(L',ref,'ko--', L',err_svd,'bo-', L',err_ls,'ro-', L',err_newton,'g^-')
title('Error on training dataset')
xlabel('$l$')
ylabel('$|| U - U^l ||^2$')
grid on    
xlim([min(L)-1 max(L)+1])
legend('Residuals', 'SVD', 'Least-squares', 'Newton', 'location', 'best')

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

K = 100;  N = 50;  L = 10;  Nte = 100;
sampler = 'unif';

%
% Run
%

% Load data
filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
    'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
    K, N, L, Nte, suffix);
load(filename);

% Select the three solutions to plot
idx = randi(Nte,3,1);

% Open a new window
figure(8);
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
% L         rank of reduced basis (no more than four values)
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

K = 100;  N = 50;  L = [5 10];  Nte = 100;
sampler = 'unif';

%
% Plot a specific solution
%

% Select the solution to plot
idx = randi(Nte,1);

% Open a new plot window
figure(9);
hold off

% Load data, plot and update legend
marker = {'b', 'r', 'g', 'm'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(L)
    filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
        'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, ...
            mu1, mu2, K, N, L(i), Nte, suffix);
    load(filename);
    plot(x(1:1:end), ur_te(1:1:end,idx), marker{i});
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
figure(10);
hold off

% Load data, plot and update legend
marker = {'bo-', 'rs-', 'g^-', 'mv-'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(L)
    filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
        'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N, L(i), Nte, suffix);
    load(filename);
    [mu_te,I] = sort(mu_te);
    err_svd_rel = err_svd_rel(I);
    plot(mu_te, err_svd_rel, marker{i});
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
% N         number of shapshots (no more than four values)
% L         rank of reduced basis
% Nte       number of testing samples
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$

K = 100;  N = [10 25 50 100];  L = 10;  Nte = 100;
sampler = 'unif';

%
% Plot a specific solution
%

% Select the solution to plot
idx = randi(3,1);

% Open a new plot window
figure(11);
hold off

% Load data, plot and update legend
marker = {'b', 'r', 'g', 'm'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(N)
    filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
        'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N(i), L, Nte, suffix);
    load(filename);
    plot(x(1:1:end), ur_te(1:1:end,idx), marker{i});
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
figure(12);
hold off

% Load data, plot and update legend
marker = {'bo-', 'rs-', 'g^-', 'mv-'};
str_leg = 'legend(''location'', ''best''';
for i = 1:length(N)
    filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
        'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
        '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
        root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
        mu2, K, N(i), L, Nte, suffix);
    load(filename);
    [mu_te,I] = sort(mu_te);
    err_svd_rel = err_svd_rel(I);
    plot(mu_te, err_svd_rel, marker{i});
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

K = 100;  N = 50;  L = 10;  Nte = 100;
sampler = 'unif';

%
% Run
%  

% Load data
filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
    'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, ...
    mu2, K, N, L, Nte, suffix);
load(filename);

% Plot basis functions
for l = 1:L
    figure(12+l);
    plot(x,VL(:,l),'b')
    title('Basis function')
    xlabel('$x$')
    str = sprintf('$\\psi^{%i}$', l);
    ylabel(str);
    grid on
end