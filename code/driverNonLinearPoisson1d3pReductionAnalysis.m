% driverNonLinearPoisson1d3pReductionAnalysis Some post-processing for the
% reduction modeling applied to one-dimensional nonlinear Poisson equation 
% $-(v(u) u'(x))' = f(x,\mu,\nu,\xi)$ in the unknown $u = u(x)$, $x \in [a,b]$,
% depending on the real parameters $\mu$, $\nu$ and $\xi$. Note that these
% parameters may enter also the boundary conditions.
% The reduced basis has been obtained through, e.g., SVD and the reduced 
% solution has been computed through the direct method, i.e. solving the 
% reduced model.

clc
clear variables
clear variables -global
%close all

%
% User-defined settings:
% a         left boundary of the domain
% b         right boundary of the domain
% K         number of grid points
% v         viscosity $v = v(u)$ as handle function
% dv        derivative of the viscosity as handle function
% f         force field $f = f(t,\mu)$ as handle function
% mu1       lower bound for $\mu$
% mu2       upper bound for $\mu$
% nu1       lower bound for $\nu$
% nu2       upper bound for $\nu$
% xi1       lower bound for $\xi$
% xi2       upper bound for $\xi$
% BCLt      kind of left boundary condition
%           - 'D': Dirichlet, 
%           - 'N': Neumann, 
%           - 'P': periodic
% BCLv      left boundary condition as handle function in mu, nu and xi
% BCRt      kind of right boundary condition
%           - 'D': Dirichlet, 
%           - 'N': Neumann, 
%           - 'P': periodic
% BCRv      right boundary conditions as handle function in mu, nu and xi
% solver    solver
%           - 'FEP1': linear finite elements; the resulting nonlinear
%                     system is solved via the Matlab built-in fucntion fsolve
%           - 'FEP1Newton': linear finite elements; the resulting
%                           nonlinear system is solved via Newton's method
% reducer   method to compute the reduced basis
%           - 'SVD': Single Value Decomposition, i.e. Proper
%					 Orthogonal Decomposition (POD)
% root      path to folder where storing the output dataset

a = -pi/2;  b = pi/2;  K = 500;
v = @(u) exp(u);  dv = @(u) exp(u);
u = @(t,mu,nu,xi) nu*exp(xi*t).*(2+sin(mu*t));
du = @(t,mu,nu,xi) nu*exp(xi*t).*(xi*(2+sin(mu*t)) + mu*cos(mu*t));
ddu = @(t,mu,nu,xi) nu*exp(xi*t).*(xi*xi*(2+sin(mu*t)) + xi*mu*cos(mu*t) + ...
    mu*xi*cos(mu*t) - mu*mu*sin(mu*t));
f = @(t,mu,nu,xi) - exp(u(t,mu,nu,xi)) .* (du(t,mu,nu,xi).^2 + ddu(t,mu,nu,xi));
mu1 = 1;  mu2 = 3;  nu1 = 1;  nu2 = 3;  xi1 = -0.5;  xi2 = 0.5;  suffix = '';
BCLt = 'D';  BCLv = @(mu,nu,xi) nu.*exp(xi*a).*(2+sin(mu*a));
BCRt = 'D';  BCRv = @(mu,nu,xi) nu.*exp(xi*b).*(2+sin(mu*b));
solver = 'FEP1';
reducer = 'SVD';
root = '../datasets';

%% Plot full and reduced solution for three values of $\mu$, $\nu$ and $\xi$. 
% This is useful to have some insights into the dependency of
% the solution on the parameters and which is best between uniform and random
% sampling method.

%
% User defined settings:
% Nmu	    number of sampled values for $\mu$
% Nnu		number of sampled values for $\nu$
% Nxi		number of sampled values for $\xi$
% L         rank of reduced basis
% Nte       number of testing samples

Nmu = 15;  Nnu = 15;  Nxi = 15;  L = 25;  Nte = 100;

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

% Get total number of snapshots
N = Nmu*Nnu*Nxi;

% Select three values for $\mu$ and $\nu$
%mu = mu1 + (mu2 - mu1) * rand(3,1);
%mu = [-0.455759 -0.455759 -0.455759];  
%nu = nu1 + (nu2 - nu1) * rand(3,1);
%nu = [0.03478 0.5 0.953269];  
%xi = xi1 + (xi2 - xi1) * rand(3,1);

% Evaluate forcing term for the just set values for $\mu$, $\nu$ and $\xi$
g = cell(3,1);
for i = 1:3
    g{i} = @(t) f(t,mu(i),nu(i),xi(i));
end

% Load data and get full and reduced solutions for uniform sampling
filename = sprintf(['%s/NonLinearPoisson1d3pSVD/' ...
    'NonLinearPoisson1d3p_%s_%sunif_a%2.2f_b%2.2f_%s_%s_' ...
    'mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_xi1%2.2f_xi2%2.2f_K%i_Nmu%i_Nnu%i_Nxi%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, a, b, BCLt, BCRt, mu1, mu2, nu1, nu2, xi1, xi2, K, ...
    Nmu, Nnu, Nxi, N, L, Nte, suffix);
load(filename);

[x, u1] = solverFcn(a, b, K, v, g{1}, BCLt, BCLv(mu(1),nu(1),xi(1)), ...
    BCRt, BCRv(mu(1),nu(1),xi(1)));
[x, alpha1_unif] = rsolverFcn(a, b, K, v, g{1}, BCLt, BCLv(mu(1),nu(1),xi(1)), ...
    BCRt, BCRv(mu(1),nu(1),xi(1)), VL);
ur1_unif = VL * alpha1_unif;
%alpha1_unif = VL \ u1;  ur1_unif = VL*alpha1_unif;
[x, u2] = solverFcn(a, b, K, v, g{2}, BCLt, BCLv(mu(2),nu(2),xi(2)), ...
    BCRt, BCRv(mu(2),nu(2),xi(2)));
[x, alpha2_unif] = rsolverFcn(a, b, K, v, g{2}, BCLt, BCLv(mu(2),nu(2),xi(2)), ...
    BCRt, BCRv(mu(2),nu(2),xi(2)), VL);
ur2_unif = VL * alpha2_unif;
%alpha2_unif = VL \ u2;  ur2_unif = VL*alpha2_unif;
[x, u3] = solverFcn(a, b, K, v, g{3}, BCLt, BCLv(mu(3),nu(3),xi(3)), ...
    BCRt, BCRv(mu(3),nu(3),xi(3)));
[x, alpha3_unif] = rsolverFcn(a, b, K, v, g{3}, BCLt, BCLv(mu(3),nu(3),xi(3)), ...
    BCRt, BCRv(mu(3),nu(3),xi(3)), VL);
ur3_unif = VL * alpha3_unif;
%alpha3_unif = VL \ u3;  ur3_unif = VL*alpha3_unif;

%{
% Load data and get reduced solutions for random sampling
filename = sprintf(['%s/NonLinearPoisson1d3pSVD/' ...
    'NonLinearPoisson1d3p_%s_%srand_a%2.2f_b%2.2f_%s_%s_' ...
    'mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_xi1%2.2f_xi2%2.2f_K%i_Nmu%i_Nnu%i_Nxi%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, a, b, BCLt, BCRt, mu1, mu2, nu1, nu2, xi1, xi2, K, ...
    N, N, N, N, L, Nte, suffix);
load(filename);

[x, alpha1_rand] = rsolverFcn(a, b, K, v, g{1}, BCLt, BCLv(mu(1),nu(1),xi(1)), ...
    BCRt, BCRv(mu(1),nu(1),xi(1)), VL);
ur1_rand = VL * alpha1_rand;
[x, alpha2_rand] = rsolverFcn(a, b, K, v, g{2}, BCLt, BCLv(mu(2),nu(2),xi(2)), ...
    BCRt, BCRv(mu(2),nu(2),xi(2)), VL);
ur2_rand = VL * alpha2_rand;
[x, alpha3_rand] = rsolverFcn(a, b, K, v, g{3}, BCLt, BCLv(mu(3),nu(3),xi(3)), ...
    BCRt, BCRv(mu(3),nu(3),xi(3)), VL);
ur3_rand = VL * alpha3_rand;

%
% Plot distribution of sampling values for $\mu$ and $\nu$ when drawn from 
% a uniform distribution
%

% Open a new window
figure(1);

% Plot distribution for $\mu$
bin = 20;
subplot(1,3,1);
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
subplot(1,3,2);
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

% Plot distribution for $\nu$
bin = 20;
subplot(1,3,3);
hold off
histogram(xi_tr,bin);
hold on
plot([xi1 xi2], Nxi/bin * [1 1], 'g')
plot(xi, zeros(size(xi)), 'rx', 'Markersize', 10);
title('Distribution of $\xi$')
xlabel('$\xi$')
legend('Random sampling', 'Uniform sampling', 'Test values', 'location', 'best')
grid on
xlim([xi1 xi2])
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
str_leg = sprintf(['Full and reduced solution to Poisson equation ($k = %i$, ' ...
    '$n_{\\mu} = %i$, $n_{\\nu} = %i$, $l = %i$)'], ...
    K, Nmu, Nnu, L);
title(str_leg)
xlabel('$x$')
ylabel('$u$')
legend(sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, full', mu(1), nu(1), xi(1)), ...
    sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, reduced (uniform)', mu(1), nu(1), xi(1)), ...
    ... %sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, reduced (random)', mu(1), nu(1), xi(1)), ...
    sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, full', mu(2), nu(2), xi(2)), ...
    sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, reduced (uniform)', mu(2), nu(2), xi(2)), ...
    ... %sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, reduced (random)', mu(2), nu(2), xi(2)), ...
    sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, full', mu(3), nu(3), xi(3)), ...
    sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, reduced (uniform)', mu(3), nu(3), xi(3)), ...
    ... %sprintf('$\\mu = %2.2f$, $\\nu = %2.2f$, $\\xi = %2.2f$, reduced (random)', mu(3), nu(3), xi(3)), ...
    'location', 'best')
grid on

%% A complete sensitivity analysis on the sampling method, the number of
% snapshots and the rank of the reduced basis: plot the maximum and average 
% error versus number of basis functions for different number of snapshots.
% In particular, in each plot we fix the number of samples for $\nu$ and we
% compare the error curves for different numbers of sampled values of $\mu$

%
% User defined settings:
% N     total number of snapshots; in case of uniform sampling, the number
%       of samples along each direction is supposed to be $N^{1/3}$;
%       no more than four values
% L     rank of reduced basis
% Nte   number of testing samples

N = [5 10].^3;  L = 1:25;  Nte = 50;

%
% Run
% 

% Get average error for all values of L and for even distribution 
% of snapshot values of $\mu$, $\nu$ and $\xi$
err_max_unif = zeros(length(L),length(N));
err_avg_unif = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        filename = sprintf(['%s/NonLinearPoisson1d3pSVD/' ...
            'NonLinearPoisson1d3p_%s_%sunif_a%2.2f_b%2.2f_' ...
            '%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'xi1%2.2f_xi2%2.2f_K%i_Nmu%i_Nnu%i_Nxi%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCRt, mu1, mu2, ...
            nu1, nu2, xi1, xi2, K, round(N(j)^(1/3)), round(N(j)^(1/3)), round(N(j)^(1/3)), N(j), ...
            L(i), Nte, suffix);
        load(filename);
        
        err_svd_abs = deleteoutliers(err_svd_abs, 0.01);
        fprintf('Number of outliers: %i\n', Nte-length(err_svd_abs))
        err_max_unif(i,j) = max(err_svd_abs);
        err_avg_unif(i,j) = sum(err_svd_abs)/Nte;
        
        %{
        e = zeros(Nte,1);  dx = (b-a)/(K-1);  x = linspace(a,b,K)';
        for r = 1:Nte
            alpha = VL'*u_te(:,r);
            e(r) = sqrt(dx) * norm(u_te(:,r) - VL*alpha);
        end
        err_max_unif(i,j) = max(e);
        err_avg_unif(i,j) = sum(e)/Nte;
        %}
    end
end
    
%{
% Get average error for all values of L and for random distribution 
% of snapshot values of $\mu$, $\nu$ and $\xi$
err_max_rand = zeros(length(L),length(N));
err_avg_rand = zeros(length(L),length(N));
for i = 1:length(L)
    for j = 1:length(N)
        filename = sprintf(['%s/NonLinearPoisson1d3pSVD/' ...
            'NonLinearPoisson1d3p_%s_%srand_a%2.2f_b%2.2f_' ...
            '%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
            'xi1%2.2f_xi2%2.2f_K%i_Nmu%i_Nnu%i_Nxi%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCRt, mu1, mu2, ...
            nu1, nu2, xi1, xi2, K, N(j), N(j), N(j), N(j), ...
            L(i), Nte, suffix);
        load(filename);
        err_svd_abs = deleteoutliers(err_svd_abs, 0.01);
        fprintf('Number of outliers: %i\n', Nte-length(err_svd_abs))
        err_max_rand(i,j) = max(err_svd_abs);
        err_avg_rand(i,j) = sum(err_svd_abs)/Nte;
    end
end
%}
    
%
% Maximum error
%

% Open a new window
figure(3);
hold off

% Plot data and dynamically update legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
%marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
str_leg = 'legend(''location'', ''best''';
for j = 1:length(N)
    semilogy(L', err_max_unif(:,j), marker_unif{j});
    hold on
    %semilogy(L', err_rand(:,j), marker_rand{j});
    str_unif = sprintf('''$n_{tr} = %i$''', N(j));
    %str_rand = sprintf('''$n_{tr} = %i$, random''', N(j));
    str_leg = strcat(str_leg, ', ', str_unif);
    %str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
end
semilogy(L,s,'ko--')
str_leg = sprintf('%s, ''Singular values'')', str_leg);
eval(str_leg)

% Define plot settings
title('Maximum error in $L^2_h$-norm on test data set')
xlabel('$l$')
ylabel('$||u - u^l||_{L^2_h}$')
grid on    
xlim([min(L)-1 max(L)+1])

%
% Average error
%

% Open a new window
figure(4);
hold off

% Plot data and dynamically update legend
marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
%marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
str_leg = 'legend(''location'', ''best''';
for j = 1:length(N)
    semilogy(L', err_avg_unif(:,j), marker_unif{j});
    hold on
    %semilogy(L', err_rand(:,j), marker_rand{j});
    str_unif = sprintf('''$n_{tr} = %i$''', N(j));
    %str_rand = sprintf('''$n_{tr} = %i$, random''', N(j));
    str_leg = strcat(str_leg, ', ', str_unif);
    %str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
end
semilogy(L,s,'ko--')
str_leg = sprintf('%s, ''Singular values'')', str_leg);
eval(str_leg)

% Define plot settings
title('Average error in $L^2_h$-norm on test data set')
xlabel('$l$')
ylabel('$||u - u^l||_{L^2_h}$')
grid on    
xlim([min(L)-1 max(L)+1])

%% A complete sensitivity analysis on the sampling method, the number of
% snapshots and the rank of the reduced basis: plot the maximum and average
% error versus number of basis functions for different number of snapshots.
% In particular, in each plot we fix the number of samples for $\mu$ and we
% compare the error curves for different numbers of sampled values of $\nu$

%
% User defined settings:
% Nmu   number of sampled values for $\mu$ 
% Nnu   number of sampled values for $\nu$ (no more than four values)
% L     rank of reduced basis
% Nte   number of testing samples

Nmu = [5 15 25 50];  Nnu = [5 15 25 50];  L = 1:25;  Nte = 100;

%
% Run
% 

for k = 1:length(Nmu)
    % Get total number of samples
    N = Nnu * Nmu(k);
    
    % Get error accumulated error for all values of L and for even distribution 
    % of snapshot values of $\mu$
    err_max_unif = zeros(length(L),length(Nnu));
    err_avg_unif = zeros(length(L),length(Nnu));
    for i = 1:length(L)
        for j = 1:length(Nnu)
            filename = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
                'NonLinearPoisson1d2p_%s_%sunif_a%2.2f_b%2.2f_' ...
                '%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
                'Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                root, solver, reducer, a, b, BCLt, BCRt, mu1, mu2, ...
                nu1, nu2, K, Nmu(k), Nnu(j), N(j), L(i), Nte, suffix);
            load(filename);
            err_svd_rel = deleteoutliers(err_svd_rel,0.01);
            err_max_unif(i,j) = max(err_svd_rel);
            err_avg_unif(i,j) = sum(err_svd_rel)/Nte;
        end
    end
    
    %{
    % Get error accumulated error for all values of L and for random distribution 
    % of shapshot values for $\mu$
    err_rand = zeros(length(L),length(Nnu));
    for i = 1:length(L)
        for j = 1:length(Nnu)
            filename = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
                'NonLinearPoisson1d2p_%s_%srand_a%2.2f_b%2.2f_' ...
                '%s_%s_mu1%2.2f_mu2%2.2f_mu1%2.2f_mu2%2.2f_K%i_' ...
                'Nmu%i_Nnu%i_N%i_L%i_Nte%i.mat'], ...
                root, solver, reducer, a, b, BCLt, BCRt, mu1, mu2, ...
                nu1, nu2, K, N(j), N(j), N(j), L(i), Nte);
            load(filename);
            err_svd_rel = deleteoutliers(err_svd_rel,0.01);
            err_max_rand(i,j) = max(err_svd_rel);
            err_avg_rand(i,j) = sum(err_svd_rel)/Nte;
        end
    end
    %}
    
    %
    % Maximum error
    %
    
    % Open a new window
    figure(10+2*k-1);
    hold off

    % Plot data and dynamically update legend
    marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
    %marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
    str_leg = 'legend(''location'', ''best''';
    for j = 1:length(Nmu)
        semilogy(L', err_max_unif(:,j), marker_unif{j});
        hold on
        %semilogy(L', err_rand(:,j), marker_rand{j});
        str_unif = sprintf('''$n_{\\nu} = %i$''', Nnu(j));
        %str_rand = sprintf('''$n_{\\nu} = %i$, random''', Nnu(j));
        str_leg = strcat(str_leg, ', ', str_unif);
        %str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
    end
    semilogy(L,s,'ko--')
    str_leg = sprintf('%s, ''Singular values'')', str_leg);
    eval(str_leg)

    % Define plot settings
    str_leg = sprintf('Maximum relative error $\\epsilon_{max}$ ($k = %i$, $n_{\\mu} = %i$, $n_{te} = %i$)', ...
        K, Nmu(k), Nte);
    title(str_leg)
    xlabel('$l$')
    ylabel('$\epsilon_{max}$')
    grid on    
    xlim([min(L)-1 max(L)+1])
    
    %
    % Average error
    %
    
    % Open a new window
    figure(10+2*k);
    hold off

    % Plot data and dynamically update legend
    marker_unif = {'bo-', 'rs-', 'g^-', 'mv-'};
    %marker_rand = {'bo:', 'rs:', 'g^:', 'mv:'};
    str_leg = 'legend(''location'', ''best''';
    for j = 1:length(Nmu)
        semilogy(L', err_avg_unif(:,j), marker_unif{j});
        hold on
        %semilogy(L', err_rand(:,j), marker_rand{j});
        str_unif = sprintf('''$n_{\\nu} = %i$, uniform''', Nnu(j));
        %str_rand = sprintf('''$n_{\\nu} = %i$, random''', Nnu(j));
        str_leg = strcat(str_leg, ', ', str_unif);
        %str_leg = sprintf('%s, %s, %s', str_leg, str_unif, str_rand);
    end
    semilogy(L,s,'ko--')
    str_leg = sprintf('%s, ''Singular values'')', str_leg);
    eval(str_leg)

    % Define plot settings
    str_leg = sprintf('Average relative error $\\epsilon_{avg}$ ($k = %i$, $n_{\\mu} = %i$, $n_{te} = %i$)', ...
        K, Nmu(k), Nte);
    title(str_leg)
    xlabel('$l$')
    ylabel('$\epsilon_{avg}$')
    grid on    
    xlim([min(L)-1 max(L)+1])
end

%% Fix the number of samples for both $\mu$ and $\nu$ and the rank of the basis
% (the optimal values should have been determined within the previous
% sections), then plot pointwise error

%
% User defined settings:
% Nmu       number of sampled values for $\mu$ 
% Nnu       number of sampled values for $\nu$
% L         rank of reduced basis
% sampler   how the values for $\mu$ and $\nu$ should have been sampled for
%           computing the snapshots:
%           - 'unif': samples form a Cartesian grid
%           - 'rand': drawn from random uniform distribution
% Nte       number of testing samples

Nmu = 50;  Nnu = 5;  L = 15;  sampler = 'unif';  Nte = 100;

%
% Run
%

% Determine total number of training patterns
if strcmp(sampler,'unif')
    N = Nmu*Nnu;
elseif strcmp(sampler,'rand')
    N = Nmu*Nnu;  Nmu = N;  Nnu = N;
end

% Load data
filename = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
    'NonLinearPoisson1d2p_%s_%sunif_a%2.2f_b%2.2f_' ...
    '%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
    'Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, a, b, BCLt, BCRt, mu1, mu2, ...
    nu1, nu2, K, Nmu, Nnu, N, L, Nte, suffix);
load(filename);

% Generate structured data out of scatter data (test samples randomly picked)
[err_svd_rel,idx] = deleteoutliers(err_svd_rel,0.01);
for i = length(idx):-1:1
    mu_te(idx(i)) = [];  nu_te(idx(i)) = [];  
    err_svd_abs(idx(i)) = [];
end
mu_g = linspace(mu1,mu2,1000);  nu_g = linspace(nu1,nu2,1000);
[MU,NU] = meshgrid(mu_g,nu_g);
Ea = griddata(mu_te, nu_te, err_svd_abs, MU, NU);
Er = griddata(mu_te, nu_te, err_svd_rel, MU, NU);

% Plot absolute value
figure(20);
contourf(MU,NU,Ea, 'LineStyle','none')
colorbar
str = sprintf('Absolute error ($k = %i$, $n_{\\mu} = %i$, $n_{\\nu} = %i$, $l = %i$)', ...
    K, Nmu, Nnu, L);
title(str);
xlabel('$\mu$')
ylabel('$\nu$')

% Plot absolute value
figure(21);
contourf(MU,NU,Er, 'LineStyle','none')
colorbar
str = sprintf('Relative error ($k = %i$, $n_{\\mu} = %i$, $n_{\\nu} = %i$, $l = %i$)', ...
    K, Nmu, Nnu, L);
title(str);
xlabel('$\mu$')
ylabel('$\nu$')

%% Plot basis functions

%
% User defined settings:
% Nmu       number of sampled values for $\mu$
% Nnu       number of sampled values for $\nu$
% L         rank of reduced basis
% Nte       number of testing samples
% sampler   how the values for $\mu$ and $\nu$ should have been sampled for
%           computing the snapshots:
%           - 'unif': samples form a Cartesian grid
%           - 'rand': drawn from random uniform distribution
% Nte       number of testing samples

Nmu = 20;  Nnu = 10;  L = 5;  sampler = 'unif';  Nte = 100;

%
% Run
%  

% Load data
N = Nmu*Nnu;
filename = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
    'NonLinearPoisson1d2p_%s_%s%s_a%2.2f_b%2.2f_%s_%s_' ...
    'mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, mu1, mu2, ...
    nu1, nu2, K, Nmu, Nnu, N, L, Nte, suffix);
load(filename);

% Plot basis functions
for l = 1:L
    figure(21+l);
    plot(x,VL(:,l),'b')
    title(sprintf('Basis function $\\psi^{%i}$',l))
    xlabel('$x$')
    ylabel(sprintf('$\\psi^{%i}$',l));
    grid on
end