% driverNonLinearPoisson1d2pReduction Consider the parametrized, nonlinear 
% one-dimensional Poisson equation $-(v(u) u'(x))' = f(x,\mu,\nu)$ in the
% unknown $u = u(x)$, $x \in [a,b]$, with $\mu$ and $\nu$ real parameters. 
% The ODE should be completed with Dirichlet, Neumann or periodic boundary 
% conditions, whose might imply the parameters. 
% This script performs the following tasks:
% - computing a discrete solution (snapshot) for different values of $\mu$
%   and $\nu$
% - build a reduced basis out of the ensemble of snapshots
% - solve the full and reduced discrete model for other values of $\mu$ 
%   and $\nu$
% - compute the error between full and reduced solution
% - store the results in a Matlab dataset

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
% f         force field $f = f(t,\mu)$ as handle function
% mu1       lower bound for $\mu$
% mu2       upper bound for $\mu$
% nu1       lower bound for $\nu$
% nu2       upper bound for $\nu$
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
%           - 'FEP1Newton': linear finite elements coupled with Newton's
%           method to solve the nonlinear system
% reducer   method to compute the reduced basis
%           - 'SVD': Single Value Decomposition
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed
%           - 'rand': drawn from a uniform random distribution
% Nmu       number of samples for $\mu$
% Nnu       number of samples for $\nu$
% L         rank of reduced basis
% Nte       number of testing samples
% root      path to folder where storing the output dataset

a = -pi;  b = pi;  
K = 100;
%v = @(u) exp(-u);  dv = @(u) -exp(-u);
v = @(u) u.^2;  dv = @(u) 2*u;
%f = @(t,mu,nu) nu*mu*mu*exp(nu*sin(mu*t)).*(sin(mu*t) - nu.*cos(mu*t).^2);
%f = @(t,mu,nu) nu*mu*mu*exp(-nu*sin(mu*t)).*(sin(mu*t) + nu.*cos(mu*t).^2);
f = @(t,mu,nu) nu.*nu.*mu.*mu.*(2+sin(mu.*t)).*(-2*nu.*cos(mu.*t).^2 + ...
    2*nu.*sin(mu.*t) + nu.*sin(mu.*t).^2);
mu1 = 1;  mu2 = 3;  nu1 = 1;  nu2 = 3;  suffix = '';
BCLt = 'D';  BCLv = @(mu,nu) nu.*(2+sin(mu*a));
BCRt = 'D';  BCRv = @(mu,nu) nu.*(2+sin(mu*b));
solver = 'FEP1Newton';
reducer = 'SVD';
sampler = {'unif'};
Nmu = [5 10 15 20 25 50];
Nnu = [5 10 15 20 25 50];
L = 1:25;  
Nte = 50;
root = '../datasets';

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1Newton')
    solverFcn = @NonLinearPoisson1dFEP1Newton;
    rsolverFcn = @NonLinearPoisson1dFEP1ReducedNewton;
end

% Set handle to reducer
if strcmp(reducer,'SVD')
    reducerFcn = @getNonLinearPoisson1d2pSVDreducedBasis;
end

% Testing values for $\mu$ and $\nu $drawn from a random uniform distribution on 
% $[\mu_1,\mu_2] \times [\nu_1,\nu_2]$
load(strcat(root,'/random_numbers.mat'));
mu_te = mu1 + (mu2-mu1) * random_on_reference_interval_first(1:Nte);
nu_te = nu1 + (nu2-nu1) * random_on_reference_interval_second(1:Nte);

% Evaluate force field for testing values of $\mu$ and $\nu$
g_te = cell(Nte,1);
parfor i = 1:Nte
    g_te{i} = @(t) f(t,mu_te(i),nu_te(i));
end

% Compute full solution for testing values of the parameter
fprintf('i = 1\n');
[x,y] = solverFcn(a, b, K, v, dv, g_te{1}, BCLt, BCLv(mu_te(1),nu_te(1)), ...
    BCRt, BCRv(mu_te(1),nu_te(1)));
u_te = zeros(size(y,1),Nte);  u_te(:,1) = y;
parfor i = 2:Nte
    fprintf('i = %i\n',i);
    [x,u_te(:,i)] = solverFcn(a, b, K, v, dv, g_te{i}, BCLt, BCLv(mu_te(i),nu_te(i)), ...
        BCRt, BCRv(mu_te(i),nu_te(i)));
end

for k = 1:length(sampler)
    % Determine total number of samples
    if strcmp(sampler{k},'unif')
        row = length(Nmu);
        Nmu = repmat(Nmu,[length(Nnu),1]);  Nmu = Nmu(:);  
        Nnu = repmat(Nnu',[row,1]);
    end
    N = Nmu.*Nnu;
    
    for n = 1:length(N)
        % Compute snapshots and reduced basis. Since the vectors of the l-rank
        % basis are also the first l vectors of the (l+n)-rank basis, we get
        % the basis for the maximum value of L
        [x, mu_tr, nu_tr, u_tr, s_xl, V_xl] = reducerFcn(mu1, mu2, nu1, nu2, ...
            sampler{k}, Nmu(n), Nnu(n), max(L), ...
            solverFcn, a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv);

        % Evaluate force field for samples of $\mu$ and $\nu$
        g_tr = cell(N(n),1);
        parfor i = 1:N(n)
            g_tr{i} = @(t) f(t,mu_tr(i),nu_tr(i));
        end

        for l = 1:length(L)
            % Shortcuts
            VL = V_xl(:,1:L(l));   s = s_xl(1:L(l));
            
            % Compute reduced solution for the values of $\mu$ and $\nu$
            % employed for the computation of the snapshots
            alpha_tr = zeros(L(l),N(n));
            parfor i = 1:N(n)
                [x,alpha_tr(:,i)] = rsolverFcn(a, b, K, v, dv, g_tr{i}, ...
                    BCLt, BCLv(mu_tr(i),nu_tr(i)), ...
                    BCRt, BCRv(mu_tr(i),nu_tr(i)), VL);
            end

            % Compute reduced solution for testing values of $\mu$
            alpha_te = zeros(L(l),Nte);
            parfor i = 1:Nte
                [x,alpha_te(:,i)] = rsolverFcn(a, b, K, v, dv, g_te{i}, ...
                    BCLt, BCLv(mu_te(i),nu_te(i)), ...
                    BCRt, BCRv(mu_te(i),nu_te(i)), VL);
            end
            ur_te = VL * alpha_te;

            % Compute error between full and reduced solutions for testing
            % values of $\mu$
            h = (b-a) / (K-1);
            err_svd_abs = zeros(Nte,1);
            err_svd_rel = zeros(Nte,1);
            parfor i = 1:Nte
                err_svd_abs(i) = sqrt(h * norm(u_te(:,i) - ur_te(:,i))^2);
                err_svd_rel(i) = norm(u_te(:,i) - ur_te(:,i)) / norm(u_te(:,i));
            end

            % Save
            filename = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
                'NonLinearPoisson1d2p_%s_%s%s_a%2.2f_b%2.2f_' ...
                '%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler{k}, a, b, BCLt, BCRt, ...
                mu1, mu2, nu1, nu2, K, Nmu(n), Nnu(n), N(n), L(l), Nte, suffix);
            save(filename, 'x', 'mu_tr', 'nu_tr', 'u_tr', 's', 'VL', 'alpha_tr', ...
                'mu_te', 'nu_te', 'u_te', 'alpha_te', 'ur_te', 'err_svd_abs', 'err_svd_rel');
        end
    end
end