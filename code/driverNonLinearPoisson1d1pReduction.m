% driverNonLinearPoisson1d1pReduction Consider the parametrized, nonlinear 
% one-dimensional Poisson equation $-(v(u) u'(x))' = f(x,\mu)$ in the
% unknown $u = u(x)$, $x \in [a,b]$, with $\mu \in [\mu_1,\mu_2]$. The ODE
% should be completed with Dirichlet, Neumann or periodic boundary conditions. 
% This script performs the following tasks:
% - computing a discrete solution (snapshot) for different values of $\mu$
% - build a reduced basis out of the ensemble of snapshots
% - solve the full and reduced discrete model for other values of $\mu$
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
%           - 'FEP1Newton': linear finite elements coupled with Newton's
%           method to solve the nonlinear system
% reducer   method to compute the reduced basis
%           - 'SVD': Single Value Decomposition
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$
% N         number of snapshots
% L         rank of reduced basis
% Nte       number of testing values for $\mu$
% root      path to folder where storing the output dataset

a = -1;  b = 1;  
K = 100;
v = @(u) 1./(u.^2);  dv = @(u) - 2./(u.^3);
f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  suffix = '';
BCLt = 'D';  BCLv = 1;
BCRt = 'D';  BCRv = 1;
solver = 'FEP1Newton';
reducer = 'SVD';
sampler = {'unif'};
N = 25; 
L = 1:25;  
Nte = 100;
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
    reducerFcn = @getNonLinearPoisson1d1pSVDreducedBasis;
end

% Testing values for $\mu$ drawn from a random uniform distribution on 
% $[\mu_1,\mu_2]$
load(strcat(root,'/random_numbers.mat'));
mu_te = mu1 + (mu2-mu1) * random_on_reference_interval_first(1:Nte);

% Evaluate force field for testing values of $\mu$
g_te = cell(Nte,1);
parfor i = 1:Nte
    g_te{i} = @(t) f(t,mu_te(i));
end

% Compute full solution for testing values of the parameter
[x,y] = solverFcn(a, b, K, v, dv, g_te{1}, BCLt, BCLv, ...
    BCRt, BCRv);
u_te = zeros(size(y,1),Nte);  u_te(:,1) = y;
parfor i = 1:Nte
    [x,u_te(:,i)] = solverFcn(a, b, K, v, dv, g_te{i}, BCLt, BCLv, ...
        BCRt, BCRv);
end

for k = 1:length(sampler)
    for n = 1:length(N)
        % Compute snapshots and reduced basis. Since the vectors of the l-rank
        % basis are also the first l vectors of the (l+n)-rank basis, we get
        % the basis for the maximum value of L
        [x, mu_tr, u_tr, s_xl, V_xl] = reducerFcn(mu1, mu2, N(n), sampler{k}, max(L), ...
            solverFcn, a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv);

        % Evaluate force field for snapshot values of $\mu$
        g_tr = cell(N(n),1);
        parfor i = 1:N(n)
            g_tr{i} = @(t) f(t,mu_tr(i));
        end

        for l = 1:length(L)
            % Shortcuts
            VL = V_xl(:,1:L(l));   s = s_xl(1:L(l));
            
            % Compute reduced solution for the values of $\mu$ employed for
            % the computation of the snapshots
            alpha_tr = zeros(L(l),N(n));
            parfor i = 1:N(n)
                [x,alpha_tr(:,i)] = rsolverFcn(a, b, K, v, dv, g_tr{i}, ...
                    BCLt, BCLv, BCRt, BCRv, VL);
            end

            % Compute reduced solution for testing values of $\mu$
            alpha_te = zeros(L(l),Nte);
            parfor i = 1:Nte
                [x,alpha_te(:,i)] = rsolverFcn(a, b, K, v, dv, g_te{i}, ...
                    BCLt, BCLv, BCRt, BCRv, VL);
            end
            ur_te = V_xl(:,1:L(l)) * alpha_te;

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
            filename = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
                'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_' ...
                '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler{k}, a, b, BCLt, BCLv, BCRt, ...
                BCRv, mu1, mu2, K, N(n), L(l), Nte, suffix);
            save(filename, 'x', 'mu_tr', 'u_tr', 's', 'VL', 'alpha_tr', ...
                'mu_te', 'u_te', 'alpha_te', 'ur_te', 'err_svd_abs', 'err_svd_rel');
        end
    end
end