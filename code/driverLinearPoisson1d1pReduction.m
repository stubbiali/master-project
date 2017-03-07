% driverLinearPoisson1d1pReduction Given the parametrized, linear one-dimensional
% Poisson equation $-u''(x) = f(x,\mu)$ defined on $[a,b]$, this script 
% performs the following tasks:
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
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$
% N         number of snapshots
% L         rank of reduced basis
% Nte       number of testing values for $\mu$
% root      path to folder where storing the output dataset

a = -1;  b = 1;  
K = 100;
%f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  suffix = '';
%f = @(t,mu) 50 * t .* cos(mu*pi*t);  mu1 = 1;  mu2 = 3;  suffix = '_bis';
f = @(t,mu) -(t < mu) + 2*(t >= mu);  mu1 = -1;  mu2 = 1;  suffix = '_ter';

BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
sampler = {'unif', 'rand'};
N = [5 10 15 20 25 50 75 100]; 
L = 1:25;  
Nte = 50;
root = '../datasets';

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @LinearPoisson1dFEP1;
elseif strcmp(solver,'FEP2')
    solverFcn = @LinearPoisson1dFEP2;
end

% Set handle to reducer
if strcmp(reducer,'SVD')
    reducerFcn = @getLinearPoisson1d1pSVDreducedBasis;
end

% Testing values for $\mu$ drawn from a random uniform distribution on 
% $[\mu_1,\mu_2]$
load(strcat(root,'/random_numbers.mat'));
mu_te = mu1 + (mu2-mu1) * random_on_reference_interval_first(1:Nte);

% Evaluate force field for testing values of $\mu$
g_te = cell(Nte,1);
for i = 1:Nte
    g_te{i} = @(t) f(t,mu_te(i));
end

for k = 1:length(sampler)
    for n = 1:length(N)
        % Compute snapshots and reduced basis. Since the vectors of the l-rank
        % basis are also the first l vectors of the (l+n)-rank basis, we get
        % the basis for the maximum value of L
        [x, mu_tr, u_tr, s_xl, UL_xl] = reducerFcn(mu1, mu2, N(n), sampler{k}, max(L), ...
            solverFcn, a, b, K, f, BCLt, BCLv, BCRt, BCRv);

        % Evaluate force field for snapshot values of $\mu$
        g_tr = cell(N(n),1);
        for i = 1:N(n)
            g_tr{i} = @(t) f(t,mu_tr(i));
        end

        % Compute full solution for testing values of the parameter
        [x, u_te] = solverFcn(a, b, K, g_te, BCLt, BCLv, BCRt, BCRv);

        for l = 1:length(L)
            % Compute the coefficients of the expansion of the snapshots in 
            % terms of the reduced basis vectors
            [x, alpha_tr] = solverFcn(a, b, K, g_tr, BCLt, BCLv, BCRt, BCRv, UL_xl(:,1:L(l)));

            % Compute reduced solution for testing values of $\mu$
            [x, alpha_te] = solverFcn(a, b, K, g_te, BCLt, BCLv, BCRt, BCRv, UL_xl(:,1:L(l)));
            ur_te = UL_xl(:,1:L(l)) * alpha_te;

            % Compute error between full and reduced solutions for testing
            % values of $\mu$
            err_svd_abs = zeros(Nte,1);
            err_svd_rel = zeros(Nte,1);
            for i = 1:Nte
                err_svd_abs(i) = norm(u_te(:,i) - ur_te(:,i));
                err_svd_rel(i) = norm(u_te(:,i) - ur_te(:,i)) / norm(u_te(:,i));
            end

            % Save
            filename = sprintf(['%s/LinearPoisson1d1pSVD/' ...
                'LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_' ...
                '%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler{k}, a, b, BCLt, BCLv, BCRt, ...
                BCRv, mu1, mu2, K, N(n), L(l), Nte, suffix);
            UL = UL_xl(:,1:L(l));   s = s_xl(1:L(l));
            save(filename, 'x', 'mu_tr', 'u_tr', 's', 'UL', 'alpha_tr', ...
                'mu_te', 'u_te', 'alpha_te', 'ur_te', 'err_svd_abs', 'err_svd_rel');
        end
    end
end
