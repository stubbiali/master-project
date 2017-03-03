% driverLinearPoisson1d2pReduction Given the parametrized, linear one-dimensional
% Poisson equation $-u''(x) = f(x,\mu)$ defined on $[a,b]$ with the right
% boundary condition parametrized in $\nu$, this script performs the following 
% tasks:
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
% nu1       lower bound for $\nu$
% nu2       upper bound for $\nu$
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
%           - 'SVD': Single Value Decomposition
% sampler   how the shapshot values for $\mu$ should be selected:
%           - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%           - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$
% Nmu       number of DIFFERENT sampled values for $\mu$
% Nnu       number of DIFFERENT sampled values for $\nu$
% L         rank of reduced basis
% Nte       number of testing samples
% root      path to folder where storing the output dataset

a = -1;  b = 1;  
K = 100;
%f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  nu1 = 0;  nu2 = 0.5;  suffix = '';
f = @(t,mu) -(t < mu) + 2*(t >= mu);  mu1 = -1;  mu2 = 1;  nu1 = 0;  nu2 = 1;  suffix = '_ter';
BCLt = 'D';  BCLv = 0;
BCRt = 'D';
solver = 'FEP1';
reducer = 'SVD';
sampler = {'unif'};
Nmu_v = [5 10 15 20 25 50]; 
Nnu_v = 1:2;
L = 1:25;  
Nte = 50;
root = '../datasets';

%
% Run
%

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @LinearPoisson1dFEP1_f;
elseif strcmp(solver,'FEP2')
    solverFcn = @LinearPoisson1dFEP2_f;
end

% Set handle to reducer
if strcmp(reducer,'SVD')
    reducerFcn = @getLinearPoisson1d2pSVDreducedBasis;
end

% Testing values for $\mu$ and $\nu$ drawn from a random uniform distribution 
% on $[\mu_1,\mu_2]$ and $[\nu_1,\nu_2]$, respectively
load(strcat(root,'/random_numbers.mat'));
mu_te = mu1 + (mu2-mu1) * random_on_reference_interval_first(1:Nte);
nu_te = nu1 + (nu2-nu1) * random_on_reference_interval_second(1:Nte);

% Evaluate force field for testing values of $\mu$
g_te = cell(Nte,1);
for i = 1:Nte
    g_te{i} = @(t) f(t,mu_te(i));
end

% Compute full solution for testing values of the parameter
for i = 1:Nte
    if i == 1
        [x, y_te] = solverFcn(a, b, K, g_te{1}, BCLt, BCLv, BCRt, nu_te(1));
        u_te = zeros(size(y_te,1),Nte);
        u_te(:,1) = y_te;
    else
        [x, u_te(:,i)] = solverFcn(a, b, K, g_te{i}, BCLt, BCLv, ...
            BCRt, nu_te(i));
    end
end

for k = 1:length(sampler)
    for i = 1:length(Nmu_v)
        for j = 1:length(Nnu_v)
            % Shortcuts
            if strcmp(sampler{k},'unif')
                Nmu = Nmu_v(i);  Nnu = Nnu_v(j);
            elseif strcmp(sampler{k},'rand')
                Nmu = Nmu_v(i)*Nnu_v(j);  Nnu = Nmu;
            end
            
            % Compute snapshots and reduced basis. Since the vectors of the l-rank
            % basis are also the first l vectors of the (l+n)-rank basis, we get
            % the basis for the maximum value of L
            [x, mu_tr, nu_tr, u_tr, s_xl, UL_xl] = reducerFcn(mu1, mu2, Nmu, ...
                nu1, nu2, Nnu, sampler{k}, max(L), ...
                solverFcn, a, b, K, f, BCLt, BCLv, BCRt);

            % Evaluate force field for snapshot values of $\mu$
            g_tr = cell(Nmu,1);
            for p = 1:Nmu
                g_tr{p} = @(t) f(t,mu_tr(p));
            end
                     
            for l = 1:length(L)
                % Compute the coefficients of the expansion of the snapshots in 
                % terms of the reduced basis vectors
                if strcmp(sampler{k},'unif')
                    alpha_tr = zeros(L(l),Nmu*Nnu);
                    for ii = 1:Nmu
                        for jj = 1:Nnu
                            [x, alpha_tr(:,(ii-1)*Nnu+j)] = ...
                                solverFcn(a, b, K, g_tr{ii}, BCLt, BCLv, BCRt, ...
                                nu_tr(jj), UL_xl(:,1:L(l)));
                        end
                    end
                elseif strcmp(sampler{k},'rand')
                    alpha_tr = zeros(L(l),Nmu);
                    for ii = 1:Nmu
                        [x, alpha_tr(:,ii)] = ...
                            solverFcn(a, b, K, g_tr{ii}, BCLt, BCLv, BCRt, ...
                            nu_tr(ii), UL_xl(:,1:L(l)));
                    end
                end

                % Compute reduced solution for testing values of $\mu$ and $\nu$ 
                alpha_te = zeros(L(l),Nte);
                for p = 1:Nte
                    [x, alpha_te(:,p)] = ...
                        solverFcn(a, b, K, g_te{p}, BCLt, BCLv, BCRt, ...
                        nu_te(p), UL_xl(:,1:L(l)));
                end
                ur_te = UL_xl(:,1:L(l)) * alpha_te;

                % Compute error between full and reduced solutions for testing
                % values of $\mu$ and $\nu$
                err_svd_abs = zeros(Nte,1);
                err_svd_rel = zeros(Nte,1);
                for p = 1:Nte
                    err_svd_abs(p) = norm(u_te(:,p) - ur_te(:,p));
                    err_svd_rel(p) = norm(u_te(:,p) - ur_te(:,p)) / norm(u_te(:,p));
                end

                % Save
                if strcmp(sampler{k},'unif')
                    filename = sprintf(['%s/LinearPoisson1d2p_%s_%s%s_a%2.2f_' ...
                        'b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_' ...
                        'nu2%2.2f_K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                        root, solver, reducer, sampler{k}, a, b, BCLt, BCLv, BCRt, ...
                        mu1, mu2, nu1, nu2, K, Nmu, Nnu, Nmu*Nnu, L(l), Nte, suffix);
                elseif strcmp(sampler{k},'rand')
                    filename = sprintf(['%s/LinearPoisson1d2p_%s_%s%s_a%2.2f_' ...
                        'b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_' ...
                        'nu2%2.2f_K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                        root, solver, reducer, sampler{k}, a, b, BCLt, BCLv, BCRt, ...
                        mu1, mu2, nu1, nu2, K, Nmu, Nmu, Nmu, L(l), Nte, suffix);
                end
                UL = UL_xl(:,1:L(l));  s = s_xl(1:L(l));
                save(filename, 'x', 'mu_tr', 'nu_tr', 'u_tr', 's', 'UL', 'alpha_tr', ...
                    'mu_te', 'nu_te', 'u_te', 'alpha_te', 'ur_te', ...
                    'err_svd_abs', 'err_svd_rel');
            end
        end
    end
end