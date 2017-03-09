clc
clear variables
clear variables -global 
close all

a = -1;  b = 1;  K = 100;
%v = @(t,nu) 1*(t < -0.5) + nu*(-0.5 <= t & t <= 0.3) + 0.25*(t > 0.3);  nu1 = 1;  nu2 = 3;
v = @(t,nu) 1 + (t+1).^nu;  nu1 = 1;  nu2 = 3;
%f = @(t,mu) gaussian(t,mu,0.1);  mu1 = -1;  mu2 = 1;
f = @(t,mu) - 1*(t < mu) + 2*(t >= mu);  mu1 = -1;  mu2 = 1;
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
sampler = {'unif'};
Nmu_v = [5 50]; 
Nnu_v = [5 50]; 
L = 1:25;  
Nte = 50;
root = '../datasets';
suffix = '_bis';

% Set handle to solver
if strcmp(solver,'FEP1')
    solverFcn = @HeterogeneousViscosityLinearPoisson1dFEP1_f;
elseif strcmp(solver,'FEP2')
    solverFcn = @HeterogeneousViscosityLinearPoisson1dFEP2_f;
end

% Testing values for $\mu$ and $\nu$ drawn from a random uniform distribution 
% on $[\mu_1,\mu_2]$ and $[\nu_1,\nu_2]$, respectively
load(strcat(root,'/random_numbers.mat'));
mu_te = mu1 + (mu2-mu1) * random_on_reference_interval_first(1:Nte);
nu_te = nu1 + (nu2-nu1) * random_on_reference_interval_second(1:Nte);

% Evaluate viscosity for testing values of $\nu$
vis_te = cell(Nte,1);
for i = 1:Nte
    vis_te{i} = @(t) v(t,nu_te(i));
end

% Evaluate force field for testing values of $\mu$
g_te = cell(Nte,1);
for i = 1:Nte
    g_te{i} = @(t) f(t,mu_te(i));
end

% Compute full solution for testing values of the parameters
for i = 1:Nte
    if i == 1
        [x,y_te] = solverFcn(a, b, K, vis_te{1}, g_te{1}, BCLt, BCLv, BCRt, BCRv);
        uu_te = zeros(size(y_te,1),Nte);
        uu_te(:,1) = y_te;
    else
        [x,uu_te(:,i)] = solverFcn(a, b, K, vis_te{i}, g_te{i}, BCLt, BCLv, BCRt, BCRv);
    end
end

for k = 1:length(sampler)
    for i = 1:length(Nmu_v)
        for j = 1:length(Nnu_v)
            for l = 1:length(L)
                if strcmp(sampler{k},'unif')
                    filename = sprintf(['%s/HeterogeneousViscosityLinearPoisson1d2pSVD/' ...
                        'HeterogeneousViscosityLinearPoisson1d2p_%s_%s%s_a%2.2f_' ...
                        'b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_nu1%2.2f_' ...
                        'nu2%2.2f_K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                        root, solver, reducer, sampler{k}, a, b, BCLt, BCLv, BCRt, BCRv, ...
                        mu1, mu2, nu1, nu2, K, Nmu_v(i), Nnu_v(j), Nmu_v(i)*Nnu_v(j), ...
                        L(l), Nte, suffix);
                elseif strcmp(sampler{k},'rand')
                    filename = sprintf(['%s/HeterogeneousViscosityLinearPoisson1d2pSVD/' ...
                        'HeterogeneousViscosityLinearPoisson1d2p_%s_%s%s_a%2.2f_' ...
                        'b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_nu1%2.2f_' ...
                        'nu2%2.2f_K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                        root, solver, reducer, sampler{k}, a, b, BCLt, BCLv, BCRt, BCRv, ...
                        mu1, mu2, nu1, nu2, K, Nmu_v(i), Nmu_v(j), Nmu_v(i), L(l), Nte, suffix);
                end
                
                load(filename);
                
                u_te = uu_te;
                err_svd_abs = zeros(Nte,1);
                err_svd_rel = zeros(Nte,1);
                for p = 1:Nte
                    err_svd_abs(p) = norm(u_te(:,p) - ur_te(:,p));
                    err_svd_rel(p) = norm(u_te(:,p) - ur_te(:,p)) / norm(u_te(:,p));
                end
                
                save(filename, 'u_te', 'err_svd_abs', 'err_svd_rel', '-append');
            end
        end
    end
end