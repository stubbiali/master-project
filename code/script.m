clc
clear variables 
clear variables -global
close all

a = -1;  b = 1;  
K = 100;
f = @(t,mu) gaussian(t,mu,0.2);  
mu1 = -1;  mu2 = 1;
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
N = [10 25 50 75 100];  L = 1:25;  Nte = 50;
root = '../datasets';

sampler_tr = 'unif';
Ntr = 10;  Nva = 0.25 * Ntr;

filename = sprintf('%s/LinearPoisson1d1p_%s_%s%s_NN%s_a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_Nva%i_Nte%i.mat', ...
    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, 50, 10, Ntr, Nva, Nte);
load(filename);
mu_te_new = mu_te;

if strcmp(solver,'FEP1')
    solverFcn = @LinearPoisson1dFEP1;
elseif strcmp(solver,'FEP2')
    solverFcn = @LinearPoisson1dFEP2;
end

g_te = cell(Nte,1);
for i = 1:Nte
    g_te{i} = @(t) f(t,mu_te(i));
end

for i = 1:length(N)
    for j = 1:length(L)
        datafile = sprintf('%s/LinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i.mat', ...
            root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N(i), L(j), Nte);
        load(datafile);
        
        mu_te = mu_te_new;
        [x, u_te, alpha_te] = solverFcn(a, b, K, g_te, BCLt, BCLv, BCRt, BCRv, UL);
        ur_te = UL * alpha_te;

        err_svd_abs = zeros(Nte,1);
        err_svd_rel = zeros(Nte,1);
        for k = 1:Nte
            err_svd_abs(k) = norm(u_te(:,k) - ur_te(:,k));
            err_svd_rel(k) = norm(u_te(:,k) - ur_te(:,k)) / norm(u_te(:,k));
        end

        save(datafile, 'x', 'mu_tr', 'u_tr', 'UL', 'alpha_tr', 'mu_te', 'u_te', 'alpha_te', 'ur_te', 'err_svd_abs', 'err_svd_rel');
    end
end
        