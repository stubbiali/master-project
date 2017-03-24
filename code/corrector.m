clc
clear variables
clear variables -global
close all

a = -1;  b = 1;  K = 100;

% Suffix ''
%v = @(t,nu) 1*(t < -0.5) + nu*(-0.5 <= t & t <= 0.3) + 0.25*(t > 0.3);  
%f = @(t,mu) gaussian(t,mu,0.1); 
%mu1 = -1;  mu2 = 1;  nu1 = 1;  nu2 = 3;  suffix = '';
% Suffix '_bis'
%v = @(t,nu) 1 + (t+1).^nu;  
%f = @(t,mu) - 1*(t < mu) + 2*(t >= mu);  
%mu1 = -1;  mu2 = 1;  nu1 = 1;  nu2 = 3;  suffix = '_bis';
% Suffix '_ter'
%v = @(t,nu) 2 + sin(nu*pi*t);
%f = @(t,mu) - 1*(t < mu) + 2*(t >= mu);  
%mu1 = -1;  mu2 = 1;  nu1 = 1;  nu2 = 3;  suffix = '_ter';
% Suffix '_quat'
v = @(t,nu) 1*(t < -0.5) + nu*(-0.5 <= t & t <= 0.5) + 1*(t > 0.5);  
f = @(t,mu) sin(mu*pi*(t+1));  
mu1 = 1;  mu2 = 3;  nu1 = 1; nu2 = 5;  suffix = '_quat';

BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
Nmu = 50;  Nnu = 25;  N = Nmu*Nnu;  L = 12;  Nte_r = 50;
root = '../datasets';

H = 19;  nruns = 1;
sampler_tr_v = {'unif'};
Nmu_tr_v = [5 10 15 20 30 40 50];  Nnu_tr_v = [5 10 15 20 30 40 50]; 
valPercentage = 0.3;  Nte_nn = 200;
transferFcn = 'tansig';
trainFcn = {'trainlm'};
showWindow = false;
tosave = true;

for k = 1:length(sampler_tr_v)
    for i = 1:length(Nmu_tr_v)
        for j = 1:length(Nnu_tr_v)
            Nmu_tr = Nmu_tr_v(i);  Nnu_tr = Nnu_tr_v(j);  
            Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);
            filename = sprintf(['%s/HeterogeneousViscosityLinearPoisson1d2pNN/' ...
                'HeterogeneousViscosityLinearPoisson1d2p_%s_%s%s_NN%s_a%2.2f_' ...
                'b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
                'Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler, sampler_tr_v{k}, a, b, BCLt, BCLv, ...
                BCRt, BCRv, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nmu_tr, Nnu_tr, ...
                Ntr, Nva, Nte_nn, suffix);
            load(filename);
            datafile = sprintf(['%s/HeterogeneousViscosityLinearPoisson1d2pSVD/' ...
                'HeterogeneousViscosityLinearPoisson1d2p_%s_%s%s_a%2.2f_' ...
                'b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
                'Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler, a, b, BCLt, BCLv, ...
                BCRt, BCRv, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nte_r, suffix);
            load(datafile);
            
            for row = 1:size(err_opt_local,1)
                for col = 1:size(err_opt_local,2)
                    net = net_opt_local{row,col};
                    e = 0;
                    for r = 1:Nte_r
                        alpha_nn = net([mu_te(r) nu_te(r)]');
                        e = e + norm(u_te(:,r) - UL*alpha_nn);
                    end
                    err_opt_local(row,col) = e/Nte_r;
                end
            end
            
            save(filename, 'datafile', 'H', 'nruns', 'trainFcn', 'Ntr', 'Nva', 'Nte', ...
                'err_opt_local', 'net_opt_local', 'tr_opt_local', 'row_opt', 'col_opt');
        end
    end
end