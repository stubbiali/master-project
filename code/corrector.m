clc
clear variables
clear variables -global
close all

a = -1;  b = 1;  K = 100;

% Suffix '_bis'
v = @(t) 1 ./ (t.^2);  dv = @(t) - 2 ./ (t.^3);
f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  suffix = '_bis';

BCLt = 'D';  BCLv = 1;
BCRt = 'D';  BCRv = 1;
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
N = 50;  L = 12;  Nte = 50;
root = '../datasets';

H = 1:25;  nruns = 10;
sampler_tr_v = {'unif'};
Ntr_v = [75];  Nva_v = ceil(0.3 * Ntr_v);  Nte_nn = 100;
transferFcn = 'tansig';
trainFcn = {'trainlm'};
showWindow = false;
tosave = true;

for i = 1:length(sampler_tr_v)
    for j = 1:length(Ntr_v)
        filename = sprintf(['%s/NonLinearPoisson1d1pNN/' ...
            'NonLinearPoisson1d1p_%s_%s%s_NN%s_a%2.2f_' ...
            'b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_' ...
            'Nva%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, sampler_tr_v{i}, a, b, BCLt, BCLv, ...
            BCRt, BCRv, mu1, mu2, K, N, L, Ntr_v(j), Nva_v(j), Nte_nn, suffix);
        load(filename);
        datafile = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
            'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_' ...
            'b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, sampler, a, b, BCLt, BCLv, ...
            BCRt, BCRv, mu1, mu2, K, N, L, Nte, suffix);
        load(datafile);
        %{
        for row = 1:size(err_opt_local,1)
            for col = 1:size(err_opt_local,2)
                net = net_opt_local{row,col};
                e = 0;
                for k = 1:Nte
                    alpha_nn = net(mu_te(k));
                    e = e + norm(u_te(:,k) - UL*alpha_nn);
                end
                err_opt_local(row,col) = e/Nte;
            end
        end
        %}
        err_opt_local = 0.5*err_opt_local;
        save(filename, 'datafile', 'H', 'nruns', 'trainFcn', 'Ntr', 'Nva', 'Nte', ...
            'err_opt_local', 'net_opt_local', 'tr_opt_local', 'row_opt', 'col_opt');
    end
end