clc
clear variables
clear variables -global
close all

a = -pi/2;  b = pi/2;  K = 500;

% Suffix ''
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
sampler = {'unif'};
Nmu = [5:10];
Nnu = [5:10];
Nxi = [5:10];
L = 1:25;  
Nte = 100;
root = '../datasets';

dx = (b-a) / (K-1);

for i = 1:length(Nmu)
    for l = 1:length(L)
        filename = sprintf(['%s/NonLinearPoisson1d3pSVD/' ...
            'NonLinearPoisson1d3p_%s_%sunif_a%2.2f_b%2.2f_' ...
            '%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_xi1%2.2f_xi2%2.2f_' ...
            'K%i_Nmu%i_Nnu%i_Nxi%i_N%i_L%i_Nte%i%s.mat'], ...
            root, solver, reducer, a, b, BCLt, BCRt, ...
            mu1, mu2, nu1, nu2, xi1, xi2, K, Nmu(i), Nnu(i), Nxi(i), ...
            Nmu(i)*Nnu(i)*Nxi(i), L(l), Nte, suffix);
        load(filename);
        err_ref_abs = zeros(Nte,1);
        for p = 1:Nte
            err_ref_abs(p) = sqrt(dx)*norm(u_te(:,p) - VL*VL'*u_te(:,p));
        end
        save(filename,'err_ref_abs','-append');
    end
end

%{
BCLt = 'D';  BCLv = @(mu,nu) nu.*(2+sin(mu*a));
BCRt = 'D';  BCRv = @(mu,nu) nu.*(2+sin(mu*b));
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
Nmu = 25;  Nnu = 25;  N = Nmu*Nnu;  Nte_r = 100;  L = 8;
root = '../datasets';

H = 5:2:25;  nruns = 15;
sampler_tr_v = {'unif'};
Nmu_tr_v = [5 10 15 20 25 50];  Nnu_tr_v = [5 10 15 20 25 50];  
valPercentage = 0.3;  Nte_nn = 200;
transferFcn = 'tansig';
trainFcn = {'trainlm'};

for k = 1:length(sampler_tr_v)
    for i = 1:length(Nmu_tr_v)
        %for j = 1:length(Nnu_tr_v)
            Nmu_tr = Nmu_tr_v(i);  Nnu_tr = Nmu_tr_v(i);  
            Ntr = Nmu_tr*Nnu_tr;  Nva = ceil(valPercentage*Ntr);
            filename = sprintf(['%s/NonLinearPoisson1d2pNN/' ...
                'NonLinearPoisson1d2p_%s_%s%s_NN%s_a%2.2f_' ...
                'b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
                'Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler, sampler_tr_v{k}, a, b, BCLt, BCRt, ...
                mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nmu_tr, Nnu_tr, ...
                Ntr, Nva, Nte_nn, suffix);
            load(filename);
            datafile = sprintf(['%s/NonLinearPoisson1d2pSVD/' ...
                'NonLinearPoisson1d2p_%s_%s%s_a%2.2f_' ...
                'b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_' ...
                'Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler, a, b, BCLt, BCRt, ...
                mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, Nte_r, suffix);
            load(datafile);
                        
            for row = 1:size(err_opt_local,1)
                for col = 1:size(err_opt_local,2)
                    net = net_opt_local{row,col};
                    e = 0;
                    for r = 1:Nte_r
                        alpha_nn = net([mu_te(r) nu_te(r)]');
                        e = e + norm(u_te(:,r) - VL*alpha_nn);
                    end
                    err_opt_local(row,col) = e/Nte;
                end
            end
            
            save(filename, 'datafile', 'H', 'nruns', 'trainFcn', 'Ntr', 'Nva', 'Nte', ...
                'err_opt_local', 'net_opt_local', 'tr_opt_local', 'row_opt', 'col_opt');
        %end
    end
end
%}