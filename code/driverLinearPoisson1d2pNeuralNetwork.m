% driverLinearPoisson1d2pNeuralNetwork Consider the one-dimensional linear
% equation $-u''(x) = f(x,\mu)$ on $[a,b]$ depending on the real parameters
% $\mu$ and $\nu$, where the latter represents the value fo the boundary 
% condition associated with the right end-point of the interval. Given a 
% reduced basis $U$ of rank L and some training reduced solutions, we train 
% a neural network to approximate the nonlinear map
% $[\mu,\nu] \mapsto \boldsymbol{\alpha}$, where $\boldsymbol{\alpha}$ are 
% the coefficients of the expansion of the reduced solution in terms of the
% reduced basis vectors.

clc
clear variables 
clear variables -global
close all

%
% User-defined settings:
% a             left boundary of the domain
% b             right boundary of the domain
% K             number of grid points
% f             force field $f = f(t,\mu)$ as handle function
% mu1           lower bound for $\mu$
% mu2           upper bound for $\mu$
% nu1           lower bound for $\nu$
% nu2           upper bound for $\nu$
% suffix        suffix used for data file name:
%               - '': f(x,mu) = @(x) gaussian(x,mu,0.2) with 
%               mu1 = -1, mu2 = 1, nu1 = 0, nu2 = 0.5
%               - '_ter': f(x,mu) = @(x) -(x < mu) + 2*(x > mu) with
%               mu1 = -1, mu2 = 1, nu1 = 0, nu2 = 1
% BCLt          kind of left boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% BCLv          value of left boundary condition
% BCRt          kind of right boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% solver        solver
%               - 'FEP1': linear finite elements
%               - 'FEP2': quadratic finite elements
% reducer       method to compute the reduced basis
%               - 'SVD': Single Value Decomposition 
% sampler       how the values for $\mu$ amd $\nu$ should have been sampled 
%               for the computation of the reduced basis:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2] \times [\nu_1,\nu_2]$
%               - 'rand': drawn from a uniform random distribution on 
%               $[\mu_1,\mu_2] \times [\nu_1,\nu_2]$
% Nmu           number of snapshot values for $\mu$
% Nnu           number of snapshot values for $\nu$
% L             rank of reduced basis
% sampler_tr    how the training values for $\mu$ should be selected:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$
% Nmu_tr_v      number of training patterns for $\mu$
% Nnu_tr_v      number of training patterns for $\nu$
% Nte           number of testing values for $\mu$ and $\nu$
% valPercentage ratio between number of validation and training values for 
%               $\mu$ and $\nu$
% root          path to folder where storing the output dataset
% H             number of hidden neurons
% nruns         number of re-training for each network architecture
% transferFcn   transfer function
%               - 'logsig': logarithimc sigmoid
%               - 'tansig': tangential sigmoid
% trainFcn      training algorithm
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization
% showWindow    TRUE to open training window, FALSE otherwise
% tosave        TRUE to store the results in a Matlab dataset, FALSE otherwise

a = -1;  b = 1;  K = 100;
%f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  nu1 = 0; nu2 = 0.5; suffix = '';
f = @(t,mu) -(t < mu) + 2*(t >= mu);  mu1 = -1;  mu2 = 1;  nu1 = 0;  nu2 = 1;  suffix = '_ter';
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
Nmu = 50;  Nnu = 10;  N = Nmu*Nnu;  L = 10; 
root = '../datasets';

H = 1:25;  nruns = 10;
sampler_tr_v = {'unif'};
Nmu_tr_v = [5 10 15 20 25 50];  Nnu_tr_v = [5 10 15 20 25 50];  
valPercentage = 0.3;  Nte = 50;
transferFcn = 'tansig';
trainFcn = {'trainlm'};
showWindow = false;
tosave = true;

%
% Run
%

% Set handle to solver function
if strcmp(solver,'FEP1')
    solverFcn = @LinearPoisson1dFEP1_f;
elseif strcmp(solver,'FEP2')
    solverFcn = @LinearPoisson1dFEP2_f;
end

% Determine total number of training and validating samples
Ntr_v = zeros(length(Nmu_tr_v)*length(Nnu_tr_v),1);
for i = 1:length(Nmu_tr_v)
    for j = 1:length(Nnu_tr_v)
        Ntr_v((i-1)*length(Nnu_tr_v)+j) = Nmu_tr_v(i)*Nnu_tr_v(j);
    end
end

% Load data for training
datafile = sprintf(['%s/LinearPoisson1d2pSVD/' ...
    'LinearPoisson1d2p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_K%i_Nmu%i_Nnu%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, mu1, mu2, ...
    nu1, nu2, K, Nmu, Nnu, N, L, Nte, suffix);
load(datafile);

%
% Create validation data set; its size is usually dependent on the number
% of training patterns
%

% Determing number of validation patterns
Nva_v = ceil(valPercentage * Ntr_v);

% Determine validation values for $\mu$ and $\nu$
mu_va = mu1 + (mu2-mu1) * rand(max(Nva_v),1);
nu_va = nu1 + (nu2-nu1) * rand(max(Nva_v),1);

% Evaluate force field at the validation values
g_va = cell(max(Nva_v),1);
for i = 1:max(Nva_v)
    g_va{i} = @(t) f(t,mu_va(i));
end

% Compute reduced solution
alpha_va = zeros(L,max(Nva_v));
for i = 1:max(Nva_v)
    [x,alpha_va(:,i)] = solverFcn(a, b, K, g_va{i}, BCLt, ...
        BCLv, BCRt, nu_va(i), UL);
end

%
% If needed, extend test data set up to size 200
%

Nte_opt = 200;

if (Nte < Nte_opt)
    % Load random data
    load(strcat(root,'/random_numbers.mat'));
    
    % Determine values for $\mu$ and $\nu$
    mu_te = mu1 + (mu2-mu1) * random_on_reference_interval_first(1:Nte_opt);
    nu_te = nu1 + (nu2-nu1) * random_on_reference_interval_second(1:Nte_opt);

    % Evaluate force field at the test values
    g_te = cell(Nte_opt,1);
    for i = 1:Nte_opt
        g_te{i} = @(t) f(t,mu_te(i));
    end

    % Compute reduced solution
    alpha_te = zeros(L,Nte_opt);
    for i = 1:Nte_opt
        if i == 1
            [x,u,alpha_te(:,i)] = solverFcn(a, b, K, g_te{i}, BCLt, ...
                BCLv, BCRt, nu_te(i), UL);
            u_te = zeros(size(u,1),Nte_opt);  u_te(:,1) = u;
        else
            [x,u_te(:,i),alpha_te(:,i)] = solverFcn(a, b, K, g_te{i}, BCLt, ...
                BCLv, BCRt, nu_te(i), UL);
        end
    end
    
    % Set Nte
    Nte = Nte_opt;
end


%
% Train the neural network
%

for s = 1:length(sampler_tr_v)
    for i = 1:length(Nmu_tr_v)
        for j = 1:length(Nnu_tr_v)
            % Shortcuts
            n = (i-1)*length(Nnu_tr_v)+j;
            Ntr = Ntr_v(n);   Nva = Nva_v(n);  sampler_tr = sampler_tr_v{s};
            if (strcmp(sampler_tr,'unif'))
                Nmu_tr = Nmu_tr_v(i);  Nnu_tr = Nnu_tr_v(j);
            elseif (strcmp(sampler_tr,'rand'))
                Nmu_tr = Ntr;  Nnu_tr = Ntr;
            end            

            % Define ratio for training, validation and test data sets
            trainRatio = Ntr / (Ntr + Nva + Nte);  
            valRatio = Nva / (Ntr + Nva + Nte);
            testRatio = 1 - trainRatio - valRatio;

            %
            % Compute training patterns and teaching inputs
            %

            % Get training patterns
            if (strcmp(sampler_tr,'unif'))
                mu_tr = linspace(mu1, mu2, Nmu_tr);
                mu_tr = repmat(mu_tr, Nnu_tr, 1);  mu_tr = mu_tr(:);
                nu_tr = linspace(nu1, nu2, Nnu_tr)';
                nu_tr = repmat(nu_tr, Nmu_tr, 1);
            elseif (strcmp(sampler_tr,'rand'))
                mu_tr = mu1 + (mu2-mu1) * rand(Ntr,1);
                nu_tr = nu1 + (nu2-nu1) * rand(Ntr,1);
            end

            % Evaluate force field for training patterns
            g_tr = cell(Ntr,1);
            for ii = 1:Ntr
                g_tr{ii} = @(t) f(t,mu_tr(ii));
            end

            % Get teaching input
            alpha_tr = zeros(L,Ntr);
            for ii = 1:Ntr
                [x,alpha_tr(:,ii)] = ...
                    solverFcn(a, b, K, g_tr{ii}, BCLt, BCLv, BCRt, nu_tr(ii), UL);
            end
                        
            % Add noise to training data to (attempt to) avoid overfitting
            %alpha_tr = alpha_tr + 0.1*randn(size(alpha_tr));

            %
            % Train
            %

            % Try different training algorithms to determine the optimal one, i.e. the
            % one yielding the minimum error
            err_opt_local = inf * ones(length(H),length(trainFcn));
            err_opt = inf;
            net_opt_local = cell(length(H),length(trainFcn));
            tr_opt_local = cell(length(H),length(trainFcn));

            for t = 1:length(trainFcn)
                % Try different number of hidden neurons to find the optimal network
                % architecture; for each architecture, re-training the network different
                % times and store the minimum test error
                err = inf * ones(length(H),1);

                for h = 1:length(H)
                    % Create the feedforward neural network
                    net = feedforwardnet(H(h),trainFcn{t});
                    %net = cascadeforwardnet(H(h),trainFcn{t});

                    % Set transfer (activation) function for hidden neurons
                    net.layers{1}.transferFcn = transferFcn;

                    % Set data division function
                    net.divideFcn = 'divideblock';

                    % Set ratio for training, validation and test data sets
                    net.divideParam.trainRatio = trainRatio;
                    net.divideParam.valRatio = valRatio;
                    net.divideParam.testRatio = testRatio;
                    
                    % Set maximum number of consecutive fails
                    net.trainParam.max_fail = 6;
                    
                    %net.trainParam.mu = 1;
                    %net.trainParam.mu_dec = 0.8;
                    %net.trainParam.mu_inc = 1.5;
                    
                    %net.performParam.regularization = 0.5;

                    % Set options for training window
                    net.trainParam.showWindow = showWindow;

                    for p = 1:nruns
                        % Initialize the network
                        net = init(net);

                        % Train the network and test the results on test set
                        [net, tr] = train(net, ...
                            [mu_tr' mu_va(1:Nva)' mu_te'; nu_tr' nu_va(1:Nva)' nu_te'], ...
                            [alpha_tr alpha_va(:,1:Nva) alpha_te]);
                        
                        % Compute average error on test data set
                        e = 0;
                        for r = 1:Nte
                            alpha_nn = net([mu_te(r) nu_te(r)]');
                            e = e + norm(u_te(:,r) - UL*alpha_nn);
                        end
                        e = e/Nte;

                        % Get the test error and keep it if it is the minimum so far
                        if (e < err_opt_local(h,t))  % Local checks
                            err_opt_local(h,t) = e;
                            net_opt_local{h,t} = net;
                            tr_opt_local{h,t} = tr;         
                            if (e < err_opt)  % Global checks
                                err_opt = e;
                                col_opt = t;
                                row_opt = h;
                            end
                        end
                    end
                end
            end

            % Save data
            if tosave
                filename = sprintf(['%s/LinearPoisson1d2pNN/' ...
                    'LinearPoisson1d2p_%s_%s%s_NN%s_' ...
                    'a%2.2f_b%2.2f_%s%2.2f_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
                    'K%i_Nmu%i_Nnu%i_N%i_L%i_Nmu_tr%i_Nnu_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
                    root, solver, reducer, sampler, sampler_tr, a, b, BCLt, ...
                    BCLv, BCRt, mu1, mu2, nu1, nu2, K, Nmu, Nnu, N, L, ...
                    Nmu_tr, Nnu_tr, Ntr, Nva, Nte, suffix);
                save(filename, 'datafile', 'H', 'nruns', 'trainFcn', 'Ntr', 'Nva', 'Nte', ...
                    'err_opt_local', 'net_opt_local', 'tr_opt_local', 'row_opt', 'col_opt');
            end

            % Print information about optimal parameters
            fprintf(['Number of training patterns: %i (%i for mu, %i for nu)\n' ...
                'Optimal training algorithm: %s\nOptimal number of hidden neurons: %i\n' ...
                'Minimum test error: %5E\n\n'], ...
                Ntr, Nmu_tr, Nnu_tr, trainFcn{col_opt}, H(row_opt), err_opt);
        end
    end
end