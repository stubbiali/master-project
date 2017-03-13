% driverNonLinearPoisson1d1pNeuralNetwork Consider the one-dimensional 
% nonlinear Poisson equation $-(v(u) u'(x)) = f(x,\mu)$ in the unknown
% $u = u(x)$, $x \in [a,b]$ depending on the real parameter $\mu \in [\mu_1,\mu_2]$. 
% Given a reduced basis $V$ of rank $l$ and some training reduced 
% solutions, we train a neural network to approximate the nonlinear map
% $\mu \mapsto \boldsymbol{\alpha}$, where $\boldsymbol{\alpha}$ are the
% coefficients of the expansion of the reduced solution in terms of the
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
% v             viscosity $v = v(u)$ as handle function
% dv            derivative of the viscosity as handle function
% f             force field $f = f(t,\mu)$ as handle function
% mu1           lower bound for $\mu$
% mu2           upper bound for $\mu$
% BCLt          kind of left boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% BCLv          value of left boundary condition
% BCRt          kind of right boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% BCRv          value of right boundary condition
% solver        solver
%               - 'FEP1Newton': linear finite elements coupled with
%               Newton's method for the resolution of the nonlinear system
% reducer       method to compute the reduced basis
%               - 'SVD': Single Value Decomposition 
% sampler       how the shapshot values for $\mu$ should have been sampled in
%               the computation of the reduced basis:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$
% N             number of snapshots
% L             rank of reduced basis
% sampler_tr    how the training values for $\mu$ should be selected:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$
% Ntr           number of training patterns
% Nte           number of testing values for $\mu$
% Nva           number of validating values for $\mu$ (chosen among testing
%               values)
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
v = @(t) 1 ./ (t.^2);  dv = @(t) - 2 ./ (t.^3);
f = @(t,mu) gaussian(t,mu,0.2);  mu1 = -1;  mu2 = 1;  suffix = '';
BCLt = 'D';  BCLv = 1;
BCRt = 'D';  BCRv = 1;
solver = 'FEP1Newton';
reducer = 'SVD';
sampler = 'unif';
N = 50;  L = 15;  Nte = 100;
root = '../datasets';

H = 1:30;  nruns = 15;
sampler_tr_v = {'unif'};
Ntr_v = [5 10 15 20 25 50 75 100];  Nva_v = ceil(0.3 * Ntr_v);  Nte_nn = 100;
transferFcn = 'tansig';
trainFcn = {'trainlm', 'trainscg', 'trainbfg'};
showWindow = false;
tosave = true;

%
% Run
%

% Set handle to solver function
if strcmp(solver,'FEP1Newton')
    solverFcn = @NonLinearPoisson1dFEP1Newton;
    rsolverFcn = @NonLinearPoisson1dFEP1ReducedNewton;
end

% Load data for training
datafile = sprintf(['%s/NonLinearPoisson1d1pSVD/' ...
    'NonLinearPoisson1d1p_%s_%s%s_a%2.2f_b%2.2f_%s%2.2f_' ...
    '%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, ...
    K, N, L, Nte, suffix);
load(datafile);

%
% Create validation data set; its isze is usually dependent on the number
% of training patterns
%

% Determine validation values for $\mu$
mu_va = mu1 + (mu2-mu1) * rand(max(Nva_v),1);

% Evaluate force field at the validation values
g_va = cell(max(Nva_v),1);
parfor i = 1:max(Nva_v)
    g_va{i} = @(t) f(t,mu_va(i));
end

% Compute reduced solution
alpha_va = zeros(L,max(Nva_v));
parfor i = 1:max(Nva_v)
    [x,alpha_va(:,i)] = rsolverFcn(a, b, K, v, dv, g_va{i}, BCLt, BCLv, ...
        BCRt, BCRv, VL);
end

%
% Train the neural network
%

for s = 1:length(sampler_tr_v)
    for n = 1:length(Ntr_v)
        % Shortcuts
        Ntr = Ntr_v(n);   Nva = Nva_v(n);  sampler_tr = sampler_tr_v{s};

        % Define ratio for training, validation and test data sets
        trainRatio = Ntr / (Ntr + Nva + Nte);  
        valRatio = Nva / (Ntr + Nva + Nte);
        testRatio = 1 - trainRatio - valRatio;
        
        %
        % Compute training patterns and teaching inputs
        %

        % Get training patterns
        if (strcmp(sampler_tr,'unif'))
            mu_tr = linspace(mu1, mu2, Ntr)';
        elseif (strcmp(sampler_tr,'rand'))
            mu_tr = mu1 + (mu2-mu1) * rand(Ntr,1);
        end

        % Get teaching input
        alpha_tr = zeros(L,Ntr);
        parfor i = 1:Ntr
            g = @(t) f(t,mu_tr(i));
            [x,alpha_tr(:,i)] = rsolverFcn(a, b, K, v, dv, g, ...
                BCLt, BCLv, BCRt, BCRv, VL);
        end
        
        %
        % Train
        %

        % Try different training algorithms to determine the optimal one, i.e. the
        % one yielding the minimum error
        err_opt_local = inf * ones(length(H),length(trainFcn));
        err_opt = inf;
        net_opt_local = cell(length(H),length(trainFcn));
        tr_opt_local = cell(length(H),length(trainFcn));

        for i = 1:length(trainFcn)
            % Try different number of hidden neurons to find the optimal network
            % architecture; for each architecture, re-training the network different
            % times and store the minimum test error
            err = inf * ones(length(H),1);

            for h = 1:length(H)
                % Create the feedforward neural network
                net = feedforwardnet(H(h),trainFcn{i});

                % Set transfer (activation) function for hidden neurons
                net.layers{1}.transferFcn = transferFcn;

                % Set data division function
                net.divideFcn = 'divideblock';

                % Set ratio for training, validation and test data sets
                net.divideParam.trainRatio = trainRatio;
                net.divideParam.valRatio = valRatio;
                net.divideParam.testRatio = testRatio;

                % Set options for training window
                net.trainParam.showWindow = showWindow;

                for j = 1:nruns
                    % Initialize the network
                    net = init(net);

                    % Train the network and test the results on test set
                    [net, tr] = train(net, [mu_tr' mu_va' mu_te'], [alpha_tr alpha_va alpha_te]);

                    % Get the test error and keep it if it is the minimum so far
                    e = tr.best_tperf;
                    if (e < err_opt_local(h,i))  % Local checks
                        err_opt_local(h,i) = e;
                        net_opt_local{h,i} = net;
                        tr_opt_local{h,i} = tr;         
                        if (e < err_opt)  % Global checks
                            err_opt = e;
                            col_opt = i;
                            row_opt = h;
                        end
                    end
                end
            end
        end

        % Save data
        if tosave
            filename = sprintf(['%s/NonLinearPoisson1d1pNN/' ...
                'NonLinearPoisson1d1p_%s_%s%s_NN%s_a%2.2f_' ...
                'b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_Ntr%i_' ...
                'Nva%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler, sampler_tr, a, b, BCLt, BCLv, ...
                BCRt, BCRv, mu1, mu2, K, N, L, Ntr, Nva, Nte, suffix);
            save(filename, 'datafile', 'H', 'nruns', 'trainFcn', 'Ntr', 'Nva', 'Nte', ...
                'err_opt_local', 'net_opt_local', 'tr_opt_local', 'row_opt', 'col_opt');
        end

        % Print information about optimal parameters
        fprintf(['Number of training patterns: %i\nOptimal training algorithm: %s\n' ...
            'Optimal number of hidden neurons: %i\nMinimum test error: %5E\n\n'], ...
            Ntr, trainFcn{col_opt}, H(row_opt), err_opt);
    end
end