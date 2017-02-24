% driverLinearPoisson1d1pNeuralNetwork Consider the one-dimensional linear
% equation $-u''(x) = f(x,\mu)$ on $[a,b]$ depending on the real parameter
% $\mu$. Given a reduced basis $U$ of rank L and some training reduced 
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
%               - 'FEP1': linear finite elements
%               - 'FEP2': quadratic finite elements
% reducer       method to compute the reduced basis
%               - 'SVD': Single Value Decomposition 
% N             number of snapshots
% L             rank of reduced basis
% J             number of verification values for $\mu$
% root          path to folder where storing the output dataset
% step          step for selecting training samples among snapshots
% H             number of hidden neurons
% nruns         number of re-training for each architecture
% transferFcn   transfer function
%               - 'logsig': logarithimc sigmoid
%               - 'tansig': tangential sigmoid
% trainFcn      training algorithm
%               - 'trainlm' : Levenberg-Marquardt
%               - 'trainscg': scaled conjugate gradient
%               - 'trainbfg': quasi-Newton method
%               - 'trainbr' : Bayesian regularization
% valRatio      among verification data, ratio of validation data with
%               respect to test data
% showWindow    TRUE to open training window, FALSE otherwise
% tosave        TRUE to store the results in a Matlab dataset, FALSE otherwise

a = -1;  b = 1;  K = 100;
mu1 = -1;  mu2 = 1;
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
N = 50;  L = 8;  J = 50;
root = '../datasets';

step = 5;  H = 1:25;  nruns = 15;
transferFcn = 'tansig';
trainFcn = {'trainlm', 'trainscg', 'trainbfg'};
valRatio = 0.3;
showWindow = true;

tosave = true;

%
% Run
%

% Load data for training
datafile = sprintf('%s/LinearPoisson1d1p_%s_%s_a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_L%i_J%i.mat', ...
            root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N, L, J);
load(datafile);

% Determine number of samples used for training and extract them
Nt = floor(N/step);
mu_t = mu_t(1:step:end);
alpha_t = alpha_t(:,1:step:end);

% Define ratio for training, validation and test data sets
trainRatio = Nt / (Nt+J);  
valRatio = valRatio * (1 - trainRatio);
testRatio = 1 - trainRatio - valRatio;

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
            [net, tr] = train(net, [mu_t' mu_v'], [alpha_t alpha_v]);

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
    filename = sprintf('%s/LinearPoisson1d1p_%s_%s_NN_a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_Nt%i_L%i_J%i.mat', ...
        root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N, Nt, L, J);
    save(filename, 'datafile', 'step', 'H', 'nruns', 'trainFcn', 'valRatio', ...
        'err_opt_local', 'net_opt_local', 'tr_opt_local', 'row_opt', 'col_opt');
end

% Print information about optimal parameters
fprintf('Optimal training algorithm: %s\nOptimal number of hidden neurons: %i\nMinimum test error: %5E\n', ...
    trainFcn{col_opt}, H(row_opt), err_opt);
