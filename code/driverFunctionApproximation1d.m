% driverFunctionApproximation1d Train a feedforward neural network to
% approximate any mono-dimensional scalar function.

clc
clear variables 
clear variables -global
close all

%
% User defined settings
%

f = @(t) 2*t - 5;
a = -1;  b = 1;
Ntr = 50;  Nva = 0.3*Ntr;  Nte = 50;
H = 10;  nruns = 10;
transferFcn = 'logsig';
trainFcn = 'trainlm';
showWindow = true;

%
% Run
%

% Get train, validation and test patterns
xtr = linspace(a,b,Ntr);        ytr = f(xtr);
xva = a + (b-a) * rand(1,Nva);  yva = f(xva);
xte = a + (b-a) * rand(1,Nte);  yte = f(xte);

% Define ratio for training, validation and test data sets
trainRatio = Ntr / (Ntr + Nva + Nte);  
valRatio = Nva / (Ntr + Nva + Nte);
testRatio = 1 - trainRatio - valRatio;

% Initialize variables for storing results
err_opt_local = inf * ones(length(H),1);
err_opt = inf;
net_opt_local = cell(length(H),1);
tr_opt_local = cell(length(H),1);
err = inf * ones(length(H),1);

for h = 1:length(H)
    % Create the feedforward neural network
    net = feedforwardnet(H(h),trainFcn);
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
        [net, tr] = train(net, [xtr xva xte], [ytr yva yte]);

        % Get the test error and keep it if it is the minimum so far
        e = tr.best_tperf;
        if (e < err_opt_local(h))  % Local checks
            err_opt_local(h) = e;
            net_opt_local{h} = net;
            tr_opt_local{h} = tr;         
            if (e < err_opt)  % Global checks
                err_opt = e;
                row_opt = h;
            end
        end
    end
end

fprintf(['Number of training patterns: %i\n' ...
    'Optimal number of hidden neurons: %i\n' ...
    'Minimum test error: %5E\n\n'], ...
    Ntr, H(row_opt), err_opt);