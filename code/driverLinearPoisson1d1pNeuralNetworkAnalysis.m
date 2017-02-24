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
% step          step for selecting training samples among snapshots
% root          path to folder where storing the output dataset

a = -1;  b = 1;  K = 100;
mu1 = -1;  mu2 = 1;
BCLt = 'D';  BCLv = 0;
BCRt = 'D';  BCRv = 0;
solver = 'FEP1';
reducer = 'SVD';
N = 50;  L = 8;  J = 50;
step = 5;
root = '../datasets';

% Determine number of data used for training
Nt = floor(N/step);

% Load the data
filename = sprintf('%s/LinearPoisson1d1p_%s_%s_NN_a%2.2f_b%2.2f_%s%2.2f_%s%2.2f_mu1%2.2f_mu2%2.2f_K%i_N%i_Nt%i_L%i_J%i.mat', ...
    root, solver, reducer, a, b, BCLt, BCLv, BCRt, BCRv, mu1, mu2, K, N, Nt, L, J);
load(filename);

%% For each training algorithm, plot the error versus the number of hidden neurons

% Open a new plot window
figure(1);
hold off

% Plot and dynamically update the legend
str_legend = 'legend(''location'', ''best''';
for i = 1:length(trainFcn)
    semilogy(H, err_opt_local(:,i));
    hold on
    str_legend = sprintf('%s, ''%s''', str_legend, trainFcn{i});
end
str_legend = sprintf('%s)', str_legend);

% Define plot settings
title('Error on test data set for different number of hidden neurons')
xlabel('Number of hidden neurons')
ylabel('MSE')
grid on
eval(str_legend)

%% For the optimal network, plot error on training, validation and test data 
% set versus epochs

% Open a new plot window
figure(2);
hold off

% Extract optimal network
net_opt = net_opt_local{row_opt,col_opt};
tr_opt = tr_opt_local{row_opt,col_opt};

% Plot and define settings
semilogy(tr_opt.epoch,tr_opt.perf,'b', tr_opt.epoch,tr_opt.vperf,'r', ...
    tr_opt.epoch,tr_opt.tperf,'g')
title('Learning curves')
xlabel('Epoch')
ylabel('MSE')
grid on
legend('Training', 'Validation', 'Test', 'location', 'best')

%% For test data, compute regression line of current output versus associated
% teaching input for all output neurons

% Load training and testing data
load(datafile);

% Among verification data, determine the ones devoted to testing
valSamples = round(valRatio * (N + J));
mu_test = mu_v(valSamples+1:end);
t = alpha_v(:,valSamples+1:end);

% Extract optimal network and compute outputs
net_opt = net_opt_local{row_opt,col_opt};
y = net_opt(mu_test');

% Compute regression for each component of the output, then plot
for i = 1:size(y,1)
    [r,m,b] = regression(t(i,:),y(i,:));
    figure(2+i);
    plot(t(i,:),y(i,:),'bo', t(i,:),t(i,:),'r', t(i,:),r,'r:');
    str = sprintf('Current output versus teaching input for output neuron %i', i);
    title(str)
    xlabel('Teaching input')
    ylabel('Current output')
    grid on
    legend('Output', 'Perfect fitting', 'Regression line', 'location', 'best')
    yl = get(gca,'xlim');
    ylim(yl);
end

%% Comparison between full and reduced solution as given by the neural network

% Load training and testing data
load(datafile);

% Among verification data, determine the ones devoted to testing
valSamples = round(valRatio * (N + J));
mu_test = mu_v(valSamples+1:end);
ur_test = u_v(:,valSamples+1:end);

% Extract optimal network and compute outputs
net_opt = net_opt_local{row_opt,col_opt};
y = net_opt(mu_test');
Y = UL * y;

% Randomly selected three test samples for $\mu$
idx = randi(size(y,2), [3,1]);

% Open a new plot window
figure;
hold off

% Plot
plot(x, ur_test(:,idx(1)), 'b');
hold on
plot(x, Y(:,idx(1)), 'b--');
plot(x, ur_test(:,idx(2)), 'r');
plot(x, Y(:,idx(2)), 'r--');
plot(x, ur_test(:,idx(3)), 'g');
plot(x, Y(:,idx(3)), 'g--');

% Define plot settings
title('Comparison between reduced solution obtained through direct method and Neural Network')
xlabel('x')
ylabel('u')
grid on
legend(sprintf('\\mu = %f', mu_test(idx(1))), ...
    sprintf('\\mu = %f, Neural Network', mu_test(idx(1))), ...
    sprintf('\\mu = %f', mu_test(idx(2))), ...
    sprintf('\\mu = %f, Neural Network', mu_test(idx(2))), ...
    sprintf('\\mu = %f', mu_test(idx(3))), ...
    sprintf('\\mu = %f, Neural Network', mu_test(idx(3))), ...
    'location', 'best')

