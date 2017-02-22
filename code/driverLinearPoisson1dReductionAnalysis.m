clc
clear variables
clear variables -global
close all

% Path to folder storing the datasets
root = '../datasets';

%% Plot full and reduced solution for three training values of $\mu$

%
% User defined settings
% 

K = 100;  N = 50;  L = 5;  J = 50;

%
% Run
%

% Load data
filename = sprintf('%s/LinearPoisson1dFEP1_K%i_N%i_L%i_J%i.mat', ...
    root, K, N, L, J);
load(filename);

% Select the three solutions to plot
idx = randi(J,3,1);
%[val, idx(3)] = max(mu_v);

% Open a new window
figure(1);
hold off

% Plot and set the legend
plot(x(1:1:end), u_v(1:1:end,idx(1)), 'b')
hold on
plot(x(1:1:end), ur_v(1:1:end,idx(1)), 'b:', 'Linewidth', 2)
plot(x(1:1:end), u_v(1:1:end,idx(2)), 'r')
plot(x(1:1:end), ur_v(1:1:end,idx(2)), 'r:', 'Linewidth', 2)
plot(x(1:1:end), u_v(1:1:end,idx(3)), 'g')
plot(x(1:1:end), ur_v(1:1:end,idx(3)), 'g:', 'Linewidth', 2)

% Define plot settings
str = sprintf('Full and reduced solution to Poisson equation (K = %i, N = %i, L = %i)', ...
    K, N, L);
title(str)
xlabel('x')
ylabel('u')
legend(sprintf('mu = %f, full', mu_v(idx(1))), sprintf('mu = %f, reduced', mu_v(idx(1))), ...
    sprintf('mu = %f, full', mu_v(idx(2))), sprintf('mu = %f, reduced', mu_v(idx(2))), ...
    sprintf('mu = %f, full', mu_v(idx(3))), sprintf('mu = %f, reduced', mu_v(idx(3))), ...
    'location', 'best')
grid on

%% Fix the number of snapshots and vary the rank of the reduces basis

%
% User defined settings
% 

K = 100;  N = 50;  L = [3 5 8 10];  J = 50;

%
% Plot a specific solution
%

% Select the solution to plot
idx = randi(J,1);

% Open a new plot window
figure(2);
hold off

% Load data, plot and update legend
str = 'legend(''location'', ''best''';
for i = 1:length(L)
    filename = sprintf('%s/LinearPoisson1dFEP1_K%i_N%i_L%i_J%i.mat', ...
    root, K, N, L(i), J);
    load(filename);
    plot(x(1:1:end), ur_v(1:1:end,idx));
    hold on
    stri = sprintf('''L = %i''', L(i));
    str = sprintf('%s, %s', str, stri);
end
plot(x(1:1:end), u_v(1:1:end,idx));
str = sprintf('%s, ''Full'')', str);
eval(str)

% Define plot settings
str = sprintf('Full and reduced solution to Poisson equation (K = %i, N = %i, mu = %f)', ...
    K, N, mu_v(idx));
title(str)
xlabel('x')
ylabel('u')
grid on

%
% Plot the error
%

% Open a new plot window
figure(3);
hold off

% Load data, plot and update legend
str = 'legend(''location'', ''best''';
for i = 1:length(L)
    filename = sprintf('%s/LinearPoisson1dFEP1_K%i_N%i_L%i_J%i.mat', ...
    root, K, N, L(i), J);
    load(filename);
    plot([1:J]', err);
    hold on
    stri = sprintf('''L = %i''', L(i));
    str = sprintf('%s, %s', str, stri);
end
str = sprintf('%s)', str);
eval(str)

% Define plot settings
str = sprintf('Absolute error between full and reduced solution to Poisson equation (K = %i, N = %i)', ...
    K, N);
title(str)
xlabel('Validation sample')
ylabel('Error')
grid on

%% Fix the rank of the reduces basis and vary the number of snapshots

%
% User defined settings
% 

K = 100;  N = [10 25 50 100];  L = 8;  J = 50;

%
% Plot a specific solution
%

% Select the solution to plot
%idx = randi(J,1);
idx = 43; %29, 33, 43, 45, 47, 50

% Open a new plot window
figure(4);
hold off

% Load data, plot and update legend
str = 'legend(''location'', ''best''';
for i = 1:length(N)
    filename = sprintf('%s/LinearPoisson1dFEP1_K%i_N%i_L%i_J%i.mat', ...
    root, K, N(i), L, J);
    load(filename);
    plot(x(1:1:end), ur_v(1:1:end,idx));
    hold on
    stri = sprintf('''N = %i''', N(i));
    str = sprintf('%s, %s', str, stri);
end
plot(x(1:1:end), u_v(1:1:end,idx));
str = sprintf('%s, ''Full'')', str);
eval(str)

% Define plot settings
str = sprintf('Full and reduced solution to Poisson equation (K = %i, L = %i, mu = %f)', ...
    K, L, mu_v(idx));
title(str)
xlabel('x')
ylabel('u')
grid on

%
% Plot the error
%

% Open a new plot window
figure(5);
hold off

% Load data, plot and update legend
str = 'legend(''location'', ''best''';
for i = 1:length(N)
    filename = sprintf('%s/LinearPoisson1dFEP1_K%i_N%i_L%i_J%i.mat', ...
    root, K, N(i), L, J);
    load(filename);
    plot([1:J]', err);
    hold on
    stri = sprintf('''N = %i''', N(i));
    str = sprintf('%s, %s', str, stri);
end
str = sprintf('%s)', str);
eval(str)

% Define plot settings
str = sprintf('Absolute error between full and reduced solution to Poisson equation (K = %i, L = %i)', ...
    K, L);
title(str)
xlabel('Validation sample')
ylabel('Error')
grid on
