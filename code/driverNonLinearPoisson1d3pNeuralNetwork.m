% driverNonLinearPoisson1d3pNeuralNetwork Consider the one-dimensional 
% nonlinear equation $-(v(u) u'(x))' = f(x,\mu,\nu,\xi)$ in the unknown 
% $x \in [a,b]$, where $\mu$, $\nu$ and $\xi$ are real parameters. Note 
% that these parameters may enter also the boundary conditions. 
% Given a reduced basis $V$ of rank $l$ and some training reduced solutions, 
% we train a neural network to approximate the nonlinear map
% $[\mu,\nu,\xi] \mapsto \boldsymbol{\alpha}$, where $\boldsymbol{\alpha}$ 
% are the coefficients of the expansion of the reduced solution in terms of 
% the reduced basis vectors.

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
% f             force field $f = f(t,\mu,\nu)$ as handle function
% mu1           lower bound for $\mu$
% mu2           upper bound for $\mu$
% nu1           lower bound for $\nu$
% nu2           upper bound for $\nu$
% xi1           upper bound for $\xi$
% xi2           upper bound for $\xi$
% suffix        suffix for data file name
% BCLt          kind of left boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% BCLv          left boundary condition as handle function in mu and nu
% BCRt          kind of right boundary condition
%               - 'D': Dirichlet, 
%               - 'N': Neumann, 
%               - 'P': periodic
% BCRv          right boundary condition as handle function in mu and nu
% solver        solver
%               - 'FEP1': linear finite elements; the resulting nonlinear
%                         system is solved via the Matlab built-in fucntion fsolve
%               - 'FEP1Newton': linear finite elements; the resulting
%                               nonlinear system is solved via Newton's method
% reducer       method to compute the reduced basis
%               - 'SVD': Single Value Decomposition 
% sampler       how the values for $\mu$ amd $\nu$ should have been sampled 
%               for the computation of the reduced basis:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2] \times [\nu_1,\nu_2]$
%               - 'rand': drawn from a uniform random distribution on 
%               $[\mu_1,\mu_2] \times [\nu_1,\nu_2]$
% Nmu           number of snapshot values for $\mu$
% Nnu           number of snapshot values for $\nu$
% Nxi           number of snapshot values for $\xi$
% L             rank of reduced basis
% sampler_tr    how the training values for $\mu$ should be selected:
%               - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%               - 'rand': drawn from a uniform random distribution on $[\mu_1,\mu_2]$
% Nmu_tr_v      number of training patterns for $\mu$
% Nnu_tr_v      number of training patterns for $\nu$
% Nxi_tr_v      number of training patterns for $\xi$
% Nte           number of testing values for $\mu$, $\nu$ and $\xi$
% valPercentage ratio between number of validation and training values for 
%               $\mu$, $\nu$ and $\xi$
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

a = -pi/2;  b = pi/2;  K = 500;

% Suffix ''
v = @(u) exp(u);  dv = @(u) exp(u);
u = @(t,mu,nu,xi) nu*exp(xi*t).*(2+sin(mu*t));
du = @(t,mu,nu,xi) nu*exp(xi*t).*(xi*(2+sin(mu*t)) + mu*cos(mu*t));
ddu = @(t,mu,nu,xi) nu*exp(xi*t).*(xi*xi*(2+sin(mu*t)) + xi*mu*cos(mu*t) + ...
    mu*xi*cos(mu*t) - mu*mu*sin(mu*t));
f = @(t,mu,nu,xi) - exp(u(t,mu,nu,xi)) .* (du(t,mu,nu,xi).^2 + ddu(t,mu,nu,xi));
mu1 = 1;  mu2 = 3;  nu1 = 1;  nu2 = 3;  xi1 = -0.5;  xi2 = 0.5;  suffix = '';

BCLt = 'D';  BCLv = @(mu,nu) nu.*(2+sin(mu*a));
BCRt = 'D';  BCRv = @(mu,nu) nu.*(2+sin(mu*b));
solver = 'FEP1';
reducer = 'SVD';
sampler = 'unif';
Nmu = 25;  Nnu = 25;  Nxi = 25;  L = 8;
root = '../datasets';

H = 5:2:25;  nruns = 10;
sampler_tr_v = {'unif','rand'};
Nmu_tr_v = [5 10 15 20 25 50];  
Nnu_tr_v = [5 10 15 20 25 50];  
Nxi_tr_v = [5 10 15 20 25 50];
valPercentage = 0.3;  Nte = 100;
transferFcn = 'tansig';
trainFcn = {'trainlm'};
showWindow = false;
tosave = true;

%
% Run
%

% Set handle to solver function
if strcmp(solver,'FEP1')
    solverFcn = @NonLinearPoisson1dFEP1;
    rsolverFcn = @NonLinearPoisson1dFEP1Reduced;
elseif strcmp(solver,'FEP1Newton')
    solverFcn = @NonLinearPoisson1dFEP1Newton;
    rsolverFcn = @NonLinearPoisson1dFEP1ReducedNewton;
end

% Determine total number of training and validating samples
Ntr_v = Nmu_tr_v .* Nnu_tr_v .* Nxi_tr_v;

% Load data for training
N = Nmu*Nnu*Nxi;
if strcmp(sampler,'rand')
    Nmu = N;  Nnu = N;  Nxi = N;
end
datafile = sprintf(['%s/NonLinearPoisson1d3pSVD/' ...
    'NonLinearPoisson1d3p_%s_%s%s_a%2.2f_b%2.2f_%s_%s_' ...
    'mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_xi1%2.2f_xi2%2.2f_K%i_' ...
    'Nmu%i_Nnu%i_Nxi%i_N%i_L%i_Nte%i%s.mat'], ...
    root, solver, reducer, sampler, a, b, BCLt, BCRt, mu1, mu2, ...
    nu1, nu2, xi1, xi2, K, Nmu, Nnu, Nxi, N, L, Nte, suffix);
load(datafile);

%
% Create validation data set; its size is usually dependent on the number
% of training patterns
%

% Determing number of validation patterns
Nva_v = ceil(valPercentage * Ntr_v);

% Determine validation values for $\mu$, $\nu$ and $\xi$
mu_va = mu1 + (mu2-mu1) * rand(max(Nva_v),1);
nu_va = nu1 + (nu2-nu1) * rand(max(Nva_v),1);
xi_va = xi1 + (xi2-xi1) * rand(max(Nva_v),1);

% Evaluate force field at the validation values
g_va = cell(max(Nva_v),1);
parfor i = 1:max(Nva_v)
    g_va{i} = @(t) f(t,mu_va(i),nu_va(i),xi_te(i));
end

% Compute reduced solution
[x,u] = solverFcn(a, b, K, v, dv, g_va{1}, BCLt, ...
    BCLv(mu_va(1),nu_va(1),xi_va(1)), BCRt, BCRv(mu_va(1),nu_va(1),xi_va(1)));
u_va = zeros(size(u,1),max(Nva_v));  u_va(:,1) = u;
parfor i = 1:max(Nva_v)
    [x,u_va(:,i)] = solverFcn(a, b, K, v, dv, g_va{i}, BCLt, ...
        BCLv(mu_va(i),nu_va(i),xi_va(i)), BCRt, BCRv(mu_va(i),nu_va(i),xi_va(i)));
end
alpha_va = VL'*u_va;

%
% Create test data set
%

Nte_opt = 200;

% Load random data
load(strcat(root,'/random_numbers.mat'));

% Determine values for $\mu$ and $\nu$
mu_te = mu1 + (mu2-mu1) * random_on_reference_interval_first(1:Nte_opt);
nu_te = nu1 + (nu2-nu1) * random_on_reference_interval_second(1:Nte_opt);
xi_te = xi1 + (xi2-xi1) * random_on_reference_interval_first(Nte_opt+1:2*Nte_opt);

% Evaluate force field at the test values
g_te = cell(Nte_opt,1);
parfor i = 1:Nte_opt
    g_te{i} = @(t) f(t,mu_te(i),nu_te(i),xi_te(i));
end

% Compute reduced solution
[x,u] = solverFcn(a, b, K, v, dv, g_te{1}, BCLt, ...
    BCLv(mu_te(1),nu_te(1),xi_te(1)), BCRt, BCRv(mu_te(1),nu_te(1),xi_te(1)));
u_te = zeros(size(u,1),Nte_opt);  u_te(:,1) = u;
parfor i = 2:Nte_opt
    [x,u_te(:,i)] = solverFcn(a, b, K, v, dv, g_te{i}, BCLt, ...
        BCLv(mu_te(i),nu_te(i),xi_te(i)), BCRt, BCRv(mu_te(i),nu_te(i),xi_te(i)));
end
alpha_te = VL'*u_te;

% Set Nte
Nte = Nte_opt;

%
% Train the neural network
%

for s = 1:length(sampler_tr_v)
    for n = 1:length(Ntr_v)
        Ntr = Ntr_v(n);   Nva = Nva_v(n);  sampler_tr = sampler_tr_v{s};
        if (strcmp(sampler_tr,'unif'))
            Nmu_tr = Nmu_tr_v(n);  Nnu_tr = Nnu_tr_v(n);  Nxi_tr = Nxi_tr_v(n);
        elseif (strcmp(sampler_tr,'rand'))
            Nmu_tr = Ntr;  Nnu_tr = Ntr;  Nxi_tr = Ntr;
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
            mu_tr_s = linspace(mu1, mu2, Nmu_tr)';
            nu_tr_s = linspace(nu1, nu2, Nnu_tr)';
            xi_tr_s = linspace(xi1, xi2, Nnu_tr)';
            
            mu_tr = zeros(Ntr,1);  nu_tr = zeros(Ntr,1);  xi_tr = zeros(Ntr,1);
            idx = 1;
            for i = 1:Nmu_tr
                for j = 1:Nnu_tr
                    for k = 1:Nxi_tr
                        mu_tr(idx) = mu_tr_s(i);
                        nu_tr(idx) = nu_tr_s(j);
                        xi_tr(idx) = xi_tr_s(k);
                        idx = idx+1;
                    end
                end
            end
        elseif (strcmp(sampler_tr,'rand'))
            mu_tr = mu1 + (mu2-mu1) * rand(Ntr,1);
            nu_tr = nu1 + (nu2-nu1) * rand(Ntr,1);
            xi_tr = nu1 + (xi2-xi1) * rand(Ntr,1);
        end

        % Evaluate force field for training patterns
        g_tr = cell(Ntr,1);
        parfor i = 1:Ntr
            g_tr{i} = @(t) f(t,mu_tr(i),nu_tr(i),xi_tr(i));
        end

        % Get teaching input
        [x,u] = solverFcn(a, b, K, v, dv, g_tr{1}, ...
            BCLt, BCLv(mu_tr(1),nu_tr(1),xi_tr(1)), ...
            BCRt, BCRv(mu_tr(1),nu_tr(1),xi_tr(1)));
        u_tr = zeros(size(u,1),Ntr);
        parfor i = 2:Ntr
            [x,u_tr(:,i)] = solverFcn(a, b, K, v, dv, g_tr{i}, ...
                BCLt, BCLv(mu_tr(i),nu_tr(i),xi_tr(i)), ...
                BCRt, BCRv(mu_tr(i),nu_tr(i),xi_tr(i)));
        end
        alpha_tr = VL'*u_tr;

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
                net = configure(net, [mu_tr'; nu_tr'; xi_tr'], alpha_tr);

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
                
                % Set parameters for Levenberg-Marquardt algorithm
                %net.trainParam.mu = 1;
                %net.trainParam.mu_dec = 0.8;
                %net.trainParam.mu_inc = 1.5;
                
                % Set regularization coefficient
                %net.performParam.regularization = 0.5;
                
                % Set tolerance for error on training data set
                %net.trainParam.goal = 1e-3;
                
                % Set performance function
                %net.performFcn = 'mse';

                % Set options for training window
                net.trainParam.showWindow = showWindow;

                for p = 1:nruns
                    % Initialize the network
                    net = init(net);

                    % Train the network and test the results on test set
                    [net, tr] = train(net, ...
                        [mu_tr' mu_va(1:Nva)' mu_te'; ...
                        nu_tr' nu_va(1:Nva)' nu_te';
                        xi_tr' xi_va(1:Nva)' xi_te'], ...
                        [alpha_tr alpha_va(:,1:Nva) alpha_te]);
                    
                    % Get error on test data set
                    e = 0;
                    for r = 1:Nte
                        alpha = net([mu_te(r) nu_te(r) xi_te(r)]');
                        e = e + norm(u_te(:,r) - VL*alpha);
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
            filename = sprintf(['%s/NonLinearPoisson1d3pNN/' ...
                'NonLinearPoisson1d3p_%s_%s%s_NN%s_' ...
                'a%2.2f_b%2.2f_%s_%s_mu1%2.2f_mu2%2.2f_nu1%2.2f_nu2%2.2f_' ...
                'xi1%2.2f_xi2%2.2f_K%i_Nmu%i_Nnu%i_Nxi%i_N%i_L%i_' ...
                'Nmu_tr%i_Nnu_tr%i_Nxi_tr%i_Ntr%i_Nva%i_Nte%i%s.mat'], ...
                root, solver, reducer, sampler, sampler_tr, a, b, BCLt, ...
                BCRt, mu1, mu2, nu1, nu2, xi1, xi2, K, Nmu, Nnu, Nxi, N, L, ...
                Nmu_tr, Nnu_tr, Nxi_tr, Ntr, Nva, Nte, suffix);
            save(filename, 'datafile', 'H', 'nruns', 'trainFcn', 'Ntr', ...
                'Nva', 'Nte', 'err_opt_local', 'net_opt_local', ...
                'tr_opt_local', 'row_opt', 'col_opt');
        end

        % Print information about optimal parameters
        fprintf(['Number of training patterns: %i (%i for mu, %i for nu, %i for xi)\n' ...
            'Optimal training algorithm: %s\nOptimal number of hidden neurons: %i\n' ...
            'Minimum test error: %5E\n\n'], ...
            Ntr, Nmu_tr, Nnu_tr, Nxi_tr, trainFcn{col_opt}, H(row_opt), err_opt);
    end
end