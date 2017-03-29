% getHeterogeneousViscosityLinearPoisson1d2pSVDreducedBasis Compute a reduced 
% POD basis for the one-dimensional parametrized Poisson equation 
% $-(v(x,\nu)u'(x))' = f(x,\mu)$, where $\mu$ and $\nu$ are real parameters. 
% The basis is computed out of an ensemble of snapshots, i.e.
% solutions to the Poisson equation for different values of $\mu$ and $\nu$. 
% Letting $Y$ the matrix whose columns store the snapshots and $Y = U S V^T$ 
% its Singular Value Decomposition (SVD), then the reduced basis of rank $l$ 
% is given by the first $l$ columns of $U$, i.e. the eigenvectors of $Y Y^T$
% associated to the $l$ largest eigenvalues.
%
% [x, mu, nu, Y, s, UL] = ...
%   getHeterogeneousViscosityLinearPoisson1d2pSVDreducedBasis ...
%   (mu1, mu2, nu1, nu2, Nmu, Nnu, L, solver, a, b, K, v, f, BCLt, BCLv, BCRt)
% \param mu1        lower-bound for $\mu$
% \param mu2        upper-bound for $\mu$
% \param Nmu        number of different sampled values for $\mu$
% \param nu1        lower-bound for $\nu$
% \param nu2        upper-bound for $\nu$
% \param Nnu        number of different sampled values for $\nu$
% \param sampler    how the values for the parameters should be
%                   sampled on $[\mu_1,\mu_2] \times [\nu_1,\nu_2]$:
%                   - 'unif': samples form a cartesian grid of size (Nmu,Nnu)
%                   - 'rand': Nmu = Nnu samples drawn from a uniform random
%                             distribution on the domain
% \param L          rank of the basis
% \param solver     handle to solver function (see, e.g., LinearPoisson1dFEP1 and
%                   LinearPoisson1dFEP2)
% \param a          left boundary of domain
% \param b          right boundary of domain
% \param K          number of subintervals
% \param v          viscosity as handle function
% \param f          right-hand side as a handle function
% \param BCLt       kind of left boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \param BCLv       value of left boundary condition
% \param BCRt       kind of right boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \out   x          computational grid
% \out   mu         vector (Nmu,1) storing the different values of $\mu$ 
%                   used to compute the snapshots
% \out   nu         vector (Nnu,1) storing the different values of $\nu$ 
%                   used to compute the snapshots
% \out   Y          matrix storing the snapshots in its columns
% \out   s          the first L singular values
% \out   UL         matrix whose columns store the vectors constituing the
%                   reduced basis

function [x, mu, nu, Y, s, UL] = ...
    getHeterogeneousViscosityLinearPoisson1d2pSVDreducedBasis ...
    (mu1, mu2, Nmu, nu1, nu2, Nnu, sampler, L, solver, a, b, K, v, f, ...
    BCLt, BCLv, BCRt, BCRv)
    % Values for the parameters $\mu$ and $\nu$
    if strcmp(sampler,'unif')
        mu = linspace(mu1, mu2, Nmu)';
        nu = linspace(nu1, nu2, Nnu)'; 
    elseif strcmp(sampler,'rand')
        % First, make sure Nmu = Nnu
        Nmu = max(Nmu,Nnu);  Nnu = Nmu;
        mu = mu1 + (mu2-mu1) * rand(Nmu,1);
        nu = nu1 + (nu2-nu1) * rand(Nmu,1);
    else
        error('Unknown sampler.')
    end
    
    % Store viscosity for different values of $\nu$ in a cell array
    vis = cell(Nnu,1);
    for i = 1:Nnu
        vis{i} = @(t) v(t,nu(i));
    end
        
    % Store the RHS for different values of $\mu$ in a cell array
    g = cell(Nmu,1);
    for j = 1:Nmu
        g{j} = @(t) f(t,mu(j));
    end
    
    % Get the snapshots, i.e. the solution for different values of $\mu$
    % and $\nu$
    if strcmp(sampler,'unif')
        for i = 1:Nnu
            for j = 1:Nmu
                if (i == 1) && (j == 1)
                    [x,y] = solver(a, b, K, vis{1}, g{1}, BCLt, BCLv, BCRt, BCRv);
                    Y = zeros(size(y,1),Nmu*Nnu);
                    Y(:,1) = y;
                else
                    [x,Y(:,(i-1)*Nmu+j)] = ...
                        solver(a, b, K, vis{i}, g{j}, BCLt, BCLv, BCRt, BCRv);
                end
            end
        end
    elseif strcmp(sampler,'rand')
        for i = 1:Nmu
            if i == 1
                [x,y] = solver(a, b, K, vis{1}, g{1}, BCLt, BCLv, BCRt, BCRv);
                Y = zeros(size(y,1),Nmu);
                Y(:,1) = y;
            else
                [x,Y(:,i)] = solver(a, b, K, vis{i}, g{i}, BCLt, BCLv, BCRt, BCRv);
            end
        end
    end
                        
    % Compute SVD decomposition of Y
    [U,S] = svd(Y);
    
    % Get the first L singular values
    s = diag(S);
    if (size(Y,2) < L)
        s = [s; zeros(L-size(Y,2),1)];
    end
        
    % Get reduced basis of rank L by retaining only the first L columns of U
    if (L < size(Y,2))
        UL = U(:,1:L); 
    else
        UL = U; 
    end
end