% getNonLinearPoisson1d2pSVDreducedBasis Compute a reduced basis for the 
% nonlinear, parametrized one-dimensional Poisson equation 
% $-(v(u) u'(x))' = f(x,\mu,\nu)$, in the unknown $u = u(x)$, $x \in [a,b]$, 
% where $\mu$ and $\nu$ are real parameters. Note that the parameters may 
% affect also the boundary conditions. The basis is computed out of an ensemble 
% of snapshots, i.e. solutions to the Poisson equation for different values 
% of $\mu$ and $\nu$. Letting $Y$ the matrix whose columns store the snapshots 
% and $Y = V S W^T$ its Singular Value Decomposition (SVD), then the reduced 
% basis of rank L is given by the first L columns of V, i.e. the eigenvectors 
% of $Y Y^T$ associated to the L largest eigenvalues.
%
% [x, mu, Y, s, V] = getLinearPoisson1d2pSVDreducedBasis(mu1, mu2, nu1, nu2, ...
%  sampler, Nmu, Nnu, L, solver, a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv)
% \param mu1        lower-bound for $\mu$
% \param mu2        upper-bound for $\mu$
% \param nu1        lower-bound for $\nu$
% \param nu2        upper-bound for $\nu$
% \param sampler    how the samples for $\mu$ and $\nu$ should be picked:
%                   - 'unif': uniformly distributed 
%                   - 'rand': drawn from a uniform random distribution; in
%                             this case, it should be Nmu = Nnu
% \param Nmu        number of samples for $\mu$
% \param Nnu        number of samples for $\nu$
% \param L          rank of the basis
% \param solver     handle to solver function (see, e.g.,
%                   NonLinearPoisson1dFEP1)
% \param a          left boundary of domain
% \param b          right boundary of domain
% \param K          number of grid points
% \param v          viscosity $v = v(u)$ as handle function
% \param dv         derivative of viscosity as handle function
% \param f          source term $f = f(x,\mu)$ as a handle function
% \param BCLt       kind of left boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \param BCLv       left boundary condition as a handle function in mu and nu
% \param BCRt       kind of right boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \param BCRv       right boundary condition as a handle function in mu and nu
% \out   x          computational grid
% \out   mu         values of $\mu$ used to compute the snapshots
% \out   Y          matrix storing the snaphsots in its columns
% \out   s          the first L singular values
% \out   V          matrix whose columns store the vectors constituing the
%                   reduced basis

function [x, mu, nu, Y, s, V] = getNonLinearPoisson1d2pSVDreducedBasis ...
    (mu1, mu2, nu1, nu2, sampler, Nmu, Nnu, L, solver, a, b, K, v, f, ...
    BCLt, BCLv, BCRt, BCRv)
    % Total number of snapshots
    if strcmp(sampler,'unif')
        N = Nmu*Nnu;
    elseif strcmp(sampler,'rand')
        N = Nmu;
    end
    
    % Values for the parameter $\mu$
    if strcmp(sampler,'unif')
        mu = linspace(mu1, mu2, Nmu);  mu = repmat(mu,[Nnu,1]);  mu = mu(:);
    elseif strcmp(sampler,'rand')
        mu = mu1 + (mu2-mu1) * rand(N,1);
    else
        error('Unknown sampler.')
    end
    
    % Values for the parameter $\nu$
    if strcmp(sampler,'unif')
        nu = linspace(nu1, nu2, Nnu)';  nu = repmat(nu,[Nmu,1]);
    elseif strcmp(sampler,'rand')
        nu = nu1 + (nu2-nu1) * rand(N,1);
    else
        error('Unknown sampler.')
    end
    
    % Evaluate right-hand side for sampled values of $\mu$ and $\nu$
    g = cell(N,1);
    for i = 1:N
        g{i} = @(t) f(t,mu(i),nu(i));
    end
    
    % Get the snapshots, i.e. the solution for different values of $\mu$
    % and $\nu$
    [x,y] = solver(a, b, K, v, g{1}, BCLt, BCLv(mu(1),nu(1)), ...
        BCRt, BCRv(mu(1),nu(1)));
    Y = zeros(size(y,1),N);  Y(:,1) = y;
    for i = 2:N
        [x,Y(:,i)] = solver(a, b, K, v, g{i}, BCLt, BCLv(mu(i),nu(i)), ...
            BCRt, BCRv(mu(i),nu(i)));
    end
        
    % Compute SVD decomposition of Y
    [U,S] = svd(Y);
    
    % Get the first L singular values
    s = diag(S);
    if (N < L)
        s = [s; zeros(L-N,1)];
    end
        
    % Get reduced basis of rank L by retaining only the first L columns of U
    if (L < N)
        V = U(:,1:L); 
    else
        V = U; 
    end
end