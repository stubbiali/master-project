% getNonLinearPoisson1d3pSVDreducedBasis Compute a reduced basis for the 
% nonlinear, parametrized one-dimensional Poisson equation 
% $-(v(u) u'(x))' = f(x,\mu,\nu,\xi)$, in the unknown $u = u(x)$, $x \in [a,b]$, 
% where $\mu$, $\nu$ and $\xi$ are real parameters. Note that the parameters may 
% affect also the boundary conditions. The basis is computed out of an ensemble 
% of snapshots, i.e. solutions to the Poisson equation for different values 
% of $\mu$, $\nu$ and $\xi$. Letting $U$ the matrix whose columns store the snapshots 
% and $U = V S W^T$ its Singular Value Decomposition (SVD), then the reduced 
% basis of rank $l$ is given by the first $l$ columns of $V$, i.e. the eigenvectors 
% of $U U^T$ associated to the $l$ largest eigenvalues.
%
% [x, mu, nu, xi, U, s, V] = getLinearPoisson1d3pSVDreducedBasis(mu1, mu2, nu1, nu2, ...
%   xi1, xi2, sampler, Nmu, Nnu, Nxi, L, solver, a, b, K, v, f, BCLt, BCLv, BCRt, BCRv)
% \param mu1        lower-bound for $\mu$
% \param mu2        upper-bound for $\mu$
% \param nu1        lower-bound for $\nu$
% \param nu2        upper-bound for $\nu$
% \param xi1        lower-bound for $\xi$
% \param xi2        upper-bound for $\xi$
% \param sampler    how the samples for $\mu$, $\nu$ and $\xi$ should be picked:
%                   - 'unif': uniformly distributed 
%                   - 'rand': drawn from a uniform random distribution; in
%                             this case, it should be Nmu = Nnu = Nxi
% \param Nmu        number of samples for $\mu$
% \param Nnu        number of samples for $\nu$
% \param Nxi        number of samples for $\xi$
% \param L          rank of the basis
% \param solver     handle to solver function (see, e.g.,
%                   NonLinearPoisson1dFEP1)
% \param a          left boundary of domain
% \param b          right boundary of domain
% \param K          number of grid points
% \param v          viscosity $v = v(u)$ as handle function
% \param f          source term $f = f(x,\mu)$ as a handle function
% \param BCLt       kind of left boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \param BCLv       left boundary condition as a handle function in mu, nu and xi
% \param BCRt       kind of right boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \param BCRv       right boundary condition as a handle function in mu, nu and xi
% \out   x          computational grid
% \out   mu         values of $\mu$ used to compute the snapshots
% \out   nu         values of $\nu$ used to compute the snapshots
% \out   xi         values of $\xi$ used to compute the snapshots
% \out   U          matrix storing the snaphsots in its columns
% \out   s          the first L singular values
% \out   V          matrix whose columns store the vectors constituing the
%                   reduced basis

function [x, mu, nu, xi, U, s, V] = getNonLinearPoisson1d3pSVDreducedBasis ...
    (mu1, mu2, nu1, nu2, xi1, xi2, sampler, Nmu, Nnu, Nxi, L, solver, a, b, K, v, dv, f, ...
    BCLt, BCLv, BCRt, BCRv)
    % Total number of snapshots
    if strcmp(sampler,'unif')
        N = Nmu*Nnu*Nxi;
    elseif strcmp(sampler,'rand')
        N = Nmu;
    end
    
    % Values for the parameter $\mu$
    if strcmp(sampler,'unif')
        mu = linspace(mu1, mu2, Nmu)';
    elseif strcmp(sampler,'rand')
        mu = mu1 + (mu2-mu1) * rand(N,1);
    else
        error('Unknown sampler.')
    end

    % Values for the parameter $\nu$
    if strcmp(sampler,'unif')
        nu = linspace(nu1, nu2, Nnu)';   
    elseif strcmp(sampler,'rand')
        nu = nu1 + (nu2-nu1) * rand(N,1);
    else
        error('Unknown sampler.')
    end
    
    % Values for the parameter $\xi$
    if strcmp(sampler,'unif')
        xi = linspace(xi1, xi2, Nxi)';
    elseif strcmp(sampler,'rand')
        xi = xi1 + (xi2-xi1) * rand(N,1);
    else
        error('Unknown sampler.')
    end

    % Gather all values for mu, nu and xi in three vectors
    if strcmp(sampler,'unif')
        mu_ext = zeros(N,1);  nu_ext = zeros(N,1);  xi_ext = zeros(N,1);
        idx = 1;
        for i = 1:Nmu
            for j = 1:Nnu
                for k = 1:Nxi
                    mu_ext(idx) = mu(i);
                    nu_ext(idx) = nu(j);
                    xi_ext(idx) = xi(k);
                    idx = idx+1;
                end
            end
        end
        mu = mu_ext;  nu = nu_ext;  xi = xi_ext;
    end 
    
    % Evaluate right-hand side for sampled values of $\mu$, $\nu$ and $\xi$
    g = cell(N,1);
    for i = 1:N
        g{i} = @(t) f(t,mu(i),nu(i),xi(i));
    end
    
    % Get the snapshots, i.e. the solution for different values of $\mu$
    % $\nu$ and $\xi$
    [x,u] = solver(a, b, K, v, dv, g{1}, BCLt, BCLv(mu(1),nu(1),xi(1)), ...
        BCRt, BCRv(mu(1),nu(1),xi(1)));
    U = zeros(size(u,1),N);  U(:,1) = u;
    for i = 2:N
        [x,U(:,i)] = solver(a, b, K, v, dv, g{i}, BCLt, BCLv(mu(i),nu(i),xi(i)), ...
            BCRt, BCRv(mu(i),nu(i),xi(i)));
    end
        
    % Compute SVD decomposition of U
    [V,S] = svd(U);
    
    % Get the first L singular values
    s = diag(S);
    if (N < L)
        s = [s; zeros(L-N,1)];
    end
        
    % Get reduced basis of rank L by retaining only the first L columns of V
    if (L < N)
        V = V(:,1:L);  
    end
end
