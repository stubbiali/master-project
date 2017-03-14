% getNonLinearPoisson1d1pSVDreducedBasis Compute a reduced basis for the 
% nonlinear, parametrized one-dimensional Poisson equation 
% $-(v(u) u'(x))' = f(x,\mu)$, in the unknown $u = u(x)$, $x \in [a,b]$, 
% where $\mu$ is a real parameter. The basis is computed out of an ensemble 
% of snapshots, i.e. solutions to the Poisson equation for different values 
% of $\mu$. Letting $Y$ the matrix whose columns store the snapshots and 
% $Y = V S W^T$ its Singular Value Decomposition (SVD), then the reduced 
% basis of rank L is given by the first L columns of V, i.e. the eigenvectors 
% of $Y Y^T$ associated to the L largest eigenvalues.
%
% [x, mu, Y, s, V] = getLinearPoisson1d1pSVDreducedBasis(mu1, mu2, sampler, ...
%   N, L, solver, a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv)
% \param mu1        lower-bound for $\mu$
% \param mu2        upper-bound for $\mu$
% \param N          number of shapshots to compute
% \param sampler    how the shapshot values for $\mu$ should be selected:
%                   - 'unif': uniformly distributed on $[\mu_1,\mu_2]$
%                   - 'rand': drawn from a uniform random distribution on 
%                             $[\mu_1,\mu_2]$
% \param L          rank of the basis
% \param solver     handle to solver function (see, e.g.,
%                   NonLinearPoisson1dFEP1Newton)
% \param a          left boundary of domain
% \param b          right boundary of domain
% \param K          number of subintervals
% \param v          viscosity $v = v(u)$ as handle function
% \param dv         derivative of viscosity as handle function
% \param f          source term $f = f(x,\mu)$ as a handle function
% \param BCLt       kind of left boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \param BCLv       value of left boundary condition
% \param BCRt       kind of right boundary condition:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
%                   - 'P': periodic
% \param BCRv       value of right boundary condition
% \out   x          computational grid
% \out   mu         values of $\mu$ used to compute the snapshots
% \out   Y          matrix storing the snaphsots in its columns
% \out   s          the first L singular values
% \out   V          matrix whose columns store the vectors constituing the
%                   reduced basis

function [x, mu, Y, s, V] = getNonLinearPoisson1d1pSVDreducedBasis(mu1, mu2, ...
    N, sampler, L, solver, a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv)
    % Values for the parameter $\mu$
    if strcmp(sampler,'unif')
        mu = linspace(mu1, mu2, N)';
    elseif strcmp(sampler,'rand')
        mu = mu1 + (mu2-mu1) * rand(N,1);
    else
        error('Unknown sampler.')
    end
    
    % Store the RHS for different values of $\mu$ in a cell array
    g = cell(N,1);
    for i = 1:N
        g{i} = @(t) f(t,mu(i));
    end
    
    % Get the snapshots, i.e. the solution for different values of $\mu$
    for i = 1:N
        if (i == 1)
            [x,y] = solver(a, b, K, v, dv, g{1}, BCLt, BCLv, BCRt, BCRv);
            Y = zeros(size(y,1),N);  Y(:,1) = y;
        else
            [x,Y(:,i)] = solver(a, b, K, v, dv, g{i}, BCLt, BCLv, BCRt, BCRv);
        end
    end
        
    % Compute SVD decomposition of Y
    %h = x(2)-x(1);  m = size(Y,1);  W = sqrt(diag([h/2; h*ones(m-2,1); h/2]));  
    %[U,S] = svd(W*Y);
    %U = W \ U;
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