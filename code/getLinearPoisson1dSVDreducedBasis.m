% getLinearPoisson1dSVDreducedBasis Compute a reduced basis for the one-dimensional
% parametrized Poisson equation $-u''(x) = f(x,\mu)$, where $\mu$ is a real
% parameter. The basis is computed out of an ensemble of snapshots, i.e.
% solutions to the Poisson equation for different values of $\mu$. Letting 
% $Y$ the matrix whose columns store the snapshots and $Y = U S V^T$ its
% Singular Value Decomposition (SVD), then the reduced basis of rank L is 
% given by the first L columns of U, i.e. the eigenvectors of $Y Y^T$
% associated to the L largest eigenvalues.
%
% [x, Y, UL] = getLinearPoisson1dSVDreducedBasis(mu1, mu2, N, L, solver, a, b, K, f, BCLt, BCLv, BCRt, BCRv)
% \param mu1    lower-bound for $\mu$
% \param mu2    upper-bound for $\mu$
% \param N      number of shapshots to compute
% \param L      rank of the basis
% \param solver handle to solver function (see, e.g., LinearPoisson1dFEP1 and
%               LinearPoisson1dFEP2)
% \param a      left boundary of domain
% \param b      right boundary of domain
% \param K      number of subintervals
% \param f      RHS as a handle function
% \param BCLt   kind of left boundary condition; 'D' = Dirichlet, 'N' =
%               Neumann, 'P' = periodic
% \param BCLv   value of left boundary condition
% \param BCRt   kind of right boundary condition; 'D' = Dirichlet, 'N' =
%               Neumann, 'P' = periodic
% \param BCRv   value of right boundary condition
% \out   x      computational grid
% \out   mu     values of $\mu$ used to compute the snapshots
% \out   Y      matrix storing the snaphsots in its columns
% \out   UL     matrix whose columns store the vectors constituing the
%               reduced basis
function [x, mu, Y, UL] = getLinearPoisson1dSVDreducedBasis(mu1, mu2, N, L, solver, a, b, K, f, BCLt, BCLv, BCRt, BCRv)
    % Values for the parameter $\mu$, evenly distributed over the interval
    % $[\mu_1,\mu_2]$
    mu = linspace(mu1, mu2, N)';
    
    % Store the RHS for different values of $\mu$ in a cell array
    g = cell(N,1);
    for i = 1:N
        g{i} = @(t) f(t,mu(i));
    end
    
    % Get the snapshots, i.e. the solution for different values of $\mu$
    [x,Y] = solver(a, b, K, g, BCLt, BCLv, BCRt, BCRv);
        
    % Compute SVD decomposition of Y
    [U, S, V] = svd(Y);
    
    % Get reduced basis of rank L by retaining only the first L columns of U
    M = size(U);
    if (L < M)
        UL = U(:,1:L);
    else
        UL = U;
    end
end