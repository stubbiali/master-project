% LinearPoisson1dFEP2reduced Reduced version of the quadratic finite elements 
% (FE-P2) solver for the linear one-dimensional Poisson equation $-u''(x) = f(x)$ 
% defined on the interval $[a,b]$. The solution is expressed in terms of a 
% provided reduced basis, computed, e.g., through getLinearPoisson1dReducedBasis.
%
% [x, u] = LinearPoisson1dFEP2reduced(U, a, b, K, f, BCLt, BCLv, BCRt, BCRv)
% \param U      matrix whose columns represent the vectors constituing the
%               reduced basis
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param f      RHS; this may be either an handle function or a cell array
%               of handle functions; in the latter case, the solution is
%               computed for each RHS
% \param BCLt   kind of left boundary condition; 'D' = Dirichlet, 'N' =
%               Neumann, 'P' = periodic
% \param BCLv   value of left boundary condition
% \param BCRt   kind of right boundary condition; 'D' = Dirichlet, 'N' =
%               Neumann, 'P' = periodic
% \param BCRv   value of right boundary condition
% \out   x      grid
% \out   u      numerical solution, i.e. coefficients of the expansion in
%               terms of the reduced basis; in case of multiple provided 
%               RHSs, u is a matrix whose columns store the solutions
%
% See LinearPoisson1dFEP2, getLinearPoisson1dReducedBasis
function [x, u] = LinearPoisson1dFEP2reduced(U, a, b, K, f, BCLt, BCLv, BCRt, BCRv)
    % Check if the problem is well-posed
    if strcmp(BCLt,'P') && strcmp(BCRt,'P')
        error(['Ill-posed boundary value problem: only periodic boundary ' ...
            'conditions are given. Consider to apply Dirichlet or Neumann conditions.']);
    end
    
    % Uniformly distribute the nodes over the interval $[a,b]$
    M = 2*K-1;
    x = linspace(a,b,M)';
    
    % Build the stiffness matrix
    A = getLinearPoisson1dFEP2stiffness(a, b, K, BCLt, BCRt);
    
    % Compute the RHS
    if ~iscell(f)
        rhs = getLinearPoisson1dFEP2rhs(a, b, K, f, BCLt, BCLv, BCRt, BCRv);
    else
        N = length(f);
        rhs = zeros(K,N);
        for i = 1:N
            rhs(:,i) = getLinearPoisson1dFEP2rhs(a, b, K, f{i}, BCLt, BCLv, BCRt, BCRv);
        end
    end
    
    % Solve the reduced system
    u = (U'*A*U) \ (U'*rhs);
end