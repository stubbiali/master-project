% LinearPoisson1dFEP2 Quadratic finite elements (FE-P2) solver for the linear 
% one-dimensional Poisson equation $-u''(x) = f(x)$ defined on the interval 
% $[a,b]$. The ODE should be completed with Dirichlet, Neumann or periodic
% boundary conditions.
% The solver supported also reduced order modeling. Given a set of
% L orthonormal vectors stored in the columns of a matrix $UL$, we seek a
% discrete solution of the form $u_L = UL \alpha$, where $\alpha$ is
% L-dimensional. Of course, we also need to project the linear system $(A,rhs)$
% yielded by the finite element methods. Then, the coefficients of the
% reduced solution solves the linear system $(UL^T A UL, UL^T rhs)$.
%
% [x, u] = LinearPoisson1dFEP2(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
% [x, alpha] = LinearPoisson1dFEP2(a, b, K, f, BCLt, BCLv, BCRt, BCRv, UL)
% [x, u, alpha] = LinearPoisson1dFEP2(a, b, K, f, BCLt, BCLv, BCRt, BCRv, UL)
%
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
% \param UL     matrix whose columns represent the vectors of the reduced basis
% \out   x      grid
% \out   u      numerical solution; in case of multiple provided RHSs, u is
%               a matrix whose columns store the solutions
% \out   alpha  reduced solution
function [x, varargout] = LinearPoisson1dFEP2(a, b, K, f, BCLt, BCLv, BCRt, BCRv, varargin)
    % Check if the problem is well-posed
    if strcmp(BCLt,'P') && strcmp(BCRt,'P')
        error(['Ill-posed boundary value problem: only periodic boundary ' ...
            'conditions are given. Consider to apply Dirichlet or Neumann conditions.']);
    end
    
    % Build a uniform grid over the interval $[a,b]$
    x = linspace(a,b,K)';
    
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
    
    % If a reduced basis is provided and the there are two output
    % arguments: compute only the reduced solution
    if (~isempty(varargin)) && (nargout == 2)
        UL = varargin{1};
        alpha = (UL'*A*UL) \ (UL'*rhs);
        varargout{1} = alpha;
        return;
    end
    
    % If here, for sure the full solution must be computed...
    u = A \ rhs;
    varargout{1} = u;
    
    % ... Yet, if a reduced basis is provided, also the reduced solution
    % must be calculated
    if ~isempty(varargin)
        UL = varargin{1};
        alpha = (UL'*A*UL) \ (UL'*rhs);
        varargout{2} = alpha;
    end
end