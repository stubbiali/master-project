% HeterogeneousViscosityLinearPoisson1dFEP1_f Linear finite elements (FE-P1) 
% solver for the linear one-dimensional Poisson equation $-(v(x)u'(x)) = f(x)$ 
% in the unknown function $u = u(x)$, $x \in [a,b]$. The ODE should be 
% completed with Dirichlet, Neumann or periodic boundary conditions.
% The solver supported also reduced order modeling. Given a set of
% $l$ orthonormal vectors stored in the columns of a matrix $U^l$, we seek a
% discrete solution of the form $u^l = U^l \boldsymbol{\alpha}$, where 
% $\boldsymbol{\alpha}$ is $l$-dimensional. Of course, we also need to project 
% the linear system $(A,rhs)$ yielded by the finite element methods. Then, 
% the coefficients of the reduced solution solves the linear system 
% $((U^l)^T A (U^l), (U^l)^T rhs)$.
% The "f" in the function name stands for "fast". Indeed, 
% persistent variables are used so to perform computations only when needed.
%
% [x, u] = HeterogeneousViscosityLinearPoisson1dFEP1_f(a, b, K, v, f, ...
%   BCLt, BCLv, BCRt, BCRv)
% [x, alpha] = HeterogeneousViscosityLinearPoisson1dFEP1_f(a, b, K, v, f, ...
%   BCLt, BCLv, BCRt, BCRv, UL)
% [x, u, alpha] = HeterogeneousViscosityLinearPoisson1dFEP1_f(a, b, K, v, f, ...
%   BCLt, BCLv, BCRt, BCRv, UL)
%
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param v      space-dependent viscosity (handle function)
% \param f      RHS; this may be either an handle function or a cell array
%               of handle functions; in the latter case, the solution is
%               computed for each RHS
% \param BCLt   kind of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCLv   value of left boundary condition
% \param BCRt   kind of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRv   value of right boundary condition
% \param UL     matrix whose columns represent the vectors of the reduced basis
% \out   x      grid
% \out   u      numerical solution; in case of multiple provided RHSs, u is
%               a matrix whose columns store the solutions
% \out   alpha  reduced solution

function [x, varargout] = HeterogeneousViscosityLinearPoisson1dFEP1_f ...
    (a, b, K, v, f, BCLt, BCLv, BCRt, BCRv, varargin)
    % Check if the problem is well-posed
    if strcmp(BCLt,'P') && strcmp(BCRt,'P')
        error(['Ill-posed boundary value problem: only periodic boundary ' ...
            'conditions are given. Consider to apply Dirichlet or Neumann conditions.']);
    end
    
    % Build a uniform grid over the interval $[a,b]$
    x = linspace(a,b,K)';
    
    % Build the stiffness matrix
    A = getHeterogeneousViscosityLinearPoisson1dFEP1stiffness_f(a, b, K, v, BCLt, BCRt);
    
    % Compute the RHS
    if ~iscell(f)
        rhs = getLinearPoisson1dFEP1rhs_f(a, b, K, f, BCLt, BCLv, BCRt, BCRv);
    else
        N = length(f);
        rhs = zeros(K,N);
        for i = 1:N
            rhs(:,i) = getLinearPoisson1dFEP1rhs_f ...
                (a, b, K, f{i}, BCLt, BCLv, BCRt, BCRv);
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