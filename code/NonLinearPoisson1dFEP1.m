% NonLinearPoisson1dFEP1 Linear finite elements (FE-P1) solver for the 
% nonlinear one-dimensional Poisson equation $-( v(u(x)) u'(x) )' = f(x)$ 
% in the unknown $u = u(x)$, $x \in [a,b]$. Note that the nonlinearity lies 
% in the solution-dependent viscosity. The ODE should be completed with 
% Dirichlet, Neumann or periodic boundary conditions.
% The nonlinear system yielded by the Galerkin-FE method is solved using
% the Matlab built-in function fsolve.
%
% [x, u] = NonLinearPoisson1dFEP1(a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv)
%
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of grid points
% \param v      viscosity $v = v(u)$ as handle function
% \param f      forcing term $f = f(x)$ as handle function
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
% \out   x      grid
% \out   u      numerical solution

function [x,u] = NonLinearPoisson1dFEP1(a, b, K, v, f, BCLt, BCLv, ...
    BCRt, BCRv)
    % Check if the problem is well-posed
    if ~(strcmp(BCLt,'D') || strcmp(BCRt,'D'))
        error(['Ill-posed boundary value problem: at least one boundary ' ...
            'of the domain must be assigned Dirichlet conditions.']);
    end
        
    % Build a uniform grid over the interval $[a,b]$
    x = linspace(a,b,K)';  h = (b-a) / (K-1);
    
    % Compute right-hand side
    rhs = getLinearPoisson1dFEP1rhs(a, b, K, f, BCLt, BCLv, BCRt, BCRv);
            
    % Initialize solution with a linear function satisfying boundary conditions
    % Note: boundary conditions may not be satisfied if the viscosity is
    % not unitary at the boundary (but it should not be a problem since it
    % is just the starting point...)
    if strcmp(BCLt,'D')
        if strcmp(BCRt,'D') 
            u = BCLv + (x-a) / (b-a) * (BCRv-BCLv);
        elseif strcmp(BCRt,'N')
            u = BCRv * x + (BCLv - BCRv*a);
        elseif strcmp(BCRt,'P')
            u = BCLv * ones(K,1);
        end
    elseif strcmp(BCRt,'D')
        if strcmp(BCLt,'N')
            u = BCLv * x + (BCRv - BCLv*b);
        elseif strcmp(BCLt,'P')
            u = BCRv * ones(K,1);
        end
    end
    
    % Set the nonlinear system yielded by the Galerkin-FE method
    F = @(x) evalNonLinearPoisson1dFEP1System(x, h, v, rhs, BCLt, BCRt);
    
    % Solve the system
    options = optimset('Display','off');
    u = fsolve(F, u, options);
end