% NonLinearPoisson1dFEP1Reduced Linear finite elements (FE-P1) 
% reduced basis solver for the nonlinear one-dimensional Poisson equation 
% $-( v(u(x)) u'(x) )' = f(x)$ in the unknown $u = u(x)$, $x \in [a,b]$. 
% Note that the nonlinearity lies in the solution-dependent viscosity. 
% The ODE should be completed with Dirichlet, Neumann or periodic boundary 
% conditions.
% The discretization via the Galerkin-FE method of the BVP yields the 
% nonlinear system 
% $\boldsymbol{F}(\boldsymbol{u}) = \boldsymbol{0} \in \mathbb{R}^K$, where
% $K$ is the number of grid points and $\boldsymbol{u} \in \mathbb{R}^K$ is
% the numerical solution. Given a basis $V \in \mathbb{R}^{K \times L}$ of
% rank $L$, reduced order modeling consists in seeking a solution in the form 
% $\boldsymbol{u}^l = V \boldsymbol{\alpha}$, $\alpha \in \mathbb{R}^L$ by
% projecting the nonlinear system on the column space of $V$, i.e. we
% compute $\boldsymbol{\alpha}$ by solving the reduced nonlinear system
% $V^T \boldsymbol{F}(V \boldsymbol{\alpha}) = \boldsymbol{0} \in \mathbb{R}^L$.
% This system is solved using the Matlab built-in function fsolve.
%
% [x,alpha] = NonLinearPoisson1dFEP1Reduced(a, b, K, v, f, BCLt, BCLv, ...
%   BCRt, BCRv, V)
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
% \param V      reduced basis stored as a matrix (K,L)
% \out   x      grid
% \out   u      numerical solution

function [x,alpha] = NonLinearPoisson1dFEP1Reduced(a, b, K, v, f, ...
    BCLt, BCLv, BCRt, BCRv, V)
    % Check if the problem is well-posed
    if ~(strcmp(BCLt,'D') || strcmp(BCRt,'D'))
        error(['Ill-posed boundary value problem: at least one boundary ' ...
            'of the domain must be assigned Dirichlet conditions.']);
    end
       
    % Build a uniform grid over the interval $[a,b]$
    x = linspace(a,b,K)';  h = (b-a) / (K-1);
    
    % Compute right-hand side
    rhs = getLinearPoisson1dFEP1rhs_f(a, b, K, f, BCLt, BCLv, BCRt, BCRv);
            
    % Initialize full solution with a linear function interpolating boundary 
    % conditions
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
    
    % Compute initial coefficients of reduced solution by least square
    alpha = V \ u;  
        
    % Set the reduced nonlinear system yielded the Galerkin-FE method
    F = @(x) evalNonLinearPoisson1dFEP1ReducedSystem(x, h, v, rhs, V, BCLt, BCRt);
    
    % Solve the system
    options = optimset('Display','off');
    alpha = fsolve(F, alpha, options);
end