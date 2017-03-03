% getLinearPoisson1dFEP1stiffness Compute stiffness matrix
% yielded by the application of linear finite elements (FE-P1) to the
% linear one-dimensional Poisson equation $-u''(x) = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions.
%
% A = getLinearPoisson1dFEP1stiffness(a, b, K, f, BCLt, BCRt)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param f      RHS; this may be either an handle function or a cell array
%               of handle functions; in the latter case, the solution is
%               computed for each RHS
% \param BCLt   kind of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRt   kind of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \out   A      stiffness matrix

function A = getLinearPoisson1dFEP1stiffness(a, b, K, BCLt, BCRt)
    % Get uniform grid spacing
    h = (b-a) / (K-1);
    
    % Build the stiffness matrix
    A = 1/h * (diag(-ones(K-1,1),-1) + diag(2*ones(K,1)) + ...
        diag(-ones(K-1,1),1));
    
    % Modify stiffness matrix applying left boundary conditions
    if strcmp(BCLt,'D')
        A(1,1) = 1/h;  A(1,2) = 0;
    elseif strcmp(BCLt,'N')
        A(1,1) = 1/h;
    elseif strcmp(BCLt,'P')
        A(1,1) = 1/h;  A(1,2) = 0;  A(1,end) = -1/h;  
    end
    
    % Modify stiffness matrix applying right boundary conditions
    if strcmp(BCRt,'D')
        A(end,end) = 1/h;  A(end,end-1) = 0;  
    elseif strcmp(BCRt,'N')
        A(end,end) = 1/h;
    elseif strcmp(BCRt,'P')
        A(end,1) = 1/h;  A(end,end-1) = 0;  A(end,end) = -1/h;  
    elseif strcmp(BCRt,'R')
        A(end,end) = A(end,end)+1;
    end
end