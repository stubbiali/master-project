% getLinearPoisson1dFEP2rhs Compute right-hand side for the linear system
% yielded by the application of quadratic finite elements (FE-P2) to the
% linear one-dimensional Poisson equation $-u''(x) = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions.
%
% rhs = getLinearPoisson1dFEP1rhs(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param f      right-hand side (handle function)
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
% \out   rhs    right-hand side

function rhs = getLinearPoisson1dFEP2rhs(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
    % Build a uniform grid over the domain [a,b]
    h = (b-a) / (K-1);
    xv = linspace(a,b,K)';
    
    % Get the nodes
    M = 2*K-1;
    xn = linspace(a,b,M)';
    
    % Allocate memory for RHS
    rhs = zeros(M,1);
    
    % Odd indeces
    g = @(t) f(t) .* 2.*(t-xn(2)).*(t-xn(3))./(h^2); 
    rhs(1) = integral(g, xv(1), xv(2));
    for i = 3:2:M-2
        g = @(t) f(t) .* (2.*(t-xn(i-2)).*(t-xn(i-1))./(h^2) .* (t < xn(i)) + ...
            2.*(t-xn(i+1)).*(t-xn(i+2))./(h^2) .* (t >= xn(i)));
        rhs(i) = integral(g, xn(i-2), xn(i+2));
    end
    g = @(t) f(t) .* 2.*(t-xn(end-2)).*(t-xn(end-1))./(h^2); 
    rhs(end) = integral(g, xv(end-1), xv(end));
    
    % Even indices
    for i = 2:2:M-1
        g = @(t) f(t) .* (-4.*(t-xn(i-1)).*(t-xn(i+1)))./(h^2);
        rhs(i) = integral(g, xn(i-1), xn(i+1));
    end
    
    % Modify RHS applying left boundary conditions
    if strcmp(BCLt,'D') 
        rhs(1) = BCLv/h;
    elseif strcmp(BCLt,'N')
        rhs(1) = rhs(1) - BCLv;
    elseif strcmp(BCLt,'P') 
        rhs(1) = 0;
    end
    
    % Modify RHS applying right boundary conditions
    if strcmp(BCRt,'D')
        rhs(end) = BCRv/h;
    elseif strcmp(BCRt,'N')
        rhs(end) = rhs(end) + BCRv;
    elseif strcmp(BCRt,'P') 
        rhs(end) = 0;
    end
end