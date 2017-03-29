% getLinearPoisson1dFEP1rhs Compute right-hand side for the linear system
% yielded by the application of linear finite elements (FE-P1) to the
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

function rhs = getLinearPoisson1dFEP1rhs(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
    % Build a uniform grid over the domain [a,b]
    h = (b-a) / (K-1);
    x = linspace(a,b,K)';
    
    % Compute RHS
    rhs = zeros(K,1);
    g = @(t) f(t) .* (x(2)-t) ./ h; 
    rhs(1) = integral(g, x(1), x(2));
    for i = 2:K-1
        g = @(t) f(t) .* ((t-x(i-1)) ./ h .* (t < x(i)) + ...
            (x(i+1)-t) ./ h .* (t >= x(i)));
        rhs(i) = integral(g, x(i-1), x(i+1));
    end
    g = @(t) f(t) .* (t-x(end-1)) ./ h; 
    rhs(end) = integral(g, x(end-1), x(end));
    
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
    elseif strcmp(BCRt,'N') || strcmp(BCRt,'R')
        rhs(end) = rhs(end) + BCRv;
    elseif strcmp(BCRt,'P')
        rhs(end) = 0;    
    end
end