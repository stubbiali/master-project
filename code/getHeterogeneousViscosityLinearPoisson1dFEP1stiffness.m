% getHeterogenousViscosityLinearPoisson1dFEP1stiffness Compute stiffness matrix
% yielded by the application of linear finite elements (FE-P1) to the
% linear one-dimensional Poisson equation $-(v(x)u'(x))' = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions.
%
% A = getHeterogeneousViscosityLinearPoisson1dFEP1stiffness(a, b, K, v, BCLt, BCRt)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param v      space-dependent viscosity (handle function)
% \param BCLt   kind of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRt   kind of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \out   A      stiffness matrix

function A = getHeterogeneousViscosityLinearPoisson1dFEP1stiffness(a, b, K, v, BCLt, BCRt)
    % Build computational grid with uniform spacing
    h = (b-a) / (K-1);
    x = linspace(a, b, K)';
    
    % Build the stiffness matrix
    A = zeros(K,K);
    for i = 2:K-1
        % Compute (i,i-1)-th entry
        A(i,i-1) = -1/(h*h) * integral(v, x(i-1), x(i));
        
        % Compute (i,i)-th entry
        A(i,i) = 1/(h*h) * integral(v, x(i-1), x(i+1));
        
        % Compute (i,i+1)-th entry
        A(i,i+1) = -1/(h*h) * integral(v, x(i), x(i+1));
    end
    
    % Apply left boundary conditions
    if strcmp(BCLt,'D')
        A(1,1) = 1/h;
    elseif strcmp(BCLt,'N')
        A(1,1) = 1/(h*h) * integral(v, x(1), x(2));
        A(1,2) = -1/(h*h) * integral(v, x(1), x(2));
    elseif strcmp(BCLt,'P')
        A(1,1) = 1/h;  A(1,end) = -1/h;  
    end
    
    % Apply right boundary conditions
    if strcmp(BCRt,'D')
        A(end,end) = 1/h;  
    elseif strcmp(BCRt,'N')
        A(end,end-1) = -1/(h*h) * integral(v, x(end-1), x(end));
        A(end,end) = 1/(h*h) * integral(v, x(end-1), x(end));
    elseif strcmp(BCRt,'P')
        A(end,1) = 1/h;  A(end,end) = -1/h;  
    end
end

