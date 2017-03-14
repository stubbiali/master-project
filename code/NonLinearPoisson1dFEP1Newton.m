% NonLinearPoisson1dFEP1Newton Linear finite elements (FE-P1) solver for the 
% nonlinear one-dimensional Poisson equation $-( v(u(x)) u'(x) )' = f(x)$ 
% in the unknown $u = u(x)$, $x \in [a,b]$. Note that the nonlinearity lies 
% in the solution-dependent viscosity. The ODE should be completed with 
% Dirichlet, Neumann or periodic boundary conditions.
% The nonlinear system yielded by the Galerkin-FE method is solved using
% Newton's method. The norm of the difference between consecutive
% iterations is used as stopping criterium. The maximum number of
% iterations and the tolerance for the stopping criterium can be tuned by 
% the user.
%
% [x, u] = NonLinearPoisson1dFEP1Newton(a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv)
% [x, u] = NonLinearPoisson1dFEP1Newton(a, b, K, v, dv, f, BCLt, BCLv, BCRt, BCRv, ...
%   'iter', iter, 'tol', tol)
%
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param v      viscosity $v = v(u)$ as handle function
% \param dv     derivative of the viscosity as handle function
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
% \param iter   maximum number of iterations for Newton's method
% \param tol    tolerance for stopping criterium for Newton's method
% \out   x      grid
% \out   u      numerical solution

function [x, u] = NonLinearPoisson1dFEP1Newton(a, b, K, v, dv, f, BCLt, BCLv, ...
    BCRt, BCRv, varargin)
    % Check if the problem is well-posed
    if ~(strcmp(BCLt,'D') || strcmp(BCRt,'D'))
        error(['Ill-posed boundary value problem: at least one boundary ' ...
            'of the domain must be assigned Dirichlet conditions.']);
    end
    
    % Set parameters for Newton's method
    iter = 1000;
    tol = 1e-8;
    for i = 1:2:length(varargin)
        if strcmp(varargin{i},'iter')
            iter = varargin{i+1};
        elseif strcmp(varargin{i},'tol')
            tol = varargin{i+1};
        end
    end
    
    tol1 = tol;
    tol2 = 1e-6;
    
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
            %u = 10*ones(K,1);
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
    
    % Allocate memory for evaluations of the nonlinear function and its
    % Jacobian
    F = zeros(K,1);  J = zeros(K,K);
    
    % Run Newton's solver
    err1 = 1+tol1;  err2 = 1+tol2;  n = 0;
    while ((err1 > tol1) || (err2 > tol2)) && (n < iter)
       % Shortcuts
       ul = u(1:end-2);  uc = u(2:end-1);  ur = u(3:end);
       ulc = 0.5*(ul+uc);  ucr = 0.5*(uc+ur);
       
       % Save current solution
       uold = u;
              
       % Evaluate nonlinear equations associated with internal points of
       % the domain
       F(2:end-1) = (1/(6*h)) * (- (v(ul) + 4*v(ulc) + v(uc)) .* ul ...
           + (v(ul) + 4*v(ulc) + 2*v(uc) + 4*v(ucr) + v(ur)) .* uc ...
           - (v(uc) + 4*v(ucr) + v(ur)) .* ur) - rhs(2:end-1);
       
       % Evaluate nonlinear equation associated with left boundary
       if strcmp(BCLt,'D')
           F(1) = (1/h) * u(1) - rhs(1);
       elseif strcmp(BCLt,'N')
           F(1) = (1/(6*h)) * ((v(u(1)) + 4*v(0.5*(u(1)+u(2))) + v(u(2))) * ...
               (u(1) - u(2))) - rhs(1);
       elseif strcmp(BCLt,'P')
           F(1) = (1/h) * (u(1) - u(end));
       end
       
        % Evaluate nonlinear equation associated with right boundary
       if strcmp(BCRt,'D')
           F(end) = (1/h) * u(end) - rhs(end);
       elseif strcmp(BCRt,'N')
           F(end) = (1/(6*h)) * ((v(u(end-1)) + 4*v(0.5*(u(end-1)+u(end))) + v(u(end))) * ...
               (-u(end-1) + u(end))) - rhs(end);
       elseif strcmp(BCRt,'P')
           F(end) = (1/h) * (u(1) - u(end));
       end
       
       % Evaluate Jacobian at current solution for equations associated
       % with internal points
       J(2:end-1,1:end-2) = (1/(6*h)) * diag(- (dv(ul) + 2*dv(ulc)) .* ul ...
           - (v(ul) + 4*v(ulc) + v(uc)) ...
           + (dv(ul) + 2*dv(ulc)) .* uc);
       J(2:end-1,2:end-1) = J(2:end-1,2:end-1) + ...
           (1/(6*h)) * diag(- (2*dv(ulc) + dv(uc)) .* ul ...
           + (2*dv(ulc) + 2*dv(uc) + 2*dv(ucr)) .* uc ...
           + (v(ul) + 4*v(ulc) + 2*v(uc) + 4*v(ucr) + v(ur)) ...
           - (dv(uc) + 2*dv(ucr)) .* ur);
       J(2:end-1,3:end) = J(2:end-1,3:end) + ...
           (1/(6*h)) * diag((2*dv(ucr) + dv(ur)) .* uc ...
           - (2*dv(ucr) + dv(ur)) .* ur ...
           - (v(uc) + 4*v(ucr) + v(ur)));
             
       % Evaluate Jacobian at current solution for the equation associated
       % with the left boundary
       if strcmp(BCLt,'D') && (n == 0)
           J(1,1) = 1/h;
       elseif strcmp(BCLt,'N')
           J(1,1) = (1/(6*h)) * ((dv(u(1)) + 2*dv(0.5*(u(1)+u(2)))) * (u(1) - u(2)) ...
               + (v(u(1)) + 4*v(0.5*(u(1)+u(2))) + v(u(2))));
           J(1,2) = (1/(6*h)) * ((2*dv(0.5*(u(1)+u(2))) + dv(u(2))) * (u(1) - u(2)) ...
               - (v(u(1)) + 4*v(0.5*(u(1)+u(2))) + v(u(2))));
       elseif strcmp(BCLt,'P') && (n == 0)
           J(1,1) = 1/h;  J(1,end) = -1/h;
       end
       
       % Evaluate Jacobian at current solution for the equation associated
       % with the right boundary
       if strcmp(BCRt,'D') && (n == 0)
           J(end,end) = 1/h;
       elseif strcmp(BCRt,'N')
           J(end,end-1) = (1/(6*h)) * ((dv(u(end-1)) + 2*dv(0.5*(u(end-1)+u(end)))) ...
               * (-u(end-1) + u(end)) ...
               - (v(u(end-1)) + 4*v(0.5*(u(end-1)+u(end))) + v(u(end))));
           J(end,end) = (1/(6*h)) * ((2*dv(0.5*(u(end-1)+u(end))) + dv(u(end))) ...
               * (-u(end-1) + u(end)) ...
               + (v(u(end-1)) + 4*v(0.5*(u(end-1)+u(end))) + v(u(end))));
       elseif strcmp(BCRt,'P') && (n == 0)
           J(end,1) = 1/h;  J(end,end) = -1/h;
       end
       
       % Compute increment and update solution
       y = - J\F;  u = uold + y;
       
       % Compute error, i.e. difference between consecutive iterations, and
       % update counter
       err1 = norm(u - uold);  err2 = norm(F);  n = n + 1;
    end
    
    %fprintf('Full solver: number of iterations %i, |F| = %5.5E\n', n, norm(F));
end