% NonLinearPoisson1dFEP1ReducedNewton Linear finite elements (FE-P1) 
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
% This system is solved using Newton's method. The norm of the difference 
% between consecutive iterations is used as stopping criterium. The maximum 
% number of iterations and the tolerance for the stopping criterium can be 
% tuned by the user.
%
% [x,alpha] = NonLinearPoisson1dFEP1ReducedNewton(a, b, K, v, dv, f, BCLt, BCLv, ...
%   BCRt, BCRv, V)
% [x,alpha] = NonLinearPoisson1dFEP1ReducedNewton(a, b, K, v, dv, f, BCLt, BCLv, ...
%   BCRt, BCRv, V, 'iter', iter, 'tol', tol)
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
% \param V      reduced basis stored as a matrix (K,L)
% \param iter   maximum number of iterations for Newton's method
% \param tol    tolerance for stopping criterium for Newton's method
% \out   x      grid
% \out   u      numerical solution

function [x,alpha] = NonLinearPoisson1dFEP1ReducedNewton(a, b, K, v, dv, f, ...
    BCLt, BCLv, BCRt, BCRv, V, varargin)
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
    
    % Allocate memory for evaluations of the nonlinear function and its
    % Jacobian with respect to the coefficients $\boldsymbol{\alpha}$
    L = size(V,2);  F = zeros(K,1);  J = zeros(K,L);
        
    % Run Newton's solver
    err1 = 1+tol1;  err2 = 1+tol2; n = 0;
    while ((err1 > tol1) || (err2 > tol2)) && (n < iter)
       % Shortcuts
       ul = V(1:end-2,:)*alpha;  uc = V(2:end-1,:)*alpha;  
       ur = V(3:end,:)*alpha;  ulc = 0.5*(ul+uc);  ucr = 0.5*(uc+ur);
       
       % Save current solution
       alphaold = alpha;
              
       % Evaluate nonlinear equations associated with internal points of
       % the domain
       F(2:end-1) = (1/(6*h)) * (- (v(ul) + 4*v(ulc) + v(uc)) .* ul ...
           + (v(ul) + 4*v(ulc) + 2*v(uc) + 4*v(ucr) + v(ur)) .* uc ...
           - (v(uc) + 4*v(ucr) + v(ur)) .* ur) - rhs(2:end-1);
       
       % Evaluate nonlinear equation associated with left boundary
       if strcmp(BCLt,'D')
           F(1) = (1/h) * V(1,:)*alpha - rhs(1);
       elseif strcmp(BCLt,'N')
           F(1) = (1/(6*h)) * ((v(V(1,:)*alpha) + 4*v(0.5*(V(1,:)+V(2,:))*alpha) ...
               + v(V(2,:)*alpha)) * (V(1,:) - V(2,:))*alpha) - rhs(1);
       elseif strcmp(BCLt,'P')
           F(1) = (1/h) * (V(1,:) - V(end,:))*alpha;
       end
       
        % Evaluate nonlinear equation associated with right boundary
       if strcmp(BCRt,'D')
           F(end) = (1/h) * V(end,:)*alpha - rhs(end);
       elseif strcmp(BCRt,'N')
           F(end) = (1/(6*h)) * ((v(V(end-1,:)*alpha) + ...
               4*v(0.5*(V(end-1,:)+V(end,:))*alpha) + v(V(end,:)*alpha)) * ...
               (-V(end-1,:) + V(end,:))*alpha) - rhs(end);
       elseif strcmp(BCRt,'P')
           F(end) = (1/h) * (V(1,:) - V(end,:))*alpha;
       end
       
       % Some shortcuts useful for evaluating Jacobian of F with respect to
       % alpha at current solution for equations associated with internal points
       Ul = repmat(V(1:end-2,:)*alpha, [1,L]);
       Uc = repmat(V(2:end-1,:)*alpha, [1,L]);
       Ur = repmat(V(3:end,:)*alpha, [1,L]);
       Ulc = 0.5*(Ul+Uc);  Ucr = 0.5*(Uc+Ur);
       Vl = V(1:end-2,:);  Vc = V(2:end-1,:);  Vr = V(3:end,:);
       Vlc = 0.5*(Vl+Vc);  Vcr = 0.5*(Vc+Vr);
       
       % Evaluate Jacobian for equations associated with internal points
       J(2:end-1,:) = (1/(6*h)) * ( ...
           - (dv(Ul).*Vl + 4*dv(Ulc).*Vlc + dv(Uc).*Vc) .* Ul ...
           - (v(Ul) + 4*v(Ulc) + v(Uc)) .* Vl ...
           + (dv(Ul).*Vl + 4*dv(Ulc).*Vlc + 2*dv(Uc).*Vc + 4*dv(Ucr).*Vcr + ...
              dv(Ur).*Vr) .* Uc ...
           + (v(Ul) + 4*v(Ulc) + 2*v(Uc) + 4*v(Ucr) + v(Ur)) .* Vc ...
           - (dv(Uc).*Vc + 4*dv(Ucr).*Vcr + dv(Ur).*Vr) .* Ur ...
           - (v(Uc) + 4*v(Ucr) + v(Ur)) .* Vr);
             
       % Evaluate Jacobian for the equation associated with the left boundary
       if strcmp(BCLt,'D') && (n == 0)
           J(1,:) = V(1,:)/h;
       end
       %{
       elseif strcmp(BCLt,'N')
           J(1,1) = (1/(6*h)) * ((dv(u(1)) + 2*dv(0.5*(u(1)+u(2)))) * (u(1) - u(2)) ...
               + (v(u(1)) + 4*v(0.5*(u(1)+u(2))) + v(u(2))));
           J(1,2) = (1/(6*h)) * ((2*dv(0.5*(u(1)+u(2))) + dv(u(2))) * (u(1) - u(2)) ...
               - (v(u(1)) + 4*v(0.5*(u(1)+u(2))) + v(u(2))));
       elseif strcmp(BCLt,'P') && (n == 0)
           J(1,1) = 1/h;  J(1,end) = -1/h;
       end
       %}
       
       % Evaluate Jacobian for the equation associated with the right boundary
       if strcmp(BCRt,'D') && (n == 0)
           J(end,:) = V(end,:)/h;
       end
       %{
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
       %}
       
       % Compute Jacobian for reduced system
       Jr = V'*J;
       
       % Compute increment and update solution
       y = - Jr \ (V'*F);  alpha = alphaold + y;
       
       % Compute error, i.e. difference between consecutive iterations, and
       % update counter
       err1 = norm(alpha - alphaold);  err2 = norm(F);  n = n + 1;
    end
    
    %fprintf('Reduced solver: number of iterations %i, |F| = %5.5E\n', n, norm(F));
end