% evalNonLinearPoisson1dFEP1SystemReduced Evaluate the system of nonlinear 
% equations yielded by the linear finite elements (FE-P1) method applied to
% the reduced nonlinear one-dimensional Poisson equation $-(v(u) u(x)')' = f(x)$ 
% in the unknown $u = u(x)$, $x \in [a,b]$.
% 
% y = evalNonLinearPoisson1dFEP1SystemReduced(alpha, h, v, rhs, V, BCLt, BCRt)
% \param alpha  point where to evaluate the equations
% \param h      grid size
% \param v      viscosity $v = v(u)$ as handle function
% \param rhs    discretized right-hand side; see getLinearPoisson1dFEP1rhs_f
% \param V      matrix whose columns store the vectors of the reduced basis
% \param BCLt   type of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRt   type of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \out   y      evaluation of the nonlinear equations

function y = evalNonLinearPoisson1dFEP1ReducedSystem(alpha, h, v, rhs, V, BCLt, BCRt)
    % Some shortcuts
    ul = V(1:end-2,:)*alpha;  uc = V(2:end-1,:)*alpha;  
    ur = V(3:end,:)*alpha;  ulc = 0.5*(ul+uc);  ucr = 0.5*(uc+ur);
    
    % Initialize vector of evaluations of full system at the reduced
    % solution
    F = zeros(size(V,1),1);

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

    % Project full system onto the reduced space
    y = V'*F;
end