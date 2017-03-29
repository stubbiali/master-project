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

function [y,Jr] = evalNonLinearPoisson1dFEP1ReducedSystem(alpha, h, v, dv, rhs, V, BCLt, BCRt)
    %
    % Function
    %
    
    % Some shortcuts
    L = length(alpha);  M = size(V,1);
    ul = V(1:end-2,:)*alpha;  uc = V(2:end-1,:)*alpha;  
    ur = V(3:end,:)*alpha;  ulc = 0.5*(ul+uc);  ucr = 0.5*(uc+ur);
        
    % Initialize vector of evaluations of full system at the reduced
    % solution
    F = zeros(M,1);

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
    
    %
    % Jacobian
    %
    
    % Allocate memory
    J = zeros(M,L);
    
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
    if strcmp(BCLt,'D')
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
    if strcmp(BCRt,'D')
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
end