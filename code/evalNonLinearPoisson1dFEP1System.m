% evalNonLinearPoisson1dFEP1System Evaluate the system of nonlinear
% equations yielding by the linear finite elements (FE-P1) method applied to
% the nonlinear one-dimensional Poisson equation $-(v(u) u(x)')' = f(x)$ in
% the unknown $u = u(x)$, $x \in [a,b]$.
% 
% F = evalNonLinearPoisson1dFEP1System(u, h, v, rhs, BCLt, BCRt)
% \param u      point where to evaluate the equations
% \param h      grid size
% \param v      viscosity $v = v(u)$ as handle function
% \param rhs    discretized right-hand side; see getLinearPoisson1dFEP1rhs_f
% \param BCLt   type of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRt   type of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \out   F      evaluation of the nonlinear system

function [F,J] = evalNonLinearPoisson1dFEP1System(u, h, v, dv, rhs, BCLt, BCRt)    
    % Some shortcuts
    ul = u(1:end-2);  uc = u(2:end-1);  ur = u(3:end);  
    ulc = 0.5*(ul+uc);  ucr = 0.5*(uc+ur);
    
    %
    % Function
    %
    
    % Initialize vector of evaluations
    F = zeros(size(u));

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
    
    %
    % Jacobian
    %
    
    % Allocate memory
    J = zeros(size(u,1),size(u,1));
    
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
    if strcmp(BCLt,'D') 
        J(1,1) = 1/h;
    elseif strcmp(BCLt,'N')
        J(1,1) = (1/(6*h)) * ((dv(u(1)) + 2*dv(0.5*(u(1)+u(2)))) * (u(1) - u(2)) ...
           + (v(u(1)) + 4*v(0.5*(u(1)+u(2))) + v(u(2))));
        J(1,2) = (1/(6*h)) * ((2*dv(0.5*(u(1)+u(2))) + dv(u(2))) * (u(1) - u(2)) ...
           - (v(u(1)) + 4*v(0.5*(u(1)+u(2))) + v(u(2))));
    elseif strcmp(BCLt,'P')
        J(1,1) = 1/h;  J(1,end) = -1/h;
    end

    % Evaluate Jacobian at current solution for the equation associated
    % with the right boundary
    if strcmp(BCRt,'D') 
        J(end,end) = 1/h;
    elseif strcmp(BCRt,'N')
        J(end,end-1) = (1/(6*h)) * ((dv(u(end-1)) + 2*dv(0.5*(u(end-1)+u(end)))) ...
           * (-u(end-1) + u(end)) ...
           - (v(u(end-1)) + 4*v(0.5*(u(end-1)+u(end))) + v(u(end))));
        J(end,end) = (1/(6*h)) * ((2*dv(0.5*(u(end-1)+u(end))) + dv(u(end))) ...
           * (-u(end-1) + u(end)) ...
           + (v(u(end-1)) + 4*v(0.5*(u(end-1)+u(end))) + v(u(end))));
    elseif strcmp(BCRt,'P')
        J(end,1) = 1/h;  J(end,end) = -1/h;
    end
end