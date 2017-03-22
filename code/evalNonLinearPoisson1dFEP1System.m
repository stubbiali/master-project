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

function [F,J] = evalNonLinearPoisson1dFEP1System(u)
    global gh gv gdv grhs gBCLt gBCRt
    
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
    F(2:end-1) = (1/(6*gh)) * (- (gv(ul) + 4*gv(ulc) + gv(uc)) .* ul ...
       + (gv(ul) + 4*gv(ulc) + 2*gv(uc) + 4*gv(ucr) + gv(ur)) .* uc ...
       - (gv(uc) + 4*gv(ucr) + gv(ur)) .* ur) - grhs(2:end-1);

    % Evaluate nonlinear equation associated with left boundary
    if strcmp(gBCLt,'D')
       F(1) = (1/gh) * u(1) - grhs(1);
    elseif strcmp(gBCLt,'N')
       F(1) = (1/(6*gh)) * ((gv(u(1)) + 4*gv(0.5*(u(1)+u(2))) + gv(u(2))) * ...
           (u(1) - u(2))) - grhs(1);
    elseif strcmp(gBCLt,'P')
       F(1) = (1/gh) * (u(1) - u(end));
    end

    % Evaluate nonlinear equation associated with right boundary
    if strcmp(gBCRt,'D')
       F(end) = (1/gh) * u(end) - grhs(end);
    elseif strcmp(gBCRt,'N')
       F(end) = (1/(6*gh)) * ((gv(u(end-1)) + 4*gv(0.5*(u(end-1)+u(end))) + gv(u(end))) * ...
           (-u(end-1) + u(end))) - grhs(end);
    elseif strcmp(gBCRt,'P')
       F(end) = (1/gh) * (u(1) - u(end));
    end
    
    %
    % Jacobian
    %
    
    % Allocate memory
    J = zeros(size(u,1),size(u,1));
    
    % Evaluate Jacobian at current solution for equations associated
    % with internal points
    J(2:end-1,1:end-2) = (1/(6*gh)) * diag(- (gdv(ul) + 2*gdv(ulc)) .* ul ...
        - (gv(ul) + 4*gv(ulc) + gv(uc)) ...
        + (gdv(ul) + 2*gdv(ulc)) .* uc);
    J(2:end-1,2:end-1) = J(2:end-1,2:end-1) + ...
        (1/(6*gh)) * diag(- (2*gdv(ulc) + gdv(uc)) .* ul ...
        + (2*gdv(ulc) + 2*gdv(uc) + 2*gdv(ucr)) .* uc ...
        + (gv(ul) + 4*gv(ulc) + 2*gv(uc) + 4*gv(ucr) + gv(ur)) ...
        - (gdv(uc) + 2*gdv(ucr)) .* ur);
    J(2:end-1,3:end) = J(2:end-1,3:end) + ...
        (1/(6*gh)) * diag((2*gdv(ucr) + gdv(ur)) .* uc ...
        - (2*gdv(ucr) + gdv(ur)) .* ur ...
        - (gv(uc) + 4*gv(ucr) + gv(ur)));

    % Evaluate Jacobian at current solution for the equation associated
    % with the left boundary
    if strcmp(gBCLt,'D') 
        J(1,1) = 1/gh;
    elseif strcmp(gBCLt,'N')
        J(1,1) = (1/(6*gh)) * ((gdv(u(1)) + 2*gdv(0.5*(u(1)+u(2)))) * (u(1) - u(2)) ...
           + (gv(u(1)) + 4*gv(0.5*(u(1)+u(2))) + gv(u(2))));
        J(1,2) = (1/(6*gh)) * ((2*gdv(0.5*(u(1)+u(2))) + gdv(u(2))) * (u(1) - u(2)) ...
           - (gv(u(1)) + 4*gv(0.5*(u(1)+u(2))) + gv(u(2))));
    elseif strcmp(gBCLt,'P')
        J(1,1) = 1/gh;  J(1,end) = -1/gh;
    end

    % Evaluate Jacobian at current solution for the equation associated
    % with the right boundary
    if strcmp(gBCRt,'D') 
        J(end,end) = 1/gh;
    elseif strcmp(gBCRt,'N')
        J(end,end-1) = (1/(6*gh)) * ((gdv(u(end-1)) + 2*gdv(0.5*(u(end-1)+u(end)))) ...
           * (-u(end-1) + u(end)) ...
           - (gv(u(end-1)) + 4*gv(0.5*(u(end-1)+u(end))) + gv(u(end))));
        J(end,end) = (1/(6*gh)) * ((2*gdv(0.5*(u(end-1)+u(end))) + gdv(u(end))) ...
           * (-u(end-1) + u(end)) ...
           + (gv(u(end-1)) + 4*gv(0.5*(u(end-1)+u(end))) + gv(u(end))));
    elseif strcmp(gBCRt,'P')
        J(end,1) = 1/gh;  J(end,end) = -1/gh;
    end
end