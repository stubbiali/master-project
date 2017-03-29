% getDiscreteContinuousErrorL2 Compute distance in discrete L2-norm between
% a discrete linear Finite Elements function and a continuous function.
%
% e = getDiscreteContinuousErrorL2(mesh, u, fun)
% \param mesh   computational mesh; this should be an object of class mesh2d
% \param u      discrete solution
% \param fun    handle to continuous function
% \out   e      error

function e = getDiscreteContinuousErrorL2(mesh, u, fun)
    % Some shortcuts
    nodes = mesh.nodes;  elems = mesh.elems;  Ne = mesh.getNumElems();
    
    % Define quadrature nodes and weights on the reference triangle
    % A three-points Gauss quadrature rule is used, but it may easily
    % changed by accordingly changed the nodes and the weights
    q1 = [1/6 1/6]';  q2 = [2/3 1/6]';  q3 = [1/6 2/3]';
    w1 = 1/3;  w2 = 1/3;  w3 = 1/3;
    
    % Initialize error
    e = 0;
    
    % Loop over all elements
    for n = 1:Ne
        % Extract vertices of the triangle
        ia = elems(1,n);   ib = elems(2,n);   ic = elems(3,n);
        va = nodes(:,ia);  vb = nodes(:,ib);  vc = nodes(:,ic);
        
        % Compute Jacobian of the map J*x+va from reference to current triangle
        J = [vb(1)-va(1) vc(1)-va(1); vb(2)-va(2) vc(2)-va(2)];
        dJ = abs(J(1,1)*J(2,2) - J(1,2)*J(2,1));
        
        % Evaluate coefficients for the plane representing the discrete
        % solution over the element
        A = [va(1) va(2) 1; vb(1) vb(2) 1; vc(1) vc(2) 1];  b = [u(ia) u(ib) u(ic)]';
        coef = A\b;
        
        % Map quadrature nodes to physical element
        p1 = J*q1 + va;  p2 = J*q2 + va;  p3 = J*q3 + va;
        
        % Update error
        e = e + 0.5 * (w1 * (coef(1:2)'*p1 + coef(3) - fun(p1))^2 + ...
            w2 * (coef(1:2)'*p2 + coef(3) - fun(p2))^2 + ...
            w3 * (coef(1:2)'*p3 + coef(3) - fun(p3))^2) * dJ;
    end
    
    % Make square root
    e = sqrt(e);
end