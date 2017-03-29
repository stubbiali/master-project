function [mesh, u, A, rhs] = LinearPoisson2dFEP1(K, f, BCs_t, BCs_v, BCe_t, BCe_v, ...
    BCn_t, BCn_v, BCw_t, BCw_v, geometry, varargin)
    % Build the mesh onto the physical domain
    [mesh,domain] = buildMesh2d(geometry,varargin{:});
    
    % Extract nodes and element-node connectivity
    nodes = mesh.nodes;   elems = mesh.elems;
    
    % Map the physical domain to the reference domain, i.e. a unit square
    % Note that from now on, all variables which refer to the reference
    % domain have '_r' as suffix
    if strcmp(geometry,'rectangle')
        % Some shortcuts
        orig = domain.origin;  b = domain.base;  
        h = domain.height;  alpha = domain.angle;
        
        % Build rotational, translational and scale matrices
        R = [cos(alpha) sin(alpha); -sin(alpha) cos(alpha)];
        T = repmat(-orig, [1,mesh.getNumNodes()]);  S = [1/b 0; 0 1/h];
        
        % Map the vertices
        nodes_r = S*(R*(nodes + T));        
    end
    
    % Create a mesh2d object on the reference domain
    mesh_r = mesh2d(nodes_r,elems);
    
    % Build stiffness matrix and right-hand side
    [A,rhs] = getLinearPoisson2dFEP1System_r(geometry, domain, mesh_r, ...
        K, f, BCs_t, BCs_v, BCe_t, BCe_v, BCn_t, BCn_v, BCw_t, BCw_v);

    % Solve the system
    u = A \ rhs;
end