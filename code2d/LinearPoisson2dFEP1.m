% LinearPoisson2dFEP1 Linear Finite Elements (FE-P1) solver for the linear
% two-dimensional Poisson equation on a general domain. Note that the
% problem is mapped and solved onto the reference, unit square, properly
% accounting for the transformation map in the numerical method.
%
% [mesh, u] = LinearPoisson2dFEP1(K, f, BCs_t, BCs_v, BCe_t, BCe_v, ...
%    BCn_t, BCn_v, BCw_t, BCw_v, geometry)
% [mesh, u] = LinearPoisson2dFEP1(K, f, BCs_t, BCs_v, BCe_t, BCe_v, ...
%    BCn_t, BCn_v, BCw_t, BCw_v, 'rectangle', 'origin',origin, 'base',base, ...
%   'height',height, 'angle',angle, 'Hmax',Hmax)
%
% \param K          handle to a function evaluating the viscosity matrix in 
%                   a generic two-dimensional point, passed as a two-elements 
%                   array; see MatrixFunctions folder
% \param f          handle to a function evaluating the forcing term in a
%                   generic two-dimensional point, passed as a two-elements
%                   array ; see ScalarFunctions folder
% \param BCs_t      type of boundary condition onto the South edge:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
% \param BCs_v      handle function to South boundary condition
% \param BCe_t      type of boundary condition onto the East edge:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
% \param BCe_v      handle function to East boundary condition
% \param BCn_t      type of boundary condition onto the North edge:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
% \param BCn_v      handle function to North boundary condition
% \param BCw_t      type of boundary condition onto the West edge:
%                   - 'D': Dirichlet
%                   - 'N': Neumann
% \param BCw_v      handle function to West boundary condition
% \param geometry   string reporting the shape of the domain:
%                   - 'rectangle': rectangular domain
% \out   mesh       computational mesh onto the physical domain; this is a
%                   mesh2d object
% \out   u          discrete solution in each vertex of the mesh
%
% Optional arguments for rectangular domains:
% \param origin     location of the "origin" of the rectangle, i.e. the 
%                   vertex which will be coincide with the actual origin  
%                   once the domain has been mapped to the reference
%                   square; defualt is [0,0]
% \param width      width of the rectangle, namely the length of the edge 
%                   whose end-points are the origin of the rectangle and 
%                   the next vertex in counterclockwise sense; default is 1
% \param height     height of the rectangle, namely the length of the edge 
%                   whose end-points are the origin of the rectangle and 
%                   the next vertex in clockwise sense; default is 1
% \param angle      angle (in radians) between the x-axis and the base; it
%                   should be in the interval (-pi/2,pi/2); default is 0
% \param Hmax       length of the longest edge in the mesh (approximately);
%                   default is 0.1

function [mesh, u] = LinearPoisson2dFEP1(K, f, BCs_t, BCs_v, BCe_t, BCe_v, ...
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
        orig = domain.origin;  w = domain.width;  
        h = domain.height;  alpha = domain.angle;
        
        % Build rotational, translational and scale matrices
        R = [cos(alpha) sin(alpha); -sin(alpha) cos(alpha)];
        T = repmat(-orig, [1,mesh.getNumNodes()]);  S = [1/w 0; 0 1/h];
        
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