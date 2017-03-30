% LinearPoisson2dFEP1 Linear Finite Elements (FE-P1) solver for the linear
% two-dimensional Poisson equation on a general domain. Note that the
% problem is mapped and solved onto the reference, unit square, properly
% accounting for the transformation map in the numerical method.
%
% [mesh, u] = LinearPoisson2dFEP1(K, f, BCs_t, BCs_v, BCe_t, BCe_v, ...
%    BCn_t, BCn_v, BCw_t, BCw_v, geometry)
% [mesh, u] = LinearPoisson2dFEP1(K, f, BCs_t, BCs_v, BCe_t, BCe_v, ...
%    BCn_t, BCn_v, BCw_t, BCw_v, 'quadrilateral', 'A',A, 'B',B, 'C',C, ...
%    'D',D, 'Hmax',Hmax)
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
%                   - 'quadrilateral': quadrilateral domain
% \out   mesh_r     mesh2d over the reference domain
% \out   mesh       mesh2d over the physical domain     
% \out   u          discrete solution in each vertex of the mesh
%
% Optional arguments for quadrilateral domains:
% \param A          coordinates of the first vertex; default is [0 0]'
% \param B          coordinates of the second vertex; default is [1 0]'
% \param C          coordinates of the third vertex; default is [1 1]'
% \param D          coordinates of the fourth vertex; default is [0 1]'
% \param Hmax       length of the longest edge in the mesh over the
%                   reference domain; default is 0.05

function [mesh_r, mesh, u] = LinearPoisson2dFEP1(K, f, BCs_t, BCs_v, BCe_t, BCe_v, ...
    BCn_t, BCn_v, BCw_t, BCw_v, geometry, varargin)
    % Build the mesh onto the reference and physical domain
    [mesh_r,mesh,domain] = buildMesh2d(geometry,varargin{:});
    
    % Build stiffness matrix and right-hand side
    [A,rhs] = getMappedLinearPoisson2dFEP1System(geometry, domain, mesh_r, ...
        K, f, BCs_t, BCs_v, BCe_t, BCe_v, BCn_t, BCn_v, BCw_t, BCw_v);

    % Solve the system
    u = A \ rhs;
end