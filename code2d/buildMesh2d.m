% buildMesh2d Construct a triangular grid a two-dimensional planar domain.
% The shape of the domain may be:
% - rectangular
% 
% [mesh,domain] = buildMesh2d(geometry)
% [mesh,domain] = buildMesh2d('rectangle', 'origin',origin, 'base',base, ...
%   'height',height, 'angle',angle, 'Hmax',Hmax)
%
% \param geometry   string reporting the shape of the domain:
%                   - 'rectangle': rectangular domain
% \out   mesh       the triangular mesh; object of class mesh2d
% \out   domain     struct storing details about the domain; these can be set
%                   through the optional input arguments
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

function [mesh, domain] = buildMesh2d(geometry, varargin)
    % Initialize PDE model
    model = createpde;
    
    % Differentiate according to the shape of the domain
    if strcmp(geometry,'rectangle')
        % Set default values for options
        Ox = 0;  Oy = 0;  width = 1;  height = 1;  angle = 0;  Hmax = 0.1;
        
        % Catch user-defined values for options
        for i = 1:2:length(varargin)
            if strcmp(varargin{i},'origin')
                tmp = varargin{i+1};  Ox = tmp(1);  Oy = tmp(2);
            elseif strcmp(varargin{i},'width')
                width = varargin{i+1};
            elseif strcmp(varargin{i},'height')
                height = varargin{i+1};
            elseif strcmp(varargin{i},'angle')
                angle = varargin{i+1};
                if (angle < -pi/2 || angle > pi/2)
                    warning(['Specified rotational angle not in the interval ' ...
                        '[-pi/2,pi/2]. Ignored.'])
                    angle = 0;
                end
            elseif strcmp(varargin{i},'Hmax')
                Hmax = varargin{i+1};
            else
                warning('Unknown option ''%s''. Ignored.',varargin{i}{1})
            end
        end
        
        % Set the vertices
        x = Ox + [0 width*cos(angle) width*cos(angle)-height*sin(angle) -height*sin(angle)];
        y = Oy + [0 width*sin(angle) width*sin(angle)+height*cos(angle) height*cos(angle)];
        
        % Create the geometry and set it within model
        gd = [3 4 x y]';  dl = decsg(gd);  geometryFromEdges(model,dl);
        
        % Build the mesh
        mesh = generateMesh(model,'Hmax',Hmax);
        
        % Convert to a mesh2d object
        mesh = mesh2d(mesh.Nodes,mesh.Elements);
        
        % Store domain details in a struct; these may be useful for constructing
        % the map from the physical to the reference domain
        domain = struct('origin',[Ox Oy]', 'width',width, 'height',height, 'angle',angle);
    else
        error('Unknown input geometry. Available geometries:\n\t''rectangle''.')
    end
end
        
        
    