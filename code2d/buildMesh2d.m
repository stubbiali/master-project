function [mesh, domain] = buildMesh2d(geometry, varargin)
    % Initialize PDE model
    model = createpde;
    
    % Differentiate according to the shape of the domain
    if strcmp(geometry,'rectangle')
        % Set default values for options
        Ox = 0;  Oy = 0;  base = 1;  height = 1;  angle = 0;  Hmax = 0.1;
        
        % Catch user-defined values for options
        for i = 1:2:length(varargin)
            if strcmp(varargin{i},'origin')
                tmp = varargin{i+1};  Ox = tmp(1);  Oy = tmp(2);
            elseif strcmp(varargin{i},'base')
                base = varargin{i+1};
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
        x = Ox + [0 base*cos(angle) base*cos(angle)-height*sin(angle) -height*sin(angle)];
        y = Oy + [0 base*sin(angle) base*sin(angle)+height*cos(angle) height*cos(angle)];
        
        % Create the geometry and set it within model
        gd = [3 4 x y]';  dl = decsg(gd);  geometryFromEdges(model,dl);
        
        % Build the mesh
        mesh = generateMesh(model,'Hmax',Hmax);
        
        % Convert to a mesh2d object
        mesh = mesh2d(mesh.Nodes,mesh.Elements);
        
        % Store domain details in a struct; these may be useful for constructing
        % the map from the physical to the reference domain
        domain = struct('origin',[Ox Oy]', 'base',base, 'height',height, 'angle',angle);
    else
        error('Unknown input geometry. Available geometries:\n\t''rectangle''.')
    end
end
        
        
    