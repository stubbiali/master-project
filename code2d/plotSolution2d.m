function plotSolution2d(mesh, u, varargin)
    % Shortcuts
    nodes = mesh.nodes;  elems = mesh.elems;
    
    % Open a new figure and plot
    figure;
    
    % Plot
    trisurf(elems', nodes(1,:)', nodes(2,:)', u)
    
    % Set default values for options
    Title = '';  Xlabel = '$x$';  Ylabel = '$y$';  Zlabel = '$u$';
    
    % Catch user-defined values for options
    for i = 1:2:length(varargin)
        if strcmp(varargin{i},'title')
            Title = varargin{i+1};
        elseif strcmp(varargin{i},'xlabel')
            Xlabel = varargin{i+1};
        elseif strcmp(varargin{i},'ylabel')
            Ylabel = varargin{i+1};
        elseif strcmp(varargin{i},'zlabel')
            Zlabel = varargin{i+1};
        else
            warning('Unknown option ''%s''. Ignored.',varargin{i})
        end
    end
    
    % Set plot options
    title(Title);  xlabel(Xlabel);  ylabel(Ylabel);  zlabel(Zlabel);  
    axis equal 
end