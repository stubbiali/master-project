% plotMesh2d Utility to plot a two-dimensional triangular grid.
%
% plotMesh2d(mesh)
% plotMesh2d(mesh, 'title',Title, 'xlabel',Xlabel, 'ylabel',Ylabel)
%
% \param mesh   computational mesh as a mesh2d object
% \param Title  string with the title; default is ''
% \param Xlabel string with the x-label; default is '$x$'
% \param Ylabel string with the y-label; default is '$y$'

function plotMesh2d(mesh, varargin)
    % Shortcuts
    nodes = mesh.nodes;  elems = mesh.elems;
    
    % Open a new figure and plot
    figure;
    
    % Plot looping over all elements
    for i = 1:mesh.getNumElems()
       v = elems(:,i);
       x = [nodes(1,v(1)) nodes(1,v(2)) nodes(1,v(3)) nodes(1,v(1))]; 
       y = [nodes(2,v(1)) nodes(2,v(2)) nodes(2,v(3)) nodes(2,v(1))];
       plot(x,y,'b')
       hold on
    end
    
    % Set default values for options
    Title = '';  Xlabel = '$x$';  Ylabel = '$y$';
    
    % Catch user-defined values for options
    for i = 1:2:length(varargin)
        if strcmp(varargin{i},'title')
            Title = varargin{i+1};
        elseif strcmp(varargin{i},'xlabel')
            Xlabel = varargin{i+1};
        elseif strcmp(varargin{i},'ylabel')
            Ylabel = varargin{i+1};
        else
            warning('Unknown option ''%s''. Ignored.',varargin{i})
        end
    end
    
    % Set plot options
    title(Title);  xlabel(Xlabel);  ylabel(Ylabel);  
    axis equal 
end