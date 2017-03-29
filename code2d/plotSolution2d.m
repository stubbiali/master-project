% plotSolution2d Utility to plot a two-dimensional linear Finite Elements
% solution over a triangular grid.
%
% plotSolution2d(mesh, u)
% plotSolution2d(mesh, u, 'title',Title, 'xlabel',Xlabel, 'ylabel',Ylabel, ...
%   'zlabel',Zlabel)
%
% \param mesh   computational mesh as a mesh2d object
% \param u      numerical solution
% \param Title  string with the title; default is ''
% \param Xlabel string with the x-label; default is '$x$'
% \param Ylabel string with the y-label; default is '$y$'
% \param Zlabel string with the z-label; default is '$u$'

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
    drawnow
end