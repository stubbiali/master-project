% README file for the folder ScalarFunctions.
% This folder stores different MATLAB functions receiving a two-elements
% array and returning a scalar. These functions are particularly useful to
% represent the forcing term and the boundary conditions for the two-dimensional
% Poisson equation. Indeed, direct functions should be prefered to handle
% functions when performance matters.
%
% The available functions are listed hereunder:
% - zerofun        : y = 0
% - onefun         : y = 1
% - unitperiodsinx : y = sin(2*pi*x(1))
% - unitperiodsinx2: y = 4*pi*pi*sin(2*pi*x(1))
% - sincos         : y = sin(x(1))*cos(x(2))
% - doublesincos   : y = 2*sin(x(1))*cos(x(2))