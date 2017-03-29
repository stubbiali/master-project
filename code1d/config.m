% config Configuration file for Matlab library implementing model reduction 
% using neural networks. This script should be run at the beginning of any 
% Matlab session intended to use the library. 

clc
clear variables
clear variables -global
close all

%
% User defined settings:
% pos       position of bottom-left corner of a figure
% width     width of a figure
% height    height of a figure

p = [1500 700];  width = 840;  height = 630;

%
% Run
%

% Set default interpreter for text, plot and legend
set(0, 'defaultTextInterpreter', 'latex');
set(0, 'defaultLegendInterpreter', 'latex');

% Set default font size
set(0, 'defaultTextFontSize', 16);
set(0, 'defaultAxesFontSize', 18);

% Set default figure size
set(0, 'defaultFigurePosition', [p(1) p(2) width height]);