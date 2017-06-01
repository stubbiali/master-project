clc
close all
clear variables
clear variables -global

path(path,'/usr/local/legendflex/legendflex')
path(path,'/usr/local/legendflex/setgetpos_V1.2')
path(path,'/usr/local/matlabfrag')

screen = get(0,'screensize');
width = 650;  height = 500;  p = [screen(3)-width-100 screen(4)-height-100];
set(0, 'defaultFigurePosition', [p(1) p(2) width height]);

% Set default interpreter for text, plot and legend
set(0, 'defaultTextInterpreter', 'latex');
set(0, 'defaultLegendInterpreter', 'latex');

% Set default font size
set(0, 'defaultTextFontSize', 16);
set(0, 'defaultAxesFontSize', 18);

%% Fermi function

x = linspace(-3,3,1001);
T = [1 0.1 0.01];  y1 = 1./(1 + exp(-x/T(1)));  y2 = 1./(1 + exp(-x/T(2)));  y3 = 1./(1 + exp(-x/T(3)));

figure(1)
hold off
h(1) = plot(x, y1, 'b', 'linewidth',1.05);
hold on
h(2) = plot(x, y2, 'c', 'linewidth',1.05);
h(3) = plot(x, y3, 'g', 'linewidth',1.05);

xlabel('$f$', 'UserData', 'matlabfrag:$v$')
ylabel('$\frac{f}{f}$', 'UserData', 'matlabfrag:\small{$\dfrac{1}{1 + \exp(-v/T)}$}')
[legend_h, plot_h, object_h] = legendflex(h, {'$T = 1$','$T = 0.1$','$T = 0.01$'});

c = get(legend_h,'children');
set(c(end-2), 'userdata', 'matlabfrag:$T = 0.01$');
set(c(end-1), 'userdata', 'matlabfrag:$T = 0.1$');
set(c(end), 'userdata', 'matlabfrag:$T = 1$');

grid on
%axis equal
xlim([-3 3])
ylim([-1.25 2.25])

matlabfrag('newplot')

%% Hyperbolic tangent

x = linspace(-3,3,1001);
y = tanh(x);

figure(2)
plot(x, y, 'b', 'linewidth',1.05)

xlabel('$f$'); %, 'UserData', 'matlabfrag:$v$')
ylabel('$\tanh(v)$'); %, 'UserData', 'matlabfrag:$\tanh(v)$')

grid on
%axis equal
xlim([-3 3])
ylim([-2.25 2.25])

matlabfrag('newplot')