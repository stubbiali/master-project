clc
close all
clear variables
clear variables -global

path(path,'/usr/local/legendflex/legendflex')
path(path,'/usr/local/legendflex/setgetpos_V1.2')
path(path,'/usr/local/matlabfrag')
path(path,'/usr/local/')

screen = get(0,'screensize');
width = 730;  height = 540;  p = [screen(3)-width-100 screen(4)-height-100];
set(0, 'defaultFigurePosition', [p(1) p(2) width height]);

% Set default interpreter for text, plot and legend
set(0, 'defaultTextInterpreter', 'latex');
set(0, 'defaultLegendInterpreter', 'latex');

% Set default font size
set(0, 'defaultTextFontSize', 30);
set(0, 'defaultAxesFontSize', 25);

%% Fermi function

x = linspace(-3,3,1001);
T = [1 0.1 0.01];  y1 = 1./(1 + exp(-x/T(1)));  y2 = 1./(1 + exp(-x/T(2)));  y3 = 1./(1 + exp(-x/T(3)));

close all
figure(1)
hold off
h(1) = plot(x, y1, 'linewidth',1.2, 'color',rgb('Blue'));
hold on
h(2) = plot(x, y2, '--', 'linewidth',1.2, 'color',rgb('red'));
h(3) = plot(x, y3, ':', 'linewidth',1.75, 'color',rgb('DarkGreen'));

xlabel('$f$', 'UserData', 'matlabfrag:$v$')
ylabel('$\frac{f}{f}$', 'UserData', 'matlabfrag:$\big[ 1 + \exp(-v/T) \big]^{-1}$')

%legend('$T = 1$', '$T = 0.1$', '$T = 0.01$', 'location', 'best')


[legend_h, plot_h, object_h] = legendflex(h, {'$T = 1$','$T = 0.1$','$T = 0.01$'}, ...
    'anchor',{'se' 'se'}, 'buffer',[-10 10]);
c = get(legend_h,'children');
set(c(end-2), 'userdata', 'matlabfrag:$T = 0.01$');
set(c(end-1), 'userdata', 'matlabfrag:$T = 0.1$');
set(c(end), 'userdata', 'matlabfrag:$T = 1$');


grid on

xlim([-3 3])
ylim([-1.25 1.25])

%axis equal

%matlabfrag('newplot')

%% Hyperbolic tangent

x = linspace(-3,3,1001);
y = tanh(x);

%close all
figure(2)
plot(x, y, 'b', 'linewidth',1.2)

xlabel('$v$'); %, 'UserData', 'matlabfrag:$v$')
ylabel('$\tanh(v)$'); %, 'UserData', 'matlabfrag:$\tanh(v)$')

grid on
%axis equal
xlim([-3 3])
ylim([-1.25 1.25])

%axis equal

%matlabfrag('newplot')