function y = foo2(x)
    global c
    y = 1/sqrt(c*pi) * exp(-0.5*(x.^2));
end