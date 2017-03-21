function perfs = apply(t,y,e,param)
%MSE.APPLY

% Copyright 2012 The MathWorks, Inc.

global gh gV;
perfs = gh*norm(t - gV*y)^2;
