function perf = myperf(net,varargin)
    global gh gV;
    if nargin >= 3
        targets = varargin{1};
        outputs = varargin{2};
        perf = gh * norm(targets - gV*outputs)^2;
    else
        perf = 100;
    end
end