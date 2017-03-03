% getLinearPoisson1dFEP2rhs_f Compute right-hand side for the linear system
% yielded by the application of quadratic finite elements (FE-P2) to the
% linear one-dimensional Poisson equation $-u''(x) = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions. The "f" in the function name stands for "fast". Indeed, 
% persistent variables are used so to perform computations only when needed.
%
% rhs = getLinearPoisson1dFEP1rhs_f(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param f      RHS; this may be either an handle function or a cell array
%               of handle functions; in the latter case, the solution is
%               computed for each RHS
% \param BCLt   kind of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCLv   value of left boundary condition
% \param BCRt   kind of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRv   value of right boundary condition
% \out   rhs    right-hand side

function rhs = getLinearPoisson1dFEP2rhs_f(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
	% Declare persistent variables
	persistent pa pb pK ph pf pBCLt pBCLv pBCRt pBCRv prhs;
	
	% Exploiting persistent variables, compute stiffness matrix
	% only when strictly necessary, i.e. the first time this function
	% is called or when input arguments have changed with respect
	% to last call
	if isempty(pa)
		todo = 1;
	elseif (pa ~= a) || (pb ~= b) || (pK ~= K) || (pBCLt ~= BCLt) ...
		|| (pBCRt ~= BCRt)
		todo = 1;
	else
		funinfo(1) = functions(pf);
		funinfo(2) = functions(f);
		todo = ~(strcmp(func2str(pf),func2str(f)) && ...
			isequal(funinfo(1).workspace{1},funinfo(2).workspace{1}));
	end
	
	% If needed, compute the RHS
	if todo
		% Set persistent variables
		pa = a;  pb = b;  pK = K;  pf = f;
		
		% Build a uniform grid over the domain [a,b]
		ph = (b-a) / (K-1);
		xv = linspace(a,b,K)';
		
		% Get the nodes
		M = 2*K-1;
		xn = linspace(a,b,M)';
		
		% Allocate memory for RHS
		prhs = zeros(M,1);
		
		% Odd indeces
		g = @(t) f(t) .* 2.*(t-xn(2)).*(t-xn(3))./(ph^2); 
		prhs(1) = integral(g, xv(1), xv(2));
		for i = 3:2:M-2
		    g = @(t) f(t) .* (2.*(t-xn(i-2)).*(t-xn(i-1))./(ph^2) .* (t < xn(i)) + ...
		        2.*(t-xn(i+1)).*(t-xn(i+2))./(ph^2) .* (t >= xn(i)));
		    prhs(i) = integral(g, xn(i-2), xn(i+2));
		end
		g = @(t) f(t) .* 2.*(t-xn(end-2)).*(t-xn(end-1))./(ph^2); 
		prhs(end) = integral(g, xv(end-1), xv(end));
		
		% Even indices
		for i = 2:2:M-1
		    g = @(t) f(t) .* (-4.*(t-xn(i-1)).*(t-xn(i+1)))./(ph^2);
		    prhs(i) = integral(g, xn(i-1), xn(i+1));
		end
	end
    
    % Modify RHS applying left boundary conditions
	if strcmp(BCLt,'D') 
	    prhs(1) = BCLv/ph;
	elseif strcmp(BCLt,'N')
        if todo
            prhs(1) = prhs(1) - BCLv;
        else
            prhs(1) = prhs(1) + pBCLv - BCLv;
        end
	elseif strcmp(BCLt,'P') 
	    prhs(1) = 0;
	end
    
    % Modify RHS applying right boundary conditions
    if strcmp(BCRt,'D')
        prhs(end) = BCRv/ph;
    elseif strcmp(BCRt,'N')
        if todo
            prhs(end) = prhs(end) + BCLv;
        else
            prhs(end) = prhs(end) - pBCRv + BCRv;
        end
    elseif strcmp(BCRt,'P') 
        prhs(end) = 0;
    end
    
    % Set persistent variables for boundary conditions
    pBCLt = BCLt;  pBCLv = BCLv;  pBCRt = BCRt;  pBCRv = BCRv; 
    
    rhs = prhs;
end
