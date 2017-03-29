% getLinearPoisson1dFEP1rhs_f Compute right-hand side for the linear system
% yielded by the application of linear finite elements (FE-P1) to the
% linear one-dimensional Poisson equation $-u''(x) = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions. The "f" in the function name stands for "fast". Indeed, 
% persistent variables are used so to perform computations only when needed.
%
% rhs = getLinearPoisson1dFEP1rhs_f(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param f      right-hand side (handle function)
% \param BCLt   kind of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCLv   value of left boundary condition
% \param BCRt   kind of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \out   rhs    right-hand side

function rhs = getLinearPoisson1dFEP1rhs_f(a, b, K, f, BCLt, BCLv, BCRt, BCRv)
	% Declare persistent variables
	persistent pa pb pK ph pf pBCLt pBCLv pBCRt pBCRv prhs
	
	% Exploiting persistent variables, compute stiffness matrix
	% only when strictly necessary, i.e. the first time this function
	% is called or when input arguments have changed with respect
	% to last call
	if isempty(pa)
		todo = 1;
	elseif (pa ~= a) | (pb ~= b) | (pK ~= K) | (pBCLt ~= BCLt) ...
		| (pBCRt ~= BCRt)
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
		x = linspace(a,b,K)';
		
		% Compute RHS
		prhs = zeros(K,1);
		g = @(t) f(t) .* (x(2)-t) ./ ph; 
		prhs(1) = integral(g, x(1), x(2));
		for i = 2:K-1
		    g = @(t) f(t) .* ((t-x(i-1)) ./ ph .* (t < x(i)) + ...
		        (x(i+1)-t) ./ ph .* (t >= x(i)));
		    prhs(i) = integral(g, x(i-1), x(i+1));
		end
		g = @(t) f(t) .* (t-x(end-1)) ./ ph; 
		prhs(end) = integral(g, x(end-1), x(end));
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
            prhs(end) = prhs(end) + BCRv;
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
