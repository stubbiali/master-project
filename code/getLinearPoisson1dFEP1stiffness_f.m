% getLinearPoisson1dFEP1stiffness_f Compute stiffness matrix
% yielded by the application of linear finite elements (FE-P1) to the
% linear one-dimensional Poisson equation $-u''(x) = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions. The "f" in the function name stands for "fast". Indeed, 
% persistent variables are used so to perform computations only when needed.
% 
% A = getLinearPoisson1dFEP1stiffness_f(a, b, K, BCLt, BCRt)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param BCLt   kind of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRt   kind of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \out   A      stiffness matrix

function A = getLinearPoisson1dFEP1stiffness_f(a, b, K, BCLt, BCRt)
	% Declare persistent variables
	persistent pa pb pK pBCLt pBCRt pA
	
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
		todo = 0;
	end
	
	% If needed, compute the stiffness matrix
	if todo
		% Set persistent variables
		pa = a;  pb = b;  pK = K;  pBCLt = BCLt;  pBCRt = BCRt;
		
		% Get uniform grid spacing
		h = (b-a) / (K-1);
		
		% Build the stiffness matrix
		pA = 1/h * (diag(-ones(K-1,1),-1) + diag(2*ones(K,1)) + ...
		    diag(-ones(K-1,1),1));
		
		% Modify stiffness matrix applying left boundary conditions
		if strcmp(BCLt,'D')
		    pA(1,1) = 1/h;  pA(1,2) = 0;  
        elseif strcmp(BCLt,'N')
            pA(1,1) = 1/h;
		elseif strcmp(BCLt,'P')
		    pA(1,1) = 1/h;  pA(1,2) = 0;  pA(1,end) = -1/h;  
		end
		
		% Modify stiffness matrix applying right boundary conditions
		if strcmp(BCRt,'D')
		    pA(end,end) = 1/h;  pA(end,end-1) = 0;  
        elseif strcmp(BCRt,'N')
            pA(end,end) = 1/h;
		elseif strcmp(BCRt,'P')
		    pA(end,1) = 1/h;  pA(end,end-1) = 0;  pA(end,end) = -1/h;  
		end
    end
    
    A = pA;
end
