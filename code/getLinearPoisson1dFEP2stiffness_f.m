% getLinearPoisson1dFEP2stiffness Compute stiffness matrix
% yielded by the application of quadratic finite elements (FE-P2) to the
% linear one-dimensional Poisson equation $-u''(x) = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions. The "f" in the function name stands for "fast". Indeed, 
% persistent variables are used so to perform computations only when needed.
%
% A = getLinearPoisson1dFEP2stiffness_f(a, b, K, f, BCLt, BCRt)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param f      RHS; this may be either an handle function or a cell array
%               of handle functions; in the latter case, the solution is
%               computed for each RHS
% \param BCLt   kind of left boundary condition; 'D' = Dirichlet, 'N' =
%               Neumann, 'P' = periodic
% \param BCRt   kind of right boundary condition; 'D' = Dirichlet, 'N' =
%               Neumann, 'P' = periodic
% \out   A      stiffness matrix

function A = getLinearPoisson1dFEP2stiffness_f(a, b, K, BCLt, BCRt)
	% Declare persistent variables
	persistent pa, pb, pK, pBCLt, BCRt, A;
	
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
		
		% Allocate memory for stiffness matrix
		M = 2*K-1;
		A = zeros(M,M);
		
		% Basis functions associated with vertices overlap with four
		% neighbouring basis functions
		A(1,[1 2 3]) = [14/3 -8/3 1/3];
		for i = 3:2:M-2
		    A(i,[i-2 i-1 i i+1 i+2]) = [1/3 -8/3 14/3 -8/3 1/3];
		end
		A(end,[end-2 end-1 end]) = [1/3 -8/3 14/3];
		
		% Basis functions associated with nodes which are not vertices overlap 
		% with two neighbouring basis functions
		for i = 2:2:M-1
		    A(i,[i-1 i i+1]) = [-8/3 16/3 -8/3];
		end
		
		% Scale stiffness matrix
		A = 1/h * A;
		
		% Modify stiffness matrix applying left boundary conditions
		if strcmp(BCLt,'D')
		    A(1,1) = 1/h;  A(1,2:3) = 0;  
		elseif strcmp(BCLt,'P')
		    A(1,1) = 1/h;  A(1,2:3) = 0;  A(1,end) = -1/h;  
		end
		
		% Modify stiffness matrix applying right boundary conditions
		if strcmp(BCRt,'D')
		    A(end,end) = 1/h;  A(end,end-2:end-1) = 0;  
		elseif strcmp(BCRt,'P')
		    A(end,1) = 1/h;  A(end,end-2:end-1) = 0;  A(end,end) = -1/h;  
		end
	end
end
