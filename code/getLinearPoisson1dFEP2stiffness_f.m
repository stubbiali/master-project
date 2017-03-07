% getLinearPoisson1dFEP2stiffness Compute stiffness matrix
% yielded by the application of quadratic finite elements (FE-P2) to the
% linear one-dimensional Poisson equation $-u''(x) = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions. The "f" in the function name stands for "fast". Indeed, 
% persistent variables are used so to perform computations only when needed.
%
% A = getLinearPoisson1dFEP2stiffness_f(a, b, K, BCLt, BCRt)
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

function A = getLinearPoisson1dFEP2stiffness_f(a, b, K, BCLt, BCRt)
	% Declare persistent variables
	persistent pa pb pK pBCLt pBCRt pA;
	
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
        pA = zeros(M,M);

        % Basis functions associated with vertices overlap with four
        % neighbouring basis functions
        pA(1,[1 2 3]) = [14/3 -8/3 1/3];
        for i = 3:2:M-2
            pA(i,[i-2 i-1 i i+1 i+2]) = [1/3 -8/3 14/3 -8/3 1/3];
        end
        pA(end,[end-2 end-1 end]) = [1/3 -8/3 14/3];

        % Basis functions associated with nodes which are not vertices overlap 
        % with two neighbouring basis functions
        for i = 2:2:M-1
            pA(i,[i-1 i i+1]) = [-8/3 16/3 -8/3];
        end

        % Modify stiffness matrix applying left boundary conditions
        if strcmp(BCLt,'D')
            pA(1,1) = 1;  pA(1,2:3) = 0;  
        elseif strcmp(BCLt,'N')
            pA(1,1) = 7/3;  pA(1,2:3) = 0;  
        elseif strcmp(BCLt,'P')
            pA(1,1) = 1;  pA(1,2:3) = 0;  pA(1,end) = -1;  
        end

        % Modify stiffness matrix applying right boundary conditions
        if strcmp(BCRt,'D')
            pA(end,end) = 1;  pA(end,end-2:end-1) = 0; 
        elseif strcmp(BCLt,'N')
            pA(end,end) = 7/3;  pA(1,2:3) = 0;  
        elseif strcmp(BCRt,'P')
            pA(end,1) = 1;  pA(end,end-2:end-1) = 0;  A(end,end) = -1;  
        end
        
        % Scale stiffness matrix
        pA = 1/h * pA;
    end
    
    A = pA;
end
