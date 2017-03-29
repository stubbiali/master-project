% getHeterogenousViscosityLinearPoisson1dFEP1stiffness_f Compute stiffness 
% matrix yielded by the application of linear finite elements (FE-P1) to the
% linear one-dimensional Poisson equation $-(v(x)u'(x))' = f(x)$ on $[a,b]$.
% The ODE should be completed with Dirichlet, Neumann or periodic boundary
% conditions. The "f" in the function name stands for "fast". Indeed, 
% persistent variables are used so to perform computations only when needed.
%
% A = getHeterogeneousViscosityLinearPoisson1dFEP1stiffness_f(a, b, K, v, BCLt, BCRt)
% \param a      left boundary of the domain
% \param b      right boundary of the domain
% \param K      number of elements
% \param v      space-dependent viscosity (handle function)
% \param BCLt   kind of left boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \param BCRt   kind of right boundary condition:
%               - 'D': Dirichlet
%               - 'N': Neumann
%               - 'P': periodic
% \out   A      stiffness matrix

function A = getHeterogeneousViscosityLinearPoisson1dFEP1stiffness_f ...
    (a, b, K, v, BCLt, BCRt)
    % Declare persistent variables
	persistent pa pb pK pv pBCLt pBCRt pA
	
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
        funinfo(1) = functions(pv);
        funinfo(2) = functions(v);
        todo = ~(strcmp(func2str(pv),func2str(v)) && ...
            isequal(funinfo(1).workspace{1},funinfo(2).workspace{1}));
    end
    
    % If needed, compute the stiffness matrix
    if todo
        % Set persistent variables
        pa = a;  pb = b;  pK = K;  pv = v;  pBCLt = BCLt;  pBCRt = BCRt;

        % Build computational grid with uniform spacing
        h = (b-a) / (K-1);
        x = linspace(a, b, K)';

        % Build the stiffness matrix
        pA = zeros(K,K);
        for i = 2:K-1
            % Compute (i,i-1)-th entry
            pA(i,i-1) = -1/(h*h) * integral(v, x(i-1), x(i));

            % Compute (i,i)-th entry
            pA(i,i) = 1/(h*h) * integral(v, x(i-1), x(i+1));

            % Compute (i,i+1)-th entry
            pA(i,i+1) = -1/(h*h) * integral(v, x(i), x(i+1));
        end

        % Apply left boundary conditions
        if strcmp(BCLt,'D')
            pA(1,1) = 1/h;
        elseif strcmp(BCLt,'N')
            pA(1,1) = 1/(h*h) * integral(v, x(1), x(2));
            pA(1,2) = -1/(h*h) * integral(v, x(1), x(2));
        elseif strcmp(BCLt,'P')
            pA(1,1) = 1/h;  pA(1,end) = -1/h;  
        end

        % Apply right boundary conditions
        if strcmp(BCRt,'D')
            pA(end,end) = 1/h;  
        elseif strcmp(BCRt,'N')
            pA(end,end-1) = -1/(h*h) * integral(v, x(end-1), x(end));
            pA(end,end) = 1/(h*h) * integral(v, x(end-1), x(end));
        elseif strcmp(BCRt,'P')
            pA(end,1) = 1/h;  pA(end,end) = -1/h;  
        end
    end
    
    A = pA;
end