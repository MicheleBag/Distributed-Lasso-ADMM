classdef LassoRegression < handle
    %LASSOREGRESSION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        learning_rate
        max_iterations
        iterations
        l1_penalty
        tolerance
        m
        n
        W
        X
        Y
    end
    
    methods
        function obj = LassoRegression(learning_rate,max_iterations, l1_penalty, tolerance)
            %LASSOREGRESSION Construct an instance of this class
            %   Detailed explanation goes here
            obj.learning_rate = learning_rate;
            obj.max_iterations = max_iterations;
            obj.l1_penalty = l1_penalty;
            obj.tolerance = tolerance;
        end
        
        function fit(obj, X, Y, algo)
            obj.m = size(X, 1); % samples
            obj.n = size(X, 2); % features
            
            obj.W = zeros(1, obj.n);
            obj.X = X;
            obj.Y = Y;
           
            disp("Working...");
            if algo == "gd"
                % gradient descent
                obj.gradient_descent();
            elseif algo == "admm"
                % ADMM
                obj.admm();
            else 
                % Distributed ADMM
            end
            disp(algo);
            disp(obj.iterations);
            disp(obj.W);
        end
        
        function admm(obj)
            rho = 1;
            z = 0;
            u = 0;
            I = eye(obj.n,obj.n);
            
            abs_tol = 1e-4;
            rel_tol = 1e-2;
            
            for i = 1:obj.max_iterations
                last_z = z;
                obj.W = (obj.X'*obj.X + rho*I)^(-1) * (obj.X'*obj.Y + rho*(z-u));
                z = obj.soft_threshold(obj.W + u, rho);
                u = u + obj.W - z;
                
                
                r_norm  = norm(obj.W - z);          % primary residual
                s_norm  = norm(-rho*(z - last_z));  % dual residual
                tol_prim = sqrt(obj.n)*abs_tol + rel_tol*max(norm(obj.W), norm(-z));    % primary tolerance
                tol_dual= sqrt(obj.n)*abs_tol + rel_tol*norm(rho*u);                    % dual tolerance
                
                obj.iterations = i;
                % stopping condition
                if r_norm < tol_prim && s_norm < tol_dual
                    break
                end
            end
            obj.W = obj.W';
        end
        
        function gradient_descent(obj)
            for i = 1:obj.max_iterations
                % show #iterations
%                 if mod(i,10000) == 0 
%                     disp(i);
%                 end
                
                Y_predict = obj.predict(obj.X);
                
                % gradients
%                 dW = zeros(1, obj.n);
%                 for j = 1 : obj.n
%                     dW(j) =   -2 * obj.X(:,j)' * (obj.Y - Y_predict);
%                     soft_term = obj.soft_threshold(obj.W(j), 1)
%                     dW(j) = (dW(j) + soft_term) / obj.m;
%                 end
                soft_term = obj.soft_threshold(obj.W, 1);
                dW = (-2 * obj.X' * (obj.Y - Y_predict) + soft_term') / obj.m;
                
                %update weights
                new_W = obj.W - obj.learning_rate * dW';
                
                if abs(new_W - obj.W) < obj.tolerance
                    break
                end   
                obj.W = new_W
                
                if sum( isnan(obj.W)) % X debug
                    break
                end
                obj.iterations = i;
            end
        end
        
        % H(x)
        function Y_predict = predict(obj, X)  
            Y_predict = X * obj.W';
        end
        
        % SOFT-THRESHOLD
        function soft_term = soft_threshold(obj, w, rho)
            th = obj.l1_penalty / rho; 
%             if w > th
%                 soft_term = w - th;
%             elseif w < -th
%                 soft_term = w + th;
%             else
%                 soft_term = 0;
%             end
            aux = max(abs(w)- th,0);
            soft_term = aux./(aux+ th).*w;
        end
        
 
    end
end

