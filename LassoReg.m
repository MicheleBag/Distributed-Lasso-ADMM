classdef LassoReg < handle
    %LASSOREGRESSION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        step_size
        max_iterations
        iterations
        l1_penalty
        tolerance
        m
        n
        W
        X
        Y
        J
    end
    
    methods
        function obj = LassoReg(step_size,max_iterations, l1_penalty, tolerance)
            %LASSOREGRESSION Construct an instance of this class
            %   Detailed explanation goes here
            obj.step_size = step_size;
            obj.max_iterations = max_iterations;
            obj.l1_penalty = l1_penalty;
            obj.tolerance = tolerance;
        end
        
        function fit(obj, X, Y, algo, agents)
            obj.m = size(X, 1); % samples
            obj.n = size(X, 2); % features
            
            obj.W = zeros(1, obj.n);
            obj.X = X;
            obj.Y = Y;
           

            if algo == "gd"
                % gradient descent
                obj.gradient_descent();
            elseif algo == "admm"
                % ADMM
                obj.admm();
            else 
                % Distributed ADMM
                obj.distributed_admm(agents);
            end
%             disp(algo);
%             disp(obj.iterations);
%             disp(obj.W);
        end
        
        function admm(obj)
            rho = obj.step_size;
            z = 0;
            u = 0;
            I = eye(obj.n,obj.n);
            
            abs_tol = obj.tolerance;
            rel_tol = abs_tol * 100; 
            
            for i = 1:obj.max_iterations
                last_z = z;
                
                obj.W = (obj.X'*obj.X + rho*I)^(-1) * (obj.X'*obj.Y + rho*(z-u));
                z = obj.soft_threshold(obj.W + u, obj.l1_penalty/rho);
                u = u + obj.W - z;
                
                r_norm  = norm(obj.W - z);          % primary residual
                s_norm  = norm(-rho*(z - last_z));  % dual residual
                tol_prim = sqrt(obj.n)*abs_tol + rel_tol*max(norm(obj.W), norm(-z));    % primary tolerance
                tol_dual= sqrt(obj.n)*abs_tol + rel_tol*norm(rho*u);                    % dual tolerance
                
                obj.iterations = i;
                obj.J(1,i) = r_norm;
                obj.J(2,i) = s_norm;
                obj.J(3,i) = tol_prim;
                obj.J(4,i) = tol_dual;


%                 obj.J(i) = obj.loss_function(obj.Y, obj.X*obj.W, z);

%                 if obj.W - z == 0
                if r_norm < tol_prim && s_norm < tol_dual   % stopping crit
                    break
                end
            end
            obj.W = obj.W';
%             disp(obj.J);
            
        end
        
        function gradient_descent(obj)
            for i = 1:obj.max_iterations
                Y_predict = obj.predict(obj.X);
                
                % gradients
%                 dW = zeros(1, obj.n);
%                 for j = 1 : obj.n
%                     dW(j) =   -2 * obj.X(:,j)' * (obj.Y - Y_predict);
%                     soft_term = obj.soft_threshold(obj.W(j), 1)
%                     dW(j) = (dW(j) + soft_term) / obj.m;
%                 end

                soft_term = obj.soft_threshold(obj.W, obj.l1_penalty);
                dW = (-2 * obj.X' * (obj.Y - Y_predict) + soft_term') / obj.m;
                new_W = obj.W - obj.step_size * dW';     % update weights
                
                if mean(abs(new_W - obj.W)) < obj.tolerance   % stopping crit
                    break
                end   
                
                obj.J(i) = mean(abs(new_W - obj.W));
                obj.W = new_W;
                obj.iterations = i;
%                 obj.J(i) = obj.loss_function(obj.Y, Y_predict, obj.W);
            end
        end
        
        function distributed_admm(obj, agents)
            rho = obj.step_size;
            z = 0;
            I = eye(obj.n,obj.n);
            
            abs_tol = obj.tolerance;
            rel_tol = abs_tol * 100; 
            converged = 0;
            
            % split by data
            [r,c] = size(obj.X);
            splitted_X   = permute(reshape(obj.X',[c,r/agents,agents]),[2,1,3]);
            splitted_Y = reshape(obj.Y,[r/agents,agents]);
            obj.W = zeros([agents c]);
            u = zeros([agents c]);
                        
            for i = 1:obj.max_iterations
                last_z = z;
                for j = 1:agents                    
                    obj.W(j,:) = (permute(splitted_X(:,:,j), [2,1,3])*splitted_X(:,:,j) + rho*I)^(-1) * (permute(splitted_X(:,:,j), [2,1,3])*splitted_Y(:,j) + rho*(z-u(j,:))');
                    z = obj.soft_threshold(mean(obj.W) + mean(u), obj.l1_penalty/rho);
                    u(j,:) = u(j,:) + (obj.W(j,:) - z);
                    
                    r_norm  = norm(mean(obj.W) - z);             % primary residual
                    s_norm  = norm(-rho*(z - last_z));          % dual residual
                    tol_prim = sqrt(obj.n)*abs_tol + rel_tol*max(norm(mean(obj.W)), norm(-z));      % primary tolerance
                    tol_dual= sqrt(obj.n)*abs_tol + rel_tol*norm(rho*mean(u));                      % dual tolerance
                    
                    % loss for each agent
%                     obj.J(j,i) = obj.loss_function(splitted_Y(:,j), splitted_X(:,:,j)*obj.W(j,:)', z);
                    obj.J(1,i,j) = r_norm;
                    obj.J(2,i,j) = s_norm;
                    obj.J(3,i,j) = tol_prim;
                    obj.J(4,i,j) = tol_dual;

%                     if obj.W(j,:) - z == 0
                    if r_norm < tol_prim && s_norm < tol_dual   % stopping crit
                        converged = 1;
                        obj.J = obj.J(:,:,j);
                        break
                    end
                end
                obj.iterations = i;

                if converged
                    break
                end
            end
%             disp(obj.W);
%             obj.J = mean(obj.J);
            obj.W = mean(obj.W);
        end
        
        % H(x)
        function Y_predict = predict(obj, X)  
            Y_predict = X * obj.W';
        end
        
        % loss function
        function loss = loss_function(obj, Y, Y_predict, W)
            loss = ( 1/2*sum((Y - Y_predict).^2) + obj.l1_penalty*norm(W,1) );
        end
      
        % SOFT-THRESHOLD
        function soft_term = soft_threshold(~, w, th)
%             if w > th
%                 soft_term = w - th;
%             elseif w < -th
%                 soft_term = w + th;
%             else
%                 soft_term = 0;
%             end
            soft_term = max(0, w-th) - max(0, -w-th);
        end
        
 
    end
end

