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
        
        function fit(obj, X, Y)
            obj.m = size(X, 1); % samples
            obj.n = size(X, 2); % features
            
            obj.W = zeros(1, obj.n);
            obj.X = X;
            obj.Y = Y;
           
            % gradient descent
            obj.gradient_descent();
            disp(obj.iterations);
        end
        
        
        function gradient_descent(obj)
            for i = 1:obj.max_iterations
                % show #iterations
                if mod(i,10000) == 0 
                    disp(i);
                end
                
                Y_predict = obj.predict(obj.X);
                % gradients
                
                dW = zeros(1, obj.n);
                for j = 1 : obj.n
                    dW(j) =   -2 * obj.X(:,j)' * (obj.Y - Y_predict);
                    soft_term = obj.soft_threshold(obj.W(j));
                    dW(j) = (dW(j) + soft_term) / obj.m;
                end               
%               dW = (-2 * obj.X' * (obj.Y - Y_predict) + soft_term) /obj.m; 

                %update weights
                new_W = obj.W - obj.learning_rate * dW;
                if abs(new_W - obj.W) < obj.tolerance
                    break
                end   
                obj.W = new_W;
                obj.iterations = i;
            end
        end
        
        
        % H(x)
        function Y_predict = predict(obj, X)  
            Y_predict = X * obj.W';
        end
        
        
        % SOFT-THRESHOLD
        function soft_term = soft_threshold(obj, w)
            if w > obj.l1_penalty 
                soft_term = w - obj.l1_penalty;
            elseif w < -obj.l1_penalty
                soft_term = w + obj.l1_penalty;
            else
                soft_term = 0;
            end       
        end
        
 
    end
end

