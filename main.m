% Lasso implementation
% TODO: con i dati nuovi r2 con admm è buono ma il metodo gd produce un vettori di
% pesi NaN.

% import data
%dataset = readtable('dataset.csv');
% EDA -------------------------------------------------------
% removing useless features
%dataset = removevars(dataset,{'dteday','casual'});
% normalizing data between [0,1] 
%dataset{:, 2:3} = normalize(dataset{:, 2:3}, "range");

% dataset 2
dataset = readtable('dataset2.csv');



% data split(train: 80%, test: 20%)
cv = cvpartition(size(dataset,1),'HoldOut',0.2);
idx = cv.test;
% % Separate to training and test data
train = dataset(~idx,:);
test  = dataset(idx,:);
% dataset 1
% X = train{:, 2:7};
% Y = train{:, 8};
% X_test = test{:, 2:7};
% Y_test = test{:, 8};

% dataset 2
X = train{:, 1:9};
Y = train{:, 10};
X_test = test{:, 1:9};
Y_test = test{:, 10};






% Parameters
iterations = 5000; 
learning_rate = 0.01;
l1_penalty = 1;
tolerance = 0.001;

% Lasso Regression
lasso = LassoRegression(learning_rate, iterations, l1_penalty, tolerance);
lasso.fit(X, Y, "gd");
Y_predicted = lasso.predict(X_test);
% disp(Y_predicted(1:5,:));
% disp(Y_test(1:5,:));
disp(corrcoef(Y_test, Y_predicted).^2);
% 
% hold on
% scatter(Y_test,Y_predicted)
% plot(Y_test,Y_test)
% xlabel('Actual label')
% ylabel('Predicted label')
% hold off

% ADMM Lasso
lasso_admm = LassoRegression(learning_rate, iterations, l1_penalty, tolerance);
lasso_admm.fit(X, Y, "admm");
Y_predicted = lasso_admm.predict(X_test);
% disp(Y_predicted(1:5,:));
% disp(Y_test(1:5,:));
disp(corrcoef(Y_test, Y_predicted).^2);
% Distributed Lasso
