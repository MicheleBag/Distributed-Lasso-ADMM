% Lasso implementation
% TODO: fare il metodo predict (centralizzato o non?)

% import data
%dataset = readtable('dataset.csv');
% EDA -------------------------------------------------------
% removing useless features
%dataset = removevars(dataset,{'dteday','casual'});
% normalizing data between [0,1] 
%dataset{:, 2:3} = normalize(dataset{:, 2:3}, "range");

% dataset 2
dataset = readtable('dataset2.csv');
% normalizing data between [0,1] 
dataset{:, [1 3 4]} = normalize(dataset{:, [1 3 4]}, "range");


% data split(train: 80%, test: 20%) -> randomized!
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
iterations = 50000; 
step_size = 0.01;
l1_penalty = 1;
tolerance = 1e-4;

% Lasso Regression
lasso = LassoRegression(step_size, iterations, l1_penalty, tolerance);
lasso.fit(X, Y, "gd");
Y_predicted = lasso.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
% plot
figure(1)
hold on
title("Lasso GD");
scatter(Y_test,Y_predicted)
plot(Y_test,Y_test)
xlabel('Actual label')
ylabel('Predicted label')
hold off


% ADMM Lasso
lasso_admm = LassoRegression(step_size, iterations, l1_penalty, tolerance);
lasso_admm.fit(X, Y, "admm");
Y_predicted = lasso_admm.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
% plot
figure(2)
hold on
title("Lasso ADMM");
scatter(Y_test,Y_predicted)
plot(Y_test,Y_test)
xlabel('Actual label')
ylabel('Predicted label')
hold off

% Distributed Lasso
agents = 9;
lasso_dist = LassoRegression(step_size, iterations, l1_penalty, tolerance);
lasso_dist.fit(X, Y, "dist", agents);
Y_predicted = lasso_dist.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
% plot
figure(3)
hold on
title("Lasso ADMM-Distributed");
scatter(Y_test,Y_predicted)
plot(Y_test,Y_test)
xlabel('Actual label')
ylabel('Predicted label')
hold off