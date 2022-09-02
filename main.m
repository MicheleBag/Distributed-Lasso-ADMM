% Lasso implementation
% TODO: distributed admm / admm loss -> perchè aumenta?
% potrei far vedere la convergenza con il plot dei residui


% import data
dataset = readtable('dataset2.csv');
% normalizing data between [0,1] 
dataset{:, [1 3 4]} = normalize(dataset{:, [1 3 4]}, "range");


% data split(train: 80%, test: 20%) -> randomized!
cv = cvpartition(size(dataset,1),'HoldOut',0.2);
idx = cv.test;
% % Separate to training and test data
train = dataset(~idx,:);
test  = dataset(idx,:);

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
% predict plot
% plot_predict("Lasso GD", Y_test, Y_predicted);
plot_loss(lasso, "Loss GD");


% % ADMM Lasso
lasso_admm = LassoRegression(step_size, iterations, l1_penalty, tolerance);
lasso_admm.fit(X, Y, "admm");
Y_predicted = lasso_admm.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
% plot
% plot_predict("Lasso ADMM", Y_test, Y_predicted);
plot_loss(lasso_admm, "Loss ADMM");


% % Distributed Lasso
agents = 9;
lasso_dist = LassoRegression(step_size, iterations, l1_penalty, tolerance);
lasso_dist.fit(X, Y, "dist", agents);
Y_predicted = lasso_dist.predict(X_test);
disp(corrcoef(Y_test, Y_predicted).^2);     % R2
% plot
% plot_predict("Lasso Distributed-ADMM", Y_test, Y_predicted);
plot_loss(lasso_dist,  "Loss Distributed-ADMM");

function plot_predict(label, Y_test, Y_predicted)
    figure
    hold on
    title(label);
    scatter(Y_test,Y_predicted)
    plot(Y_test,Y_test)
    xlabel('Actual label')
    ylabel('Predicted label')
    hold off
end

function plot_loss(lasso, label)
    figure
    hold on
    title(label);
    plot(lasso.J)
    xlabel('Iterations')
    ylabel('Loss')
    hold off
end
