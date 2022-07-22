% Lasso implementation

% ##weathersit : 
%		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
%		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
%		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
%		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

% import data
dataset = readtable('dataset.csv');

% EDA -------------------------------------------------------
% removing useless features
dataset = removevars(dataset,{'dteday','casual'});
% normalizing data between [0,1] 
dataset{:, 2:3} = normalize(dataset{:, 2:3}, "range");

% data split(train: 70%, test: 30%)
cv = cvpartition(size(dataset,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
train = dataset(~idx,:);
test  = dataset(idx,:);
X = train{:, 2:7};
Y = train{:, 8};
X_test = test{:, 2:7};
Y_test = test{:, 8};


% Parameters
iterations = 100000; 
learning_rate = 0.01;
l1_penalty = 1;
tolerance = 0.001;

% Lasso Regression
lasso = LassoRegression(learning_rate, iterations, l1_penalty, tolerance);
lasso.fit(X, Y);
Y_predicted = lasso.predict(X_test);
disp(Y_predicted(1:5,:));
disp(Y_test(1:5,:));
disp(corrcoef(Y_test, Y_predicted).^2);

hold on
scatter(Y_test,Y_predicted)
plot(Y_test,Y_test)
xlabel('Actual label')
ylabel('Predicted label')
hold off

% ADMM Lasso
% Distributed Lasso
