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


% Lasso Regression

% ADMM Lasso

% Distributed Lasso
