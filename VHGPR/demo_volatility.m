%
% Volatility estimation example using OU covariance function
%
clear; close all;
load Girolami200

[n, D] = size(y);
x_tr = (1:n)';
y_tr = y(1:n);

% Hyperparameter initialization
sigmaest=0.5;betaest=0.5;phiest=0.5;

% Optimize
display('Initializing VHGPR (keeping hyperparameters fixed)...')
LambdaTheta = [log(0.5)*ones(n,1);atanh(phiest);log(sigmaest);2*log(betaest)];
[LambdaTheta, convergence1] = minimize_vhgpr(LambdaTheta, 'vhgpr', 30, {'covZero'}, {'covAR1'}, 2, x_tr, y_tr);
display('Running VHGPR...')
[LambdaTheta, convergence2] = minimize_vhgpr(LambdaTheta, 'vhgpr', 100, {'covZero'}, {'covAR1'}, 0, x_tr, y_tr);
loghyper = LambdaTheta(n+1:n+3);
phiest =  tanh( LambdaTheta(n+1) )
sigmaest = exp(LambdaTheta(n+2))
mu0 = LambdaTheta(n+2+1);
betaest = exp(mu0/2)

% Predict
[Ey, Vy, mutst, diagSigmatst]= vhgpr(LambdaTheta, {'covZero'}, {'covAR1'}, 0, x_tr, y_tr, x_tr);

figure()
plotvarianza(x_tr, Ey, Vy); hold on
plot(x_tr,Ey)
plot(x_tr, y_tr, 'x');
ylabel('y(x)')

figure()
plotvarianza(x_tr, mutst-mu0, diagSigmatst)
hold on
plot(x_tr, Truex, 'x');
ylabel('g(x)')

figure()
scatter(Truex,y,'x')
