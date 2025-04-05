% Jungho Kim
% hGP model using VHGPR (2011)

function [mu_Y, s2_Y, LambdaTheta] = hGP_dist(X_train,Y_train,X_test)

if max(max(abs(X_train))) > 1e3
    X_train = X_train./100;
    X_test = X_test./100;
end

mean_p = mean(Y_train);
[n_x, D_x] = size(X_train);

% Covariance functions
covfuncSignal = {'covSum_vhgpr',{'covSEisoj_vhgpr','covConst_vhgpr'}};
covfuncNoise  = {'covSum_vhgpr',{'covSEisoj_vhgpr','covNoise_vhgpr'}};

% Hyperparameter initialization
SignalPower = var(Y_train,1);
NoisePower = SignalPower/4;
lengthscales = log((max(X_train)-min(X_train))'/2);

% Running standard, homoscedastic GP
loghyperGP = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower);-0.5*log(max(SignalPower/20,mean_p^2))];
loghyperGP = minimize_vhgpr(loghyperGP, 'gpr_vhgpr', 40, {'covSum_vhgpr', {'covSEardj_vhgpr','covNoise_vhgpr','covConst_vhgpr'}}, X_train, Y_train);

lengthscales = loghyperGP(1:D_x);

SignalPower = exp(2*loghyperGP(D_x+1));
NoisePower = exp(2*loghyperGP(D_x+2));
ConstPower = exp(-2*loghyperGP(D_x+3));

sn2 = 1;
mu0 = log(NoisePower) - sn2/2 - 2;
loghyperSignal = [0; 0.5*log(SignalPower); -0.5*log(ConstPower)];
loghyperNoise =  [0; 0.5*log(sn2); 0.5*log(sn2*0.25)];

% Initializing VHGPR (keeping hyperparameters fixed)
LambdaTheta = [log(0.5)*ones(n_x,1);loghyperSignal;loghyperNoise;mu0];
[LambdaTheta, ~] = minimize_vhgpr(LambdaTheta, 'vhgpr', 30, covfuncSignal, covfuncNoise, 2, X_train, Y_train);
% Running VHGPR
[LambdaTheta, ~] = minimize_vhgpr(LambdaTheta, 'vhgpr', 100, covfuncSignal, covfuncNoise, 0, X_train, Y_train);

% Currernt candidates for learning
[mu_Y, s2_Y] = vhgpr(LambdaTheta, covfuncSignal, covfuncNoise, 0, X_train, Y_train, X_test);

end % function end

