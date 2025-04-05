% Jungho Kim
% hGP model using VHGPR (2011)

function [mu_Y, s2_Y] = hGP_dist_test(X_train,Y_train,X_test,LambdaTheta_opt)

if max(max(abs(X_train))) > 1e3
    X_train = X_train./100;
    X_test = X_test./100;
end

% Covariance functions
covfuncSignal = {'covSum_vhgpr',{'covSEisoj_vhgpr','covConst_vhgpr'}};
covfuncNoise  = {'covSum_vhgpr',{'covSEisoj_vhgpr','covNoise_vhgpr'}};

LambdaTheta = LambdaTheta_opt;

% Currernt candidates for learning
[mu_Y, s2_Y] = vhgpr(LambdaTheta, covfuncSignal, covfuncNoise, 0, X_train, Y_train, X_test);

end % function end

