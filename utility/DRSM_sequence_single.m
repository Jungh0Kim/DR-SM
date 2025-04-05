% Jungho Kim, junghokim@berkeley.edu
% Simulate random sequence using the transition kernel (prediction stage in DR-SM).
% This implements a fixed-point iteration to obtain the stationary distribution.

function [y_sequence_sv] = DRSM_sequence_single(y_start, x_test, Psiz_mapping, Condi_hyp, iter_n, y_train, Psi_z)

% x_test;    % testing inputs

y_sequence_sv = zeros(iter_n+1, 1);
y_sequence_sv(1,1) = y_start;
for k = 1:iter_n
    if k==1  % starting point
        y_iter = y_start;
    elseif k>1  % from prior step
        y_iter = y_sequence;
    end

    % Embedding
    Psiz_DR1_k = out_of_sample([x_test y_iter], Psiz_mapping);

    % Conditional model: hGP or KDE
    if isstruct(Condi_hyp)==0 % hGP
        [mu_y_Psiz_k, s2_y_Psiz_k] = hGP_dist_test(Psi_z, y_train, Psiz_DR1_k, Condi_hyp);
    elseif isstruct(Condi_hyp)==1 % KDR
        [mu_y_Psiz_k, s2_y_Psiz_k] = GMRTest(Condi_hyp, Psiz_DR1_k);
    end

    y_sequence = normrnd(mu_y_Psiz_k, sqrt(s2_y_Psiz_k), [1, 1]);    % draw sample from iteration
    y_sequence_sv(k+1,1) = y_sequence;
end

end % function end
