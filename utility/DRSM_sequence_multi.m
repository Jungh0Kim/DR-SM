% Jungho Kim, junghokim@berkeley.edu
% Simulate random sequence using the transition kernel (prediction stage in DR-SM).
% This implements a fixed-point iteration to obtain the stationary distribution.

function [y_sequence_sv] = DRSM_sequence_multi(y_start, x_test, Psiz_mapping, Condi_hyp, iter_n)

% x_test;    % testing inputs

Y_dim = size(y_start, 2);

y_sequence_sv = zeros(iter_n+1, Y_dim);
y_sequence_sv(1,:) = y_start;
for k = 1:iter_n
    if k==1  % starting point
        y_iter = y_start;
    elseif k>1  % from prior step
        y_iter = y_sequence;
    end

    % Embedding
    Psiz_DR1_k = out_of_sample([x_test y_iter], Psiz_mapping);

    % Condi. model by KDE
    [mu_y_Psiz_k, COV_y_Psiz_k] = GMRTest(Condi_hyp, Psiz_DR1_k);
    COV_mat = (COV_y_Psiz_k{1,1} + COV_y_Psiz_k{1,1}')/2;    % for symmetry
    y_sequence = mvnrnd(mu_y_Psiz_k, COV_mat, 1);    % draw sample from iteration

    % save
    y_sequence_sv(k+1,:) = y_sequence;
end

end % function end
