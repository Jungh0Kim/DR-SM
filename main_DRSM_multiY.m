%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jungho Kim, UC Berkeley
% junghokim@berkeley.edu
%
% This code is part of the following preprint:
% Kim, J., Yi, S. R., & Wang, Z. (2024). Dimensionality reduction can be used as a surrogate
% model for high-dimensional forward uncertainty quantification. arXiv preprint arXiv:2402.04582.
% https://doi.org/10.48550/arXiv.2402.04582
%
% This script presents the DR-SM method for predicting responses of a 3D
% truss structure.
% Inputs = [x1, ... ,x57] - structural uncertain parameters
% Outputs = [y1, y2, y3] - displacements in the x, y, and z directions
%
% Note: This is a "modular" framework, allowing you to choose a suitable
% dimensionality reduction model and conditional distribution model,
% with appropriately tuned parameters. The accuracy of the surrogate method
% depends on the effectiveness of the chosen DR & Condi models.
% This version uses simple choices, e.g., PCA and KDE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; tic;
set(0,'DefaultFigureColor',[1 1 1])
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(0,'defaulttextinterpreter','latex','DefaultAxesFontSize',14);

%% Add paths

addpath('drtoolbox\corefunc')
addpath('drtoolbox\gui')
addpath('drtoolbox\techniques')
addpath('netlab_GMR')
addpath('utility')

%% Set parameters

rng(10)

n_dim = 57;            % Input dimension

n_train = 200;         % # of training data
n_test = 300;          % # of testing data

iter_n = 1200;         % Length of random sequence (N_t)
cutoff_n = 200;        % Burn-in (N_b)

% Dimensionality reduction mapping (H)
DR_method = 'PCA';
% DR_method = 'KPCA';
% DR_para = [1.0 1.0];

% Conditional distribution model
Condi_dist = 'KDE';

Y_func_name = 'Truss3d_model';     % computational model
Y_func = str2func(Y_func_name);

Data_str = TS_Geometry;

%% Training inputs and outputs

% Training inputs
lhs_doe = lhsdesign(n_train, n_dim, 'criterion','correlation');
x_train = norminv(lhs_doe);

% Training outputs by structural analysis
y_train = Y_func(x_train, Data_str);
Y_size = size(y_train,2);

%% Construct dimension reduction mapping and conditional distribution model

% Augmented vector - input X and output Y
Tot_train = [x_train, y_train];

n_dim_redc = 5;     % This should be selected based on Algorithm 1
% From our experience, the selection of n_dim_redc is not critical
% since the core of the proposed method is to "extract" a stochastic
% surrogate model from the low-dimensional representation of the
% input-output space.
% You may choose a suitable "dimensionality reduction algorithm" and
% "conditional distribution model", and tune their parameters and the
% reduced dimensionality accordingly.

% Input-output mapping
if strcmp(DR_method, 'PCA')
    [Psi_z, Psiz_mapping] = compute_mapping(Tot_train, DR_method, n_dim_redc);
elseif strcmp(DR_method, 'KPCA')
    [Psi_z, Psiz_mapping] = compute_mapping(Tot_train, DR_method, n_dim_redc, 'poly', DR_para(1), DR_para(2));
end

% Conditional distribution modeling
iM = 7; % Number of mixture components
Condi_hyp = GMR(Psi_z, y_train, iM, 'full');

toc;

fprintf('\n Training finished. \n')

%% Extract stochastic surrogate modeling for testing dataset

% Testing inputs and outputs
x_test = normrnd(0, 1, [n_test, n_dim]);
y_test = Y_func(x_test, Data_str);  % Testing outputs to be predicted

% Simulate random sequence using transition kernel in Eq.(8).
% This fixed-point iteration is performed independently for each test input
% to extract a stationary distribution from the transition kernel.
% Parallel computing (e.g., parfor) can be used to accelerate this process.
mu_G_train = mean(y_train);
y_sequence = zeros(iter_n+1, Y_size, n_test);
for i = 1:n_test
    x_test_i = x_test(i,:);
    y_start = mu_G_train;    % starting point

    y_sequence_i = DRSM_sequence_multi(y_start, x_test_i, Psiz_mapping, Condi_hyp, iter_n);
    y_sequence(:,:,i) = y_sequence_i;

    disp(i);
end

% Remove burn-in
y_sequence_test = y_sequence(cutoff_n+1:end, :, :);

% unit
y_train = y_train./1e2;
y_test = y_test./1e2;
y_sequence_test = y_sequence_test./1e2;

% Final prediction mean and std.
y_pred_mean = mean(y_sequence_test, 1);
y_pred_mean = reshape(y_pred_mean, Y_size, n_test);
y_pred_mean = y_pred_mean';
y_pred_std = std(y_sequence_test);
y_pred_std = reshape(y_pred_std, Y_size, n_test);
y_pred_std = y_pred_std';

toc;

fprintf('\n Prediction finished. \n')

%% Prediction error

% Relative mean squared error (Eq. (13))
for kk=1:Y_size
    rMSE(:,kk) = mean((y_test(:,kk) - y_pred_mean(:,kk)).^2)/std(y_test(:,kk)).^2;
end
rMSE

%% Plot prediction

kp = 2;
shaded = [0.7 0.7 0.7];

% Plot prediction mean/+-2std based on sorted testing data
figure()
for kk=1:Y_size
    [~,y_dix] = sort(y_pred_mean(:,kk));
    y_test_p = y_test(y_dix,kk);
    y_pred_p = y_pred_mean(y_dix,kk);
    y_pred_std_p = y_pred_std(y_dix,kk);
    y_pred_CI_p = [y_pred_p + kp*y_pred_std_p; flip(y_pred_p - kp*y_pred_std_p,1)];

    subplot(1, 3, kk)
    fill([(1:length(y_pred_p))'; flip((1:length(y_pred_p))',1)], y_pred_CI_p,...
        shaded, 'EdgeColor', shaded, 'FaceAlpha', 0.9);  hold on
    plot(1:length(y_pred_p), y_pred_p,'linewidth',2,'color','k');
    plot(1:length(y_test_p), y_test_p,'b.','markersize',5)
    xlabel('Test data sorted in ascending order')
    ylabel(strcat('$Y_{',num2str(kk),'}$'))
    title("Error:" + num2str(rMSE(kk)))
    if kk==1
        lgnd = legend('Mean 2std',' Mean',' True y','Location','northwest');
        set(lgnd,'FontSize',13,'NumColumns',1);
    end
    hold off
end
set(gcf,'unit','centimeters','position',[0 0 35 10]);
