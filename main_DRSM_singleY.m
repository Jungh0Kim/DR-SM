%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jungho Kim, UC Berkeley
% junghokim@berkeley.edu
%
% This code is part of the following preprint:
% Kim, J., Yi, S. R., & Wang, Z. (2024). Dimensionality reduction can be used as a surrogate
% model for high-dimensional forward uncertainty quantification. arXiv preprint arXiv:2402.04582.
% https://doi.org/10.48550/arXiv.2402.04582
%
% This script presents the DR-SM method for predicting responses of linear
% elastic bar with random axial rigidity.
% Inputs = [x1, ... ,x100] - variables for KL expansion
% Outputs = [y1] - displacement at tip of the bar
%
% Note: This is a "modular" framework, allowing you to choose a suitable
% dimensionality reduction algorithm and conditional distribution model,
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
addpath('VHGPR')
addpath('utility')

%% Load training and testing datasets

% When generating training data, using latin hypercube sampling
% with sample decorrelation is highly recommended
% e.g., lhs_train = lhsdesign(n_train, n_rvs, 'criterion','correlation');
% x_train = norminv(lhs_train);

load('bar_train_test_data.mat','x_train','y_train','x_test','y_test')

% Training set (300 data)
x_train;                       % Inputs (dim=100)
y_train;                       % Outputs (dim=1)

% Testing set (1000 data)
x_test;
y_test;

[n_train, Y_size] = size(y_train);
n_test = size(y_test,1);

%% Set parameters

rng(13)

iter_n = 1200;         % Length of random sequence (N_t)
cutoff_n = 200;        % Burn-in (N_b)

% Dimensionality reduction mapping (H)
DR_method = 'PCA';

% Conditional distribution model
% Condi_dist = 'hGP';
Condi_dist = 'KDE';

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
[Psi_z, Psiz_mapping] = compute_mapping(Tot_train, DR_method, n_dim_redc);

% Conditional distribution modeling
if strcmp(Condi_dist, 'hGP')
    [~, ~, Condi_hyp] = hGP_dist(Psi_z, y_train, Psi_z);
elseif strcmp(Condi_dist, 'KDE')
    iM = 7; % Number of mixture components
    Condi_hyp = GMR(Psi_z, y_train, iM, 'full');
end

toc;

fprintf('\n Training finished. \n')

%% Extract stochastic surrogate modeling for testing dataset

% Simulate random sequence using transition kernel in Eq.(8).
% This fixed-point iteration is performed independently for each test input
% to extract a stationary distribution from the transition kernel.
% Parallel computing (e.g., parfor) can be used to accelerate this process.
mu_G_train = mean(y_train);
y_sequence = zeros(n_test, iter_n+1);
for i = 1:n_test
    x_test_i = x_test(i,:);
    y_start = mu_G_train;    % starting point

    y_sequence_i = DRSM_sequence_single(y_start, x_test_i, Psiz_mapping, Condi_hyp, iter_n, y_train, Psi_z);
    y_sequence(i,:) = y_sequence_i';

    disp(i);
end

% Remove burn-in
y_sequence_test = y_sequence(:, cutoff_n+1:end);

% unit: [m]
y_train = y_train./1e4;
y_test = y_test./1e4;
y_sequence_test = y_sequence_test./1e4;

% Final prediction mean and std.
y_pred_mean = mean(y_sequence_test, 2);
y_pred_std = std(y_sequence_test,0,2);

toc;

fprintf('\n Prediction finished. \n')

%% Prediction error

% Relative mean squared error (Eq. (13))
rMSE = mean((y_test - y_pred_mean).^2)/std(y_test).^2

%% Plot prediction

kp = 2;
shaded = [0.7 0.7 0.7];

% Plot prediction mean/+-2std based on sorted testing data
[~,y_dix] = sort(y_pred_mean(:,:));
y_test_p = y_test(y_dix,:);
y_pred_p = y_pred_mean(y_dix,:);
y_pred_std_p = y_pred_std(y_dix,:);
y_pred_CI_p = [y_pred_p + kp*y_pred_std_p; flip(y_pred_p - kp*y_pred_std_p,1)];

figure()
fill([(1:length(y_pred_p))'; flip((1:length(y_pred_p))',1)], y_pred_CI_p,...
    shaded, 'EdgeColor', shaded, 'FaceAlpha', 0.9);  hold on
plot(1:length(y_pred_p), y_pred_p,'linewidth',2,'color','k');
plot(1:length(y_test_p), y_test_p,'b.','markersize',5)
xlabel('Test data sorted in ascending order')
title("Error:" + num2str(rMSE))
lgnd = legend('Mean 2std',' Mean',' True y','Location','northwest');
set(lgnd,'FontSize',13,'NumColumns',1);
hold off
set(gcf,'unit','centimeters','position',[0 0 14 10]);
