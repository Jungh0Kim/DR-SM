% Jungho Kim, junghokim@berkeley.edu
%
% 3D space truss structure with uncertain parameters.
% x1~x7: applied forces, x8~x32: Young's modulus, x33~57: sectioanl areas
%
% See Section 4.2 of following reference for details:
% Kim, J., Wang, Z., Song, J. (2024). Adaptive active subspace-based metamodeling
% for high-dimensional reliability analysis. Structural Safety, 106, 102404.

function [Y_response] = Truss3d_model(u_rand, Data_str)

n_sim = size(u_rand,1);
n_dim = size(u_rand,2);

%% Random variable parameters

P1.mu = 1000;         P1.cov = 0.1;
P2_5.mu = 10000;      P2_5.cov = 0.05;
P6.mu = 600;          P6.cov = 0.1;
P7.mu = 500;          P7.cov = 0.1;
E.mu = 1e7;           E.cov = 0.05;
A1.mu = 0.4;          A1.cov = 0.1;
A2_5.mu = 0.1;        A2_5.cov = 0.1;
A6_9.mu = 3.4;        A6_9.cov = 0.1;
A10_11.mu = 0.4;      A10_11.cov = 0.1;
A12_13.mu = 1.3;      A12_13.cov = 0.1;
A14_17.mu = 0.9;      A14_17.cov = 0.1;
A18_21.mu = 1;        A18_21.cov = 0.1;
A22_25.mu = 3.4;      A22_25.cov = 0.1;

[P1.lambda, P1.zeta] = Logn_para(P1.mu, P1.mu*P1.cov);
[P2_5.lambda, P2_5.zeta] = Logn_para(P2_5.mu, P2_5.mu*P2_5.cov);
[P6.lambda, P6.zeta] = Logn_para(P6.mu, P6.mu*P6.cov);
[P7.lambda, P7.zeta] = Logn_para(P7.mu, P7.mu*P6.cov);
[E.lambda, E.zeta] = Logn_para(E.mu, E.mu*E.cov);

%% U to X (transformation)

x_rand = zeros(n_sim, n_dim);
x_rand(:, 1) = logninv(normcdf(u_rand(:, 1)), P1.lambda,P1.zeta);
x_rand(:, 2:5) = logninv(normcdf(u_rand(:, 2:5)), P2_5.lambda,P2_5.zeta);
x_rand(:, 6) = logninv(normcdf(u_rand(:, 6)), P6.lambda,P6.zeta);
x_rand(:, 7) = logninv(normcdf(u_rand(:, 7)), P7.lambda,P7.zeta);
x_rand(:, 8:32) = logninv(normcdf(u_rand(:, 8:32)), E.lambda,E.zeta);
x_rand(:, 33) = u_rand(:, 33).*A1.mu.*A1.cov + A1.mu;
x_rand(:, 34:37) = u_rand(:, 34:37).*A2_5.mu.*A2_5.cov + A2_5.mu;
x_rand(:, 38:41) = u_rand(:, 38:41).*A6_9.mu.*A6_9.cov + A6_9.mu;
x_rand(:, 42:43) = u_rand(:, 42:43).*A10_11.mu.*A10_11.cov + A10_11.mu;
x_rand(:, 44:45) = u_rand(:, 44:45).*A12_13.mu.*A12_13.cov + A12_13.mu;
x_rand(:, 46:49) = u_rand(:, 46:49).*A14_17.mu.*A14_17.cov + A14_17.mu;
x_rand(:, 50:53) = u_rand(:, 50:53).*A18_21.mu.*A18_21.cov + A18_21.mu;
x_rand(:, 54:57) = u_rand(:, 54:57).*A22_25.mu.*A22_25.cov + A22_25.mu;

%% Truss analysis

Disp_sv = zeros(n_sim, 3);
for i_sim = 1:n_sim
    Data_str.Load(1,1) = x_rand(i_sim,1);
    Data_str.Load(2,1) = x_rand(i_sim,2);
    Data_str.Load(3,1) = x_rand(i_sim,3);
    Data_str.Load(2,2) = x_rand(i_sim,4);
    Data_str.Load(3,2) = x_rand(i_sim,5);
    Data_str.Load(1,3) = x_rand(i_sim,6);
    Data_str.Load(1,6) = x_rand(i_sim,7);
    Data_str.E(:,1) = x_rand(i_sim, 8:32)';
    Data_str.A(:,1) = x_rand(i_sim,33:57)';

    [~, Disp] = TS_analysis(Data_str);
    Disp_sv(i_sim,1) = Disp(1, 1);
    Disp_sv(i_sim,2) = Disp(2, 1);
    Disp_sv(i_sim,3) = Disp(3, 1);
end

Y_response = Disp_sv.*100;

end % function end


