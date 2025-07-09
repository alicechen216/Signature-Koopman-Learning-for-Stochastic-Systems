% =========================================================================
% FILE: main_koopman_signature_experiment.m
% =========================================================================
% This script validates the use of the signature transform as a basis for
% Koopman analysis of a nonlinear system driven by a rough stochastic path.
%
% Experiment Steps:
% 1. Define a stochastic Duffing oscillator driven by fractional Brownian motion.
% 2. Generate training and testing datasets of (driving path, final state).
% 3. Train a linear Koopman model on the signature features of the driving paths.
% 4. Validate the model's predictive accuracy on the test set.
% =========================================================================

clear; clc; close all;

%% ===== 1. Experiment Setup =====
disp('1. Setting up the experiment...');

% System Parameters (Stochastic Duffing Oscillator)
alpha = 1.0;   % Instability term
beta = 0.5;    % Nonlinear damping
sigma = 0.3;   % Noise intensity
Z0 = 1.0;      % Initial condition

% Path and Time Parameters
T = 1.0;       % Final time
N_steps = 100; % Number of time steps
t_grid = linspace(0, T, N_steps + 1)';
H = 0.3;       % Hurst parameter for fBm (H < 0.5 is "rough")

% Model and Data Parameters
M = 4;              % Signature truncation order
d = 1;              % Dimension of the driving noise (univariate fBm)
N_train = 2000;     % Number of training trajectories
N_test = 500;       % Number of testing trajectories

% Generate multi-indices for the signature
indices = generate_indices(d, M);
L = length(indices);
fprintf('Signature order M=%d, Hurst H=%.2f, Num features L=%d\n', M, H, L);

%% ===== 2. Data Generation =====
disp('2. Generating training and testing data...');

% --- Training Data ---
fprintf('   Generating %d training samples...\n', N_train);
S_train = zeros(N_train, L); % Matrix of signature features
Z_final_train = zeros(N_train, 1);   % Vector of final states

for i = 1:N_train
    % Generate a fractional Brownian motion path
    fBm_path = fbm_generator(H, N_steps);
    
    % Solve the SDE to get the state trajectory
    Z_path = solve_rough_sde(alpha, beta, sigma, Z0, t_grid, fBm_path);
    
    % Store the final state
    Z_final_train(i) = Z_path(end);
    
    % Construct the augmented driving path X_t = [t, W_t^H]
    X_path = [t_grid, [0; fBm_path]];
    
    % Calculate the signature of the driving path up to the final time T
    S_train(i, :) = calculate_final_signature(X_path, indices);
end

% --- Testing Data ---
fprintf('   Generating %d testing samples...\n', N_test);
S_test = zeros(N_test, L);
Z_final_test = zeros(N_test, 1);

for i = 1:N_test
    fBm_path = fbm_generator(H, N_steps);
    Z_path = solve_rough_sde(alpha, beta, sigma, Z0, t_grid, fBm_path);
    Z_final_test(i) = Z_path(end);
    X_path = [t_grid, [0; fBm_path]];
    S_test(i, :) = calculate_final_signature(X_path, indices);
end

disp('Data generation complete.');

%% ===== 3. Train the Koopman Model =====
disp('3. Training the linear Koopman model on signature features...');

% We want to find a linear model K such that Z_final ≈ S_train * K
% This is a standard linear regression problem. We add a bias term (intercept).
X_train_reg = [ones(N_train, 1), S_train]; % Add column of ones for intercept

% Solve for the Koopman model coefficients using least squares
K_model = X_train_reg \ Z_final_train;

fprintf('Training complete. Koopman model has %d coefficients.\n', length(K_model));

%% ===== 4. Validation and Analysis =====
disp('4. Validating the model on the test set...');

% Use the trained model to predict the final states for the test set
X_test_reg = [ones(N_test, 1), S_test];
Z_final_pred = X_test_reg * K_model;

% --- Performance Metrics ---
% R-squared value
ss_total = sum((Z_final_test - mean(Z_final_test)).^2);
ss_resid = sum((Z_final_test - Z_final_pred).^2);
r_squared = 1 - ss_resid / ss_total;

% Mean Squared Error (MSE)
mse = mean((Z_final_test - Z_final_pred).^2);

fprintf('Validation complete.\n');
fprintf('   R-squared: %.4f\n', r_squared);
fprintf('   Mean Squared Error (MSE): %.4f\n', mse);

% --- Visualization ---
% Scatter plot of Predicted vs. True values
figure('Name', 'Koopman Model Prediction Accuracy', 'NumberTitle', 'off');
scatter(Z_final_test, Z_final_pred, 30, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
% Plot the ideal y=x line
lims = [min([Z_final_test; Z_final_pred]), max([Z_final_test; Z_final_pred])];
plot(lims, lims, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Perfect Prediction');
grid on;
box on;
xlabel('True Final State Z(T)', 'FontSize', 12);
ylabel('Predicted Final State Z(T)', 'FontSize', 12);
title(sprintf('Koopman Prediction with Signature Features (R^2 = %.3f)', r_squared), 'FontSize', 14);
legend('Model Predictions', 'Perfect Prediction', 'Location', 'best');
axis equal;
xlim(lims);
ylim(lims);

% Plot a few sample paths
%figure('Name', 'Sample Trajectories', 'NumberTitle', 'off');
%subplot(2, 1, 1);
%plot(t_grid, [0; fbm_generator(H, N_steps)], 'r-');
%title('Sample Driving Path (fBm, H=0.3)');
%xlabel('Time t'); ylabel('W_t^H'); grid on;
%subplot(2, 1, 2);
%Z_sample = solve_rough_sde(alpha, beta, sigma, Z0, t_grid, fbm_generator(H, N_steps));
%plot(t_grid, Z_sample, 'b-');
%title('Resulting System Trajectory Z(t)');
%xlabel('Time t'); ylabel('Z(t)'); grid on;

figure('Name', 'Sample Trajectories', 'NumberTitle', 'off');
fBm_sample = fbm_generator(H, N_steps);

subplot(2, 1, 1);
plot(t_grid, [0; fBm_sample], 'r-');
title('Sample Driving Path (fBm, H=0.3)');
xlabel('Time t'); ylabel('W_t^H'); grid on;

subplot(2, 1, 2);
Z_sample = solve_rough_sde(alpha, beta, sigma, Z0, t_grid, fBm_sample);
plot(t_grid, Z_sample, 'b-');
title('Resulting System Trajectory Z(t)');
xlabel('Time t'); ylabel('Z(t)'); grid on;
