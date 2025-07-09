% =========================================================================
% FILE: solve_rough_sde.m
% =========================================================================
% Solves the stochastic Duffing equation using the Euler-Maruyama scheme.
% This is a simplification; for very rough paths, a more advanced solver
% would be needed, but this is sufficient for demonstration.
function Z = solve_rough_sde(alpha, beta, sigma, Z0, t_grid, W_path)
    dt = t_grid(2) - t_grid(1);
    N = length(t_grid) - 1;
    Z = zeros(size(t_grid));
    Z(1) = Z0;
    dW = diff([0; W_path]);

    for j = 1:N
        % Euler-Maruyama step for dZ = (a*Z - b*Z^3)dt + s*Z*dW
        drift = (alpha * Z(j) - beta * Z(j)^3) * dt;
        diffusion = sigma * Z(j) * dW(j);
        Z(j+1) = Z(j) + drift + diffusion;
    end
end
