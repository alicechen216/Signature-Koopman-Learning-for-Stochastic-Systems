% =========================================================================
% FILE: fbm_generator.m
% =========================================================================
% Generates a fractional Brownian motion path using the Davies-Harte method.
function path = fbm_generator(H, N)
    % H: Hurst parameter
    % N: Number of steps
    T = 1;
    dt = T / N;
    
    % Covariance matrix
    gamma = @(k) 0.5 * (abs(k-1).^(2*H) - 2*abs(k).^(2*H) + abs(k+1).^(2*H));
    
    r = [gamma(0:N-1), gamma(N)];
    r_full = [r, fliplr(r(2:end-1))];
    
    % Eigenvalues of the covariance matrix
    lambda = real(fft(r_full));
    if any(lambda < 0)
        error('Negative eigenvalues in covariance matrix. FBM generation failed.');
    end
    
    % Generate the path
    W = fft(sqrt(lambda/ (2*N)) .* (randn(1, 2*N) + 1i*randn(1, 2*N)));
    
    path = cumsum(real(W(1:N)))' * (dt^H);  % Ensure column vector output
end
