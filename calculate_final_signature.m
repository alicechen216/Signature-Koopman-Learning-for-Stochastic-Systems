% =========================================================================
% FILE: calculate_final_signature.m
% =========================================================================
% Calculates the signature of a path ONLY at its final time T.
% This is a simplified version of the `build_signature_matrix` from the
% previous example, optimized for this specific task.
function S_final = calculate_final_signature(X, indices)
    N = size(X, 1) - 1;
    L = length(indices);
    
    idx_map = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for k = 1:L
        idx_map(mat2str(indices{k})) = k;
    end
    
    S_prev = zeros(1, L);
    
    for j = 1:N
        delta_X = X(j+1, :) - X(j, :);
        
        sig_increment = zeros(1, L);
        for k = 1:L
            idx = indices{k};
            len_idx = length(idx);
            term = 1 / factorial(len_idx);
            for l_idx = 1:len_idx
                term = term * delta_X(idx(l_idx) + 1);
            end
            sig_increment(k) = term;
        end
        
        S_current = S_prev;
        for k = 1:L
            idx_k = indices{k};
            len_k = length(idx_k);
            S_current(k) = S_current(k) + sig_increment(k);
            for p = 1:(len_k - 1)
                idx_prefix = idx_k(1:p);
                idx_suffix = idx_k(p+1:end);
                k_prefix = idx_map(mat2str(idx_prefix));
                k_suffix = idx_map(mat2str(idx_suffix));
                S_current(k) = S_current(k) + S_prev(k_prefix) * sig_increment(k_suffix);
            end
        end
        S_prev = S_current;
    end
    S_final = S_prev;
end
