% =========================================================================
% FILE: generate_indices.m (Re-used from previous example)
% =========================================================================
function indices = generate_indices(d, M)
    indices = {};
    current_level_indices = num2cell(0:d);
    
    for k = 1:M
        indices = [indices, current_level_indices];
        next_level_indices = {};
        for i = 1:length(current_level_indices)
            base_index = current_level_indices{i};
            for j = 0:d
                next_level_indices{end+1} = [base_index, j];
            end
        end
        current_level_indices = next_level_indices;
    end
end
