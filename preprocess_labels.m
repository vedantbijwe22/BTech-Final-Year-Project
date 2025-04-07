function labels = preprocess_labels(labels)
    % Convert string labels to categorical
    labels = categorical(labels);
end