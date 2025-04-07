function [X_train, X_val, y_train, y_val] = split_data(images, labels)
    % Split data into training and validation sets
    cv = cvpartition(labels, 'HoldOut', 0.2);
    idx = cv.test;

    % Separate to training and validation data
    X_train = images(:, :, :, ~idx);
    y_train = labels(~idx);
    X_val = images(:, :, :, idx);
    y_val = labels(idx);
end