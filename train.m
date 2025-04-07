% Main script to train the AFPN model
clear; clc;

% Define the base path for the dataset
base_path = 'D:\Notes\Phy\k\cursed\Cervical Cancer\data';

% Load and preprocess data
[images, labels] = load_data(base_path);
labels = preprocess_labels(labels);

% Split dataset into training and validation sets
[X_train, X_val, y_train, y_val] = split_data(images, labels);

% Define the model
model = build_afpn_model();

% Train the model
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {X_val, y_val}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the model
trainedModel = trainNetwork(X_train, y_train, model, options);

% Evaluate the model
YPred = classify(trainedModel, X_val);
accuracy = sum(YPred == y_val) / numel(y_val);
fprintf('Validation Accuracy: %.4f\n', accuracy);

% Save the model
save('afpn_model.mat', 'trainedModel');