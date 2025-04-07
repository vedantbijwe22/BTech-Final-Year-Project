% Main script to train the AFPN model
clear; clc;

% Define the base path for the dataset
base_path = 'D:\Notes\Phy\k\cursed\Cervical Cancer\data';

% Load and preprocess data
[images, labels] = load_data(base_path);
labels = preprocess_labels(labels);

% Split dataset into training and validation sets
[X_train, X_val, y_train, y_val] = split_data(images, labels);

% Data augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ... % Random rotation between -20 and 20 degrees
    'RandXReflection', true, ...    % Random horizontal flip
    'RandYReflection', true);       % Random vertical flip

augmentedTrainData = augmentedImageDatastore([128 128 3], X_train, y_train, ...
    'DataAugmentation', imageAugmenter);

% Define the model
model = build_afpn_model();

% Train the model
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...           % Increase the number of epochs
    'MiniBatchSize', 32, ...
    'ValidationData', {X_val, y_val}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the model using augmented data
trainedModel = trainNetwork(augmentedTrainData, model, options);

% Evaluate the model
YPred = classify(trainedModel, X_val);
accuracy = sum(YPred == y_val) / numel(y_val);
fprintf('Validation Accuracy: %.4f\n', accuracy);

% Save the model
save('afpn_model_aug.mat', 'trainedModel');