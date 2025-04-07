[X_train_pap, X_test_pap, X_train_colpo, X_test_colpo, y_train, y_test] = load_data();
net_pap = cnn_model();
net_colpo = cnn_model();

augmenter = imageDataAugmenter( ...
    'RandRotation', [-30 30], ... % Rotate images randomly between -30 to 30 degrees
    'RandXReflection', true, ...  % Randomly flip images horizontally
    'RandXScale', [0.8, 1.2], ... % Slightly scale images in X direction
    'RandYScale', [0.8, 1.2]);    % Slightly scale images in Y direction


augimdsTrain = augmentedImageDatastore([224 224 3], X_train_pap, y_train, 'DataAugmentation', augmenter);
augimdsTest = augmentedImageDatastore([224 224 3], X_test_pap, y_test);

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augimdsTest, ...
    'ValidationPatience', 5, ... % Early stopping
    'InitialLearnRate', 0.0001, ...
    'L2Regularization', 0.0005, ...
    'Plots', 'training-progress');

net_pap = trainNetwork(augimdsTrain, net_pap, options);
net_colpo = trainNetwork(augimdsTrain, net_colpo, options);

save('pap_model1.mat', 'net_pap');
save('colpo_model1.mat', 'net_colpo');
