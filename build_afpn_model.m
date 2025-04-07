function layers = build_afpn_model()
    layers = [
        imageInputLayer([128 128 3], 'Name', 'input')
        
        % Feature extraction layers
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
        
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool3')
        
        convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv4')
        batchNormalizationLayer('Name', 'bn4')
        reluLayer('Name', 'relu4')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool4')
        
        % Additional layers for feature pyramid
        convolution2dLayer(3, 512, 'Padding', 'same', 'Name', 'conv5')
        batchNormalizationLayer('Name', 'bn5')
        reluLayer('Name', 'relu5')
        
        % Fully connected layers
        fullyConnectedLayer(3, 'Name', 'fc') % Adjust number of classes
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
end