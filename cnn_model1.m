function net = cnn_model()
    layers = [
        imageInputLayer([224 224 3])

        convolution2dLayer(3, 32, 'Padding', 'same', 'WeightLearnRateFactor', 0.001)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 64, 'Padding', 'same', 'WeightLearnRateFactor', 0.001)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 128, 'Padding', 'same', 'WeightLearnRateFactor', 0.001)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        dropoutLayer(0.6)  % Increased dropout

        fullyConnectedLayer(128)
        reluLayer
        dropoutLayer(0.6)  % Increased dropout

        fullyConnectedLayer(3)  % 3 classes: Normal, Mild, Severe
        softmaxLayer
        classificationLayer
    ];

    net = layerGraph(layers);
end
