function net = cnn_model()
    layers = [
        imageInputLayer([224 224 3])

        convolution2dLayer(3, 32, 'Padding', 'same', 'WeightL2Factor', 0.01) 
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 64, 'Padding', 'same', 'WeightL2Factor', 0.01)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 128, 'Padding', 'same', 'WeightL2Factor', 0.01)
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 256, 'Padding', 'same', 'WeightL2Factor', 0.01) % Added new layer
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        dropoutLayer(0.6)

        fullyConnectedLayer(512) % Increased FC neurons
        reluLayer
        dropoutLayer(0.6)

        fullyConnectedLayer(3)  % 3 classes: Normal, Mild, Severe
        softmaxLayer
        classificationLayer
    ];

    net = layerGraph(layers);
end
