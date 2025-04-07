function plot_results(training_info)
    figure;
    
    % Accuracy Plot
    subplot(1,2,1);
    plot(training_info.TrainingAccuracy, 'b', 'LineWidth', 1.5);
    hold on;
    plot(training_info.ValidationAccuracy, 'r', 'LineWidth', 1.5);
    xlabel('Epochs');
    ylabel('Accuracy');
    legend('Training Accuracy', 'Validation Accuracy');
    title('Training and Validation Accuracy');
    
    % Loss Plot
    subplot(1,2,2);
    plot(training_info.TrainingLoss, 'b', 'LineWidth', 1.5);
    hold on;
    plot(training_info.ValidationLoss, 'r', 'LineWidth', 1.5);
    xlabel('Epochs');
    ylabel('Loss');
    legend('Training Loss', 'Validation Loss');
    title('Training and Validation Loss');
end
