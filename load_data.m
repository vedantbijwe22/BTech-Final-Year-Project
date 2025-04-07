function [images, labels] = load_data(base_path)
    images = [];
    labels = [];
    labelNames = {'Normal', 'Mild', 'Severe'};
    
    for i = 1:length(labelNames)
        label = labelNames{i};
        label_path = fullfile(base_path, label);
        
        if ~isfolder(label_path)
            fprintf('Label path does not exist: %s\n', label_path);
            continue;
        end
        
        patientFolders = dir(label_path);
        for j = 1:length(patientFolders)
            if patientFolders(j).isdir && ~ismember(patientFolders(j).name, {'.', '..'})
                patient_path = fullfile(label_path, patientFolders(j).name);
                folderFiles = dir(patient_path);
                for k = 1:length(folderFiles)
                    if folderFiles(k).isdir && ~ismember(folderFiles(k).name, {'.', '..'})
                        imgFiles = dir(fullfile(patient_path, folderFiles(k).name, '*.jpg')); % Adjust extension as needed
                        for l = 1:length(imgFiles)
                            img_path = fullfile(patient_path, folderFiles(k).name, imgFiles(l).name);
                            try
                                img = imresize(imread(img_path), [128, 128]);
                                images = cat(4, images, im2double(img)); % Store images in 4D array
                                labels = [labels; string(label)]; % Store labels
                            catch ME
                                fprintf('Error loading image %s: %s\n', img_path, ME.message);
                            end
                        end
                    end
                end
            end
        end
    end
    
    fprintf('Loaded %d images with labels: %s\n', size(images, 4), join(unique(labels), ', '));
end