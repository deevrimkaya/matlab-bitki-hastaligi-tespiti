% Kaynak: Word belgesi, Deneme 4. imdsTrain, imdsVal ve imds onceki hazirlik adimina baglidir.
% Orijinal kod aktarimi; calistirilmasi dogrulanmadi.
% Veri Çoğaltma (Data Augmentation) eklenerek başarı artırılmıştır.
augmenter = imageDataAugmenter('RandXReflection', true, 'RandRotation', [-20 20]);
augimdsTrain = augmentedImageDatastore([224 224], imdsTrain, 'DataAugmentation', augmenter);
% İleri Seviye Model Mimarisi
baslangicModeli = [
    imageInputLayer([224 224 3], 'Name', 'Giris')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'Relu1')  
    % Bottleneck (Darboğaz) Bloğu: Parametre sayısını optimize eder
    convolution2dLayer(1, 16, 'Name', 'DarBogaz_1') 
    reluLayer('Name', 'Relu_DB1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'Genislet_1')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'elu2')
    globalAveragePooling2dLayer('Name', 'GAP')
    fullyConnectedLayer(numel(categories(imds.Labels)), 'Name', 'FC_Final')
    softmaxLayer()
    classificationLayer()];
% Eğitim Ayarları (Adam Optimizer ile daha hızlı öğrenme)
options = trainingOptions('adam', 'MaxEpochs', 5, 'InitialLearnRate', 3e-4, 'Plots', 'training-progress');
finalModel = trainNetwork(augimdsTrain, baslangicModeli, options);
% Profesyonel Matris Görselleştirme
[YPred, ~] = classify(finalModel, augmentedImageDatastore([224 224], imdsVal));
fig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.8 0.7]);
cm = confusionchart(imdsVal.Labels, YPred, 'FontSize', 10);
cm.RowSummary = 'row-normalized'; cm.ColumnSummary = 'column-normalized';
