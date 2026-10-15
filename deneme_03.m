% Kaynak: Word belgesi, Deneme 3. Orijinal kod aktarimi; calistirilmasi dogrulanmadi.
imds = imageDatastore(pwd, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% Veri Ön İşleme (CMYK Temizliği)
okunabilir = true(numel(imds.Files), 1);
for i = 1:numel(imds.Files)
    info = imfinfo(imds.Files{i});
    if isfield(info, 'ColorType') && strcmp(info.ColorType, 'CMYK'), okunabilir(i)=0; end
end
imds = subset(imds, okunabilir);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomize');
% Model Mimarisi (Bottleneck Yapısı)
layers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3, 16, 'Padding', 'same'), batchNormalizationLayer(), reluLayer()
    convolution2dLayer(1, 16, 'Name', 'DarBogaz') 
    convolution2dLayer(3, 32, 'Padding', 'same'), batchNormalizationLayer(), reluLayer()
    globalAveragePooling2dLayer(), fullyConnectedLayer(4), softmaxLayer(), classificationLayer()];
% Eğitim (Başarıyı %77-78 bandında tutmak için LearnRate: 3.5e-4)
opts = trainingOptions('adam', 'MaxEpochs', 5, 'InitialLearnRate', 3.5e-4);
net = trainNetwork(imdsTrain, layers, opts);
% Analiz ve Görselleştirme
[YPred, ~] = classify(net, imdsVal);confusionchart(imdsVal.Labels, YPred, 'FontSize', 10, 'Title', 'Deneme 3: %77.8');
