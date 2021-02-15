clear all
close all

seeds = [42, 69, 322, 1337, 9000];

images_path = [pwd filesep 'images' filesep];
features_path = [pwd filesep 'features' filesep];
networks_path = [pwd filesep 'networks' filesep];
results_path = [pwd filesep 'results' filesep];

t = readtable('dmos.csv');
mos = t.dmos;
names = t.dist_img;

numberOfImages = size(mos,1);

net = inceptionv3;    
Layers = {'mixed0','mixed1','mixed2','mixed3','mixed4','mixed5','mixed6',...
    'mixed7','mixed8','mixed9','mixed10'};

lengthOfFeats = 10048;
depth  = size(Layers,2);

lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'predictions_softmax','ClassificationLayer_predictions'});
newLayers = [fullyConnectedLayer(1,'Name','fcfin')
             regressionLayer('Name','regoutput')];
lgraph = addLayers(lgraph, newLayers);
net = connectLayers(lgraph,'predictions','fcfin');

names = arrayfun(@(x) (strcat(images_path,filesep,x)),names);

origs = unique(t.ref_img);

numberOfTrainImages = floor(len(origs)*0.8)*125;
numberOfValImages = ceil(len(origs)*0.2)*125;

%% Faulty Fine-Tuning

for i=1:len(seeds)
    if ~exist([networks_path 'kadid10k_iv3_trained_network_faulty_' num2str(seeds(i)) '.mat'],'file')
        disp(seeds(i));
        rng(seeds(i));
        p = randperm(numberOfImages); 


        train = table(names(p(1:numberOfTrainImages)),mos(p(1:numberOfTrainImages)), 'VariableNames', {'Files','Labels'});
        vali = table(names(p(numberOfTrainImages+1:end)),mos(p(numberOfTrainImages+1:end)), 'VariableNames', {'Files','Labels'});
        imdstrain = augmentedImageDatastore([299 299 3],train,'OutputSizeMode','randcrop');
        imdsvali = augmentedImageDatastore([299 299 3],vali,'OutputSizeMode','randcrop');
        miniBatchSize = 28;
        validationFrequency = floor(numberOfTrainImages/miniBatchSize);

        opts = trainingOptions('adam', ...
                                'InitialLearnRate',0.0001,...
                                'MaxEpochs',100, ...
                                'Shuffle','every-epoch', ...
                                'Plots','training-progress', ...
                                'MiniBatchSize',miniBatchSize,...
                                'DispatchInBackground',true,...
                                'Verbose',false,...
                                'ValidationData',imdsvali,...
                                'ValidationFrequency',validationFrequency,...
                                'ValidationPatience',3);

        [netft, info] = trainNetwork(imdstrain,net,opts);

        save([networks_path 'kadid10k_iv3_trained_network_faulty_' num2str(seeds(i)) '.mat'],'netft','imdstrain','imdsvali','info','-v7.3','-nocompression');
    end
end

%% Feature Extraction

names = t.dist_img;
if ~exist([features_path 'kadid10k_iv3_feats_faulty_ft_' num2str(seeds(i)) '.mat'],'file')
    for i=1:len(seeds)
        load([networks_path 'kadid10k_iv3_trained_network_faulty_' num2str(seeds(i)) '.mat'],'netft');
        Features = GetAllFeatures(numberOfImages, images_path, names, netft, Layers, lengthOfFeats);
        save([features_path 'kadid10k_iv3_feats_faulty_ft_' num2str(seeds(i)) '.mat'],'Features','-v7.3','-nocompression')
    end
end

%% Faulty SVR Training

if ~exist([results_path 'kadid10k_iv3_results_ft_' num2str(seeds(i)) '_faulty.mat'],'file')
    for i=1:len(seeds)
        load([features_path 'kadid10k_iv3_feats_faulty_ft_' num2str(seeds(i)) '.mat']);
        disp(seeds(i));
        rng(seeds(i));
        p = randperm(numberOfImages); 

        Target = mos(p);
        Data   = Features(p,:);

        YTrain    = Target(1:numberOfTrainImages);
        DataTrain = Data(1:numberOfTrainImages,:);

        YTest    = Target(numberOfTrainImages+1:end,:);
        DataTest = Data(numberOfTrainImages+1:end,:);

        Mdl  = fitrsvm(DataTrain,YTrain,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
        models{i} = Mdl;
        YHat = predict(Mdl,DataTest);

        save([results_path 'kadid10k_iv3_results_ft_' num2str(seeds(i)) '_faulty.mat'],'YHat','YTest','-v7.3','-nocompression')
    end
end

%%

for i=1:len(seeds)
    load([results_path 'kadid10k_iv3_results_ft_' num2str(seeds(i)) '_faulty.mat'],'YHat','YTest')

    SROCC(i) = corr(YHat,YTest,'Type','Spearman');
    KROCC(i) = corr(YHat,YTest,'Type','Kendall');
    
    beta(1) = max(YTest);
    beta(2) = min(YTest);
    beta(3) = mean(YTest);
    beta(4) = 0.5;
    beta(5) = 0.1;

    %fitting a curve using the data
    [bayta,ehat,J] = nlinfit(YHat,YTest,@logistic,beta);
    [pred_test_mos_align, junk] = nlpredci(@logistic,YHat,bayta,ehat,J);

    PLCC(i) = corr(pred_test_mos_align,YTest,'type','Pearson');
    RMSE(i) = sqrt(mean((YTest - pred_test_mos_align).^2));
end

disp(['PLCC: ' num2str(round(mean(PLCC(:)),2)) ' ($\pm$' num2str(round(std(PLCC(:)),3)) ')']);
disp(['SROCC: ' num2str(round(mean(SROCC(:)),2)) ' ($\pm$' num2str(round(std(SROCC(:)),3)) ')']);

%% Correct Fine-Tuning

for i=1:len(seeds)
    if ~exist([networks_path 'kadid10k_iv3_trained_network_' num2str(seeds(i)) '.mat'],'file')
        disp(seeds(i));
        rng(seeds(i));
        p = randperm(81);
        k = [];
        for j=1:len(p)
            k = [k [1:125]+((p(j)-1)*125)];
        end
        p = k;


        train = table(names(p(1:numberOfTrainImages)),mos(p(1:numberOfTrainImages)), 'VariableNames', {'Files','Labels'});
        vali = table(names(p(numberOfTrainImages+1:end)),mos(p(numberOfTrainImages+1:end)), 'VariableNames', {'Files','Labels'});
        imdstrain = augmentedImageDatastore([299 299 3],train,'OutputSizeMode','randcrop');
        imdsvali = augmentedImageDatastore([299 299 3],vali,'OutputSizeMode','randcrop');
        miniBatchSize = 28;
        validationFrequency = floor(numberOfTrainImages/miniBatchSize);

        opts = trainingOptions('adam', ...
                                'InitialLearnRate',0.0001,...
                                'MaxEpochs',100, ...
                                'Shuffle','every-epoch', ...
                                'Plots','training-progress', ...
                                'MiniBatchSize',miniBatchSize,...
                                'DispatchInBackground',true,...
                                'Verbose',false,...
                                'ValidationData',imdsvali,...
                                'ValidationFrequency',validationFrequency,...
                                'ValidationPatience',3);

        [netft, info] = trainNetwork(imdstrain,net,opts);

        save([networks_path 'kadid10k_iv3_trained_network_' num2str(seeds(i)) '.mat'],'netft','imdstrain','imdsvali','info','-v7.3','-nocompression');
    end
end

%% Feature Extraction

names = t.dist_img;
if ~exist([features_path 'kadid10k_iv3_feats_ft_' num2str(seeds(i)) '.mat'],'file')
    for i=1:len(seeds)
        load([networks_path 'kadid10k_iv3_trained_network_' num2str(seeds(i)) '.mat'],'netft');
        Features = GetAllFeatures(numberOfImages, images_path, names, netft, Layers, lengthOfFeats);
        save([features_path 'kadid10k_iv3_feats_ft_' num2str(seeds(i)) '.mat'],'Features','-v7.3','-nocompression')
    end
end

%% Correct SVR Training

if ~exist([results_path 'kadid10k_iv3_results_ft_' num2str(seeds(i)) '.mat'],'file')
    for i=1:len(seeds)
        load([features_path 'kadid10k_iv3_feats_ft_' num2str(seeds(i)) '.mat']);
        disp(seeds(i));
        rng(seeds(i));
        p = randperm(81);
        k = [];
        for j=1:len(p)
            k = [k [1:125]+((p(j)-1)*125)];
        end
        p = k;

        Target = mos(p);
        Data   = Features(p,:);

        YTrain    = Target(1:numberOfTrainImages);
        DataTrain = Data(1:numberOfTrainImages,:);

        YTest    = Target(numberOfTrainImages+1:end,:);
        DataTest = Data(numberOfTrainImages+1:end,:);

        Mdl  = fitrsvm(DataTrain,YTrain,'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
        models{i} = Mdl;
        YHat = predict(Mdl,DataTest);

        save([results_path 'kadid10k_iv3_results_ft_' num2str(seeds(i)) '.mat'],'YHat','YTest','-v7.3','-nocompression')
    end
end

%%

for i=1:len(seeds)
    load([results_path 'kadid10k_iv3_results_ft_' num2str(seeds(i)) '.mat'],'YHat','YTest')

    SROCC(i) = corr(YHat,YTest,'Type','Spearman');
    KROCC(i) = corr(YHat,YTest,'Type','Kendall');
    
    beta(1) = max(YTest);
    beta(2) = min(YTest);
    beta(3) = mean(YTest);
    beta(4) = 0.5;
    beta(5) = 0.1;

    %fitting a curve using the data
    [bayta,ehat,J] = nlinfit(YHat,YTest,@logistic,beta);
    [pred_test_mos_align, junk] = nlpredci(@logistic,YHat,bayta,ehat,J);

    PLCC(i) = corr(pred_test_mos_align,YTest,'type','Pearson');
    RMSE(i) = sqrt(mean((YTest - pred_test_mos_align).^2));
end

disp(['PLCC: ' num2str(round(mean(PLCC(:)),2)) ' ($\pm$' num2str(round(std(PLCC(:)),3)) ')']);
disp(['SROCC: ' num2str(round(mean(SROCC(:)),2)) ' ($\pm$' num2str(round(std(SROCC(:)),3)) ')']);