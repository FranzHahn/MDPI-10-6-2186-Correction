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

if ~exist([features_path 'kadid10k_iv3_feats_no_ft.mat'],'file')
    disp(' - Extracting Activations - ');
    Features = GetAllFeatures(numberOfImages, images_path, names, net, Layers, lengthOfFeats);
    save([features_path 'kadid10k_iv3_feats_no_ft.mat'],'Features','-v7.3','-nocompression');
else
    disp(' - Loading Activations - ');
    load([features_path 'kadid10k_iv3_feats_no_ft.mat']);
end

%% Faulty SVR Training

origs = unique(t.ref_img);
numberOfTrainImages = ceil(len(origs)*0.8)*125;

for i=1:len(seeds)
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
    
    save([results_path 'kadid10k_iv3_results_no_ft_' num2str(seeds(i)) '_faulty.mat'],'YHat','YTest','-v7.3','-nocompression')
end

%%

for i=1:len(seeds)
    load([results_path 'kadid10k_iv3_results_no_ft_' num2str(seeds(i)) '_faulty.mat'],'YHat','YTest')

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

%% Correct SVR Training

for i=1:len(seeds)
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
    
    save([results_path 'kadid10k_iv3_results_no_ft_' num2str(seeds(i)) '_correct.mat'],'YHat','YTest','-v7.3','-nocompression')
end


%%

for i=1:len(seeds)
    load([results_path 'kadid10k_iv3_results_no_ft_' num2str(seeds(i)) '_correct.mat'],'YHat','YTest')

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