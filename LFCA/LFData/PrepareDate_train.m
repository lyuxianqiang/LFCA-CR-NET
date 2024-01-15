%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate training data for LFCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all;

%% path
data_folder = 'Y:\LF_Dataset\Dataset_kalantari_SIG2016\SIGGRAPHAsia16_ViewSynthesis_Trainingset';
savepath = 'train_LFCA_Kalantari.mat';
an = 7;

%%initilization
lf = zeros(an,an,600,600,3,'uint8');
lfSize = zeros(2,1,'uint16');
count = 0;

%% read datasets
data_list = dir(data_folder);
data_list = data_list(3:end);


%% read lfs
for i_lf = 1:length(data_list)
    lfname = data_list(i_lf).name;
    read_path = fullfile(data_folder,lfname);
    lf_rgb = read_eslf(read_path,14,an);

    H = size(lf_rgb,1);
    W = size(lf_rgb,2);

    count = count +1;
    lf(:,:,1:H,1:W,:,count) = permute(lf_rgb,[4,5,1,2,3]);
    lfSize(:,count)=[H,W];         
end
    
%% generate data
order = randperm(count);
lf = permute(lf(:, :, :, :, :, order),[6,1,2,3,4,5]); %[u,v,x,y,c,N] -> [N,u,v,x,y,c]  
lfSize = permute(lfSize(:,order),[2,1]);  %[N,2]

%% writing to mat
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

save(savepath,'lf','lfSize','-v7.3');

