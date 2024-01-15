%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate test data for LFCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;close all;

%% params

%  Kalantari
data_folder = '\path to the eslf files';
savepath = 'test_LFCA_Kalantari.mat';
an = 7;
h = 372;
w = 540;

%% initilization
lf   = [];
LF_name = {};
count = 0;
data_list = dir(data_folder);
data_list = data_list(3:end);

%% generate data
for k = 1:length(data_list)
    lfname = data_list(k).name;
    read_path = fullfile(data_folder,lfname);
    lf_gt_rgb = read_eslf(read_path, 14, an); %[h,w,3,ah,aw]
    lf_gt_rgb = lf_gt_rgb(1:h,1:w,:,:,:);   
    lf = cat(6,lf,lf_gt_rgb); %[h,w,3,ah,aw,N]
    LF_name = cat(1,LF_name,abs(lfname(1:end-4))); %[N,1]
end

lf = permute(lf,[6,4,5,1,2,3]);  %[h,w,3,ah,aw,N]==>[N,u,v,h,w,3]

%% save data
if exist(savepath,'file')
  fprintf('Warning: replacing existing file %s \n', savepath);
  delete(savepath);
end 

save(savepath, 'lf', 'LF_name','-v6');
