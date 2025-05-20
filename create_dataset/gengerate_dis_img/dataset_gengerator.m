%% setup
clear; clc;
addpath(genpath('code_imdistort'));


%% read the info of pristine images

sourcePath = 'F:/kadis700k/';
targetPath = 'E:/wxq/';

mkdir(targetPath,'SAQT_IQA')

for i =1:50
 num = sprintf('%03d', i);
 mkdir([targetPath, 'SAQT_IQA','/'],num2str(num));
end
 
tb = readtable('../info_save/similarity_img_in_kadid-10k.csv');
%% tb = readtable('../info_save/similarity_img_in_iqadataset.csv');

tb = table2cell(tb);

% Extract just the first column (source images)
source_images = tb(:,1);

%% generate distorted images in dist_imgs folder

for i = 1:size(source_images, 1)
    img_idx = sprintf('%03d', ceil(i/1000));
    ref_im = imread([sourcePath, 'ref_imgs/' tb{i,1}]);
    for dist_type = 1:25
    %dist_type = tb{i,2};
        for dist_level = 1:5
            [dist_im] = imdist_generator(ref_im, dist_type, dist_level);
            strs = split(tb{i,1},'.');
            dist_im_name = [strs{1}  '_' num2str(dist_type,'%02d')  '_' num2str(dist_level,'%02d') '.bmp'];
            disp(dist_im_name);
            imwrite(dist_im, [targetPath, 'SAQT_IQA','/', img_idx, '/', dist_im_name]);
        end
        
    end 
    
end







