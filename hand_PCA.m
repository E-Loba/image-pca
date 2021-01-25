% clear;
% old = path;
% new = 'C:\Users\admin\Documents\pattern recognition\handdata';
% path(old, new);
% my_dir = dir(fullfile('C:','Users','admin','Documents','pattern recognition','handdata','*.bmp'));

% read the image data

img_data = zeros(41,4096);
count = 1;
for file = my_dir'
    [temp, dummy] = imread(file.name);
    % convert image into feature vector
    % each pixel is a feature
    img_data(count,:) = reshape(temp',1,[]);
    count = count + 1;
end

img_min = min(img_data);
img_max = max(img_data);

% subtract mean from sample
sample_mean = mean(img_data,2);
normalised = img_data - sample_mean;
% compute the PCA
covariance_matrix = (normalised' * normalised)/(42 * 4096 -1);
[eigVec, eigVal] = eig(covariance_matrix);
% sort eigenvecotrs according to eigenvalue size 
[eigVal, index] = sort(diag(eigVal), 'descend');
sorted_eigVec = eigVec(:,index);

% plot the CCR curve
target_ccr = sum(eigVal)*0.8;
plot([0,cumsum(eigVal)']);
hold on
plot(ones(1,42)*target_ccr);
legend("cumulative sum of PCA contribution", "80% line");
axis([1 42 0 700]);

% this is how many principal components are needed for 80% CCR
step = 0;
count = 0;
while step < target_ccr
    count = count + 1
    step = step + eigVal(count);
end

% the top 3 components provide most of the data variance

pr_cmp = sorted_eigVec(:,1:3);

% convert the principal components to brightness value range of image
% formula for translation = (new_range/old_range)*(v-old_min)+new_min

for i = 1:3
    v = pr_cmp(:,i);
    if i == 1
        new_min = min(img_min);
        new_range = max(img_max) - new_min;
    elseif i == 2
        new_min = max(img_min);
        new_range = min(img_max) - new_min;
    else
        new_min = mean(img_min);
        new_range = mean(img_max) - new_min;
    end
    old_min = min(v);
    old_range = max(v) - old_min;
    pr_cmp(:,i) = (v - old_min) * (new_range/old_range) + new_min;
    
end

% visualise the top 3 components
figure;
imshow(reshape(uint8(pr_cmp(:,1)), 64, 64), dummy);
title("first component");
figure;
imshow(reshape(uint8(pr_cmp(:,2)), 64, 64), dummy);
title("second component");
figure;
imshow(reshape(uint8(pr_cmp(:,3)), 64, 64), dummy);
title("third component");

% reconstruct some images
% compute the projection of an image on the principal axes
principal_axes = sorted_eigVec(:,1:3);
translated_images = mtimes(img_data, principal_axes);
test_img3 = translated_images(22,:) .* principal_axes;
test_img = (test_img3(:,1) + test_img3(:,2) + test_img3(:,3));

% follow the same procedure when visualising the principal axes
new_min = min(img_min);
new_range = max(img_max) - new_min;
old_min = min(test_img);
old_range = max(test_img) - old_min;
test_img = uint8((test_img - old_min) * (new_range/old_range) + new_min);
figure
imshow(reshape(uint8(test_img), 64, 64), dummy);
title("reconstructed image");