% performs eigenface based recognition
clear;
close all;

person = struct('faces', {});

SUBJECTS = 10; %number of people
DATADIR = '/temp/cvproject/data/orl';
%DATADIR = '/temp/cvproject/org/att/orl_faces';
FRACTION = .4; %fraction of eigen vectors to use
total = 0;
row = 0;
col = 0;


trainsize = 8;


%read data
for i=1:SUBJECTS
    files =  dir(fullfile(DATADIR, ['s' num2str(i)], '*'));
    faces = {};
    count = 0;
    for j=1:size(files,1)
        if( ~ ( strcmp(files(j).name, '.') || strcmp(files(j).name, '..') || strcmp(files(j).isdir, '1') ) )
            %files(j).name
            im = imread(fullfile(DATADIR, ['s' num2str(i)], files(j).name) );
            count = count +1;
            [row col] = size(im);
            faces{count} = im;
            %imshow(im);
            %pause();
            total = total + 1;
        end
    end
    %store this persons all faces
    person(i).faces = faces;
end

disp(['total face images : ' num2str(total)]);

%verify
for i=1:SUBJECTS
    for j=1:size(person(i).faces, 1)
        %imshow(cell2mat(person(i).faces(j)));
        %pause();
    end
end

%% start eigenface computation
%append the images in columns of a matrix A
images = []; 
for i=1:SUBJECTS
    for j=1:trainsize
        face = cell2mat(person(i).faces(j));
        facecol = reshape(face, [], 1);
        images = [images facecol];
    end
end
%
images = double(images);

% find mean
meanimage = mean(images, 2); %mean in column DIM
%subtract mean image
A = zeros(size(images)); % A = mean subtracted image
for i=1:size(images,2)
    A(:, i) = images(:,i) - meanimage;
end
%covariance matrix
X = A * A';
% PCA of this covariance is computationally expensive, so workaround
Y = A' * A;
whos images meanimage subtractedmean X Y
% after finding eigen vector of Y, eigen vector of X = subtractedmean * V
% eigen values are the same

[V, lambda] = eig(Y);

images = double(images);
eigenfaces = A * V; %eigenvectors

for i=1:size(lambda,1)
    lambda_values(i) = lambda(i,i);
end
%sort the vectors according to eigenvalues
[tr index] = sort(lambda_values, 2, 'descend');
eigenfaces = eigenfaces(:, index);
lambda_values = lambda_values(index);
figure
plot(1:length(lambda_values), lambda_values);
xlabel('Eigenvalue index');
ylabel('Eigenvalues');
vectorsize = ceil( FRACTION * length(lambda_values) );
eigenfacestouse = eigenfaces(:,1:vectorsize);

%% plot eigenfaces and mean image
figure
colormap('gray');
for i=1:10
    subplot(2,5,i)
    %vector normalized, reshape to image
    imagesc(reshape(eigenfaces(:,i) ./ norm(eigenfaces(:,i)), row, col));
    axis off
    %imshow(reshape(eigenfaces(:,i) ./ (max(eigenfaces(:,i))) , row, col));
end
suptitle('Eigenfaces');
figure
imshow(uint8(reshape(meanimage, row, col)));
title('Mean image');

%% Find weight of projection of training images on each of the eigenvectors
w = eigenfaces(:, i)' * A;
%only take subset of these weights
w = w(1:vectorsize);

%% Testing
figure;
for i=1:SUBJECTS
    for j=trainsize:size(person(i).faces, 1)
        imshow(cell2mat(person(i).faces(j))); pause();
    end
end



