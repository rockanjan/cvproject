% performs eigenface based recognition
clear;
close all;

load person
FRACTION = .4; %fraction of eigen vectors to use

trainsize = 8;
%% start eigenface computation
%append the images in columns of a matrix A
images = []; 

for i=1:SUBJECTS
    for j=1:trainsize
        face = cell2mat(person(i).faces(j));
        facecol = reshape(face, [], 1);
        images = [images facecol];
        totaltrain = totaltrain + 1;
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
%X = A * A';
% PCA of this covariance is computationally expensive, so workaround
%Y = 1/(SUBJECTS * trainsize) * A' * A;
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

%% Find weight (coordinate) of projection of training images on each of the eigenvectors
%wtrain = zeros(vectorsize, totaltrain);
wtrain = eigenfacestouse' * A;
%% Testing
figure;
testimagenumber = 0;
error = 0;
for i=1:SUBJECTS
    for j=trainsize+1:length(person(i).faces)
        testimagenumber = testimagenumber + 1;
        testim = cell2mat(person(i).faces(j));
        %imshow(im); pause();
        
        %subtract mean image
        testimdouble = double(testim) - reshape(meanimage, row, col);
        %imshow(uint8(im));
        %pause();
        
        %project mean subtracted image into the face space, 
        %calculate weight i.e. coordinate acc to new basis vectors
        testimvector = reshape(testimdouble, [], 1);
        wtest = eigenfacestouse' * testimvector;
        
        distanceToTrain = zeros(totaltrain, 1);
        %find distance between this test image and training images
        for k=1:totaltrain
            distanceToTrain(k) = norm(wtrain(:,k) - wtest, 2);
        end
        [tr index] = sort(distanceToTrain);
        disp(['matched image' num2str(index(1))]);
        idPersonPredicted = floor((index(1) - 1) / trainsize) + 1;
        disp(['matched image id ' num2str(idPersonPredicted)]);
        
        if( i ~= idPersonPredicted)
            error = error + 1;
        end
        %display
        subplot(1,4,1);
        imshow(testim)
        title('test image');
        firstmatch = uint8(reshape(images(:,index(1)), row, col));
        secondmatch = uint8(reshape(images(:,index(2)), row, col));
        thirdmatch = uint8(reshape(images(:,index(3)), row, col));
        subplot(1,4,2);
        imshow(firstmatch);
        title('first match');
        subplot(1,4,3);
        imshow(secondmatch);
        title('second match');
        subplot(1,4,4);
        imshow(thirdmatch);
        title('third match');
        pause();
    end
end
disp(['error : ' num2str(error/testimagenumber * 100)]);
close all



