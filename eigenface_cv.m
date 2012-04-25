% performs eigenface based recognition
clear;
close all;

%load
load person
FRACTION = .4; %fraction of eigen vectors to use

%% cross validation
tic
folditer = 5;
fold = folditer;
fold_size = length(person(1).faces) / fold;
train_size = (fold-1) * fold_size;
total_train = train_size * SUBJECTS;
all_size = train_size + fold_size;
disp(['folds ' num2str(fold)]);
final_total_count = 0;
final_correct_count = 0;
for thisfold=1:fold %fold times
    test_start = (thisfold-1) * fold_size + 1;
    test_end = test_start + fold_size - 1;
    train_start1 = 1;
    train_end1 = test_start - 1;
    train_start2 = test_end + 1;
    train_end2 = length(person(1).faces);
    disp(['teststart testend trainstart1 trainend1 trainstart2 trainend2 = ' num2str([test_start test_end train_start1 train_end1 train_start2 train_end2])]);
    total_count = 0;
    correct_count = 0;
    % start eigenface computation
    %append the images in columns of a matrix A
    clear images;
    images = [];
    for i=1:SUBJECTS
        for j=train_start1:train_end1
            face = cell2mat(person(i).faces(j));
            facecol = reshape(face, [], 1);
            images = [images facecol];
        end
        for j=train_start2:train_end2
            face = cell2mat(person(i).faces(j));
            facecol = reshape(face, [], 1);
            images = [images facecol];
        end
    end
    %disp(['images array size = ' num2str(size(images))]);
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
    %Y = 1/(SUBJECTS * train_size) * A' * A;
    Y = A' * A;
    %whos images meanimage subtractedmean X Y
    % after finding eigen vector of Y, eigen vector of X = subtractedmean * V
    % eigen values are the same
    [V, lambda] = eig(Y);
    eigenfaces = A * V; %eigenvectors
    clear lambda_values;
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

    % plot eigenfaces and mean image
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

    % Find weight (coordinate) of projection of training images on each of the eigenvectors
    %wtrain = zeros(vectorsize, total_train);
    wtrain = eigenfacestouse' * A;
    % Testing
    figure;
    testimagenumber = 0;
    error = 0;
    for i=1:SUBJECTS
        for j=test_start:test_end
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

            distanceToTrain = zeros(total_train, 1);
            %find distance between this test image and training images
            for k=1:total_train
                distanceToTrain(k) = norm(wtrain(:,k) - wtest, 2);
            end
            
            [tr index] = sort(distanceToTrain);
            %manage image offset due to cross validation
            %TODO: Check
            idPersonPredicted = floor((index(1) - 1) / train_size) + 1;
            image_id = mod( index(1), train_size);
            if(image_id == 0)
                image_id = train_end2;
            else
                if(image_id >= test_start)
                    image_id = image_id + fold_size;
                end
            end
            size(index);
            %pause();
            %disp(['index= ' num2str(index(1)) ' subj_id = ' num2str(idPersonPredicted) ' image_id = ' num2str(image_id)]); 
            %disp(['matched image id ' num2str(image_id)]);
            %disp(['matched person id ' num2str(idPersonPredicted)]);

            if( i ~= idPersonPredicted)
                disp('error');
                error = error + 1;
            end
            %display
            %{
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
            %}
        end
    end
    disp(['error percent : ' num2str(error/testimagenumber * 100)]);
    close all
    final_total_count = final_total_count + testimagenumber;
    final_correct_count = final_correct_count + (testimagenumber - error);
end
disp(['final accuracy : ' num2str(final_correct_count/final_total_count * 100)]);


