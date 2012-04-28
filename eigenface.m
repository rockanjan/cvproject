% performs eigenface based recognition
clear;
close all;

%load person_yale_30sub_40images_histeq
load person_orl_40
%load yale_histeq
%load yale_histeq
%load person_ioe_histeq

FRACTION = .4; %fraction of eigen vectors to use
remove_first = 0; %remove this number of eigen vectors (lighting variants)
trainsize = 5;
final_total = 0;
final_correct = 0;
for iter = 1:50
    %randomize the images
    for i=1:SUBJECTS
        faces = person(i).faces;
        beforerand = cell2mat(person(i).faces(1));
        randvalues = rand(1, length(faces));
        [rval rind] = sort(randvalues);
        person(i).faces = faces(rind);
        afterrand = cell2mat(person(i).faces(1));
        %{
        subplot(1,2,1)
        imshow(beforerand);
        subplot(1,2,2)
        imshow(afterrand);
        pause()
        %}
    end
    %% start eigenface computation
    %append the images in columns of a matrix A
    images = []; 
    totaltrain = 0;
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
    %whos images meanimage subtractedmean X Y
    % after finding eigen vector of Y, eigen vector of X = subtractedmean * V
    % eigen values are the same

    [V, lambda] = eig(Y);

    images = double(images);
    eigenfaces = A * V; %eigenvectors
    lambda_values = zeros(1, size(lambda, 2));
    for i=1:size(lambda,1)
        lambda_values(i) = lambda(i,i);
    end
    %sort the vectors according to eigenvalues
    [tr index] = sort(lambda_values, 'descend');
    eigenfaces = eigenfaces(:, index);
    lambda_values = lambda_values(index);
    
    %{
    figure
    plot(1:length(lambda_values), lambda_values);
    xlabel('Eigenvalue index');
    ylabel('Eigenvalues');
    %}
    
    vectorsize = ceil( FRACTION * length(lambda_values) );
    eigenfacestouse = eigenfaces(:, remove_first+1:vectorsize+remove_first);
%{
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
%}
    %% Find weight (coordinate) of projection of training images on each of the eigenvectors
    %normalize
    for i = 1:size(eigenfacestouse, 2)
        eigenfacestouse(:, i) = eigenfacestouse(:,i) ./ norm(eigenfacestouse(:, i));
    end
    wtrain = eigenfacestouse' * A;
    %% Testing
    %figure;
    testimagenumber = 0;
    correct = 0;
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
            %disp(['matched image ' num2str(index(1))]);
            idPersonPredicted = floor((index(1) - 1) / trainsize) + 1;
            %disp(['matched image id ' num2str(idPersonPredicted)]);

            if( i == idPersonPredicted)
                %disp('error');
                correct = correct + 1;
            end
            %{
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
            %pause();
            %}
        end
    end
    disp(['correct % : ' num2str(correct/testimagenumber * 100)]);
    final_total = final_total + testimagenumber;
    final_correct = final_correct + correct;
    
end
close all
disp(['   final correct % ' num2str(100 * final_correct / final_total) ]);
