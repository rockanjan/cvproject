% performs eigenface based recognition
clear;
close all;

load person_orl_40
%load person_orl_40
trainsize = 5;
final_correct = 0;
final_total = 0;
for iter = 1:50
    %randomize the images
    for i=1:SUBJECTS
        faces = person(i).faces;
        beforerand = cell2mat(person(i).faces(1));
        randvalues = rand(1, length(faces));
        [rval rind] = sort(randvalues);
        person(i).faces = faces(rind);
        afterrand = cell2mat(person(i).faces(1));
    end
    %% start eigenface computation
    %append the images in columns of a matrix A
    images = []; 
    totaltrain=0;
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
    Y = A' * A;
    [V, lambda] = eig(Y);
    images = double(images);
    eigenfaces = A * V; %eigenvectors
    for i=1:size(lambda,1)
        lambda_values(i) = lambda(i,i);
    end
    %sort the vectors according to eigenvalues
    [tr index] = sort(lambda_values, 'descend');
    eigenfaces = eigenfaces(:, index);
    lambda_values = lambda_values(index);
    eigenfacestouse = eigenfaces(:, 1: (totaltrain - SUBJECTS));

    %% project in eigen space (coordinates for the images in new space)
    images_eigen_space = eigenfacestouse' * A;

    %A_pca has the images projected in new space (in each column)
    %find overall mean
    mean_eigen_space = mean(images_eigen_space, 2);
    %find class specific mean images
    class_mean_eigen_space = zeros(size(mean_eigen_space, 1), SUBJECTS);
    for i=1:SUBJECTS
        start = (i-1) * trainsize + 1;
        col_span = start : (start+trainsize-1);
        class_mean_eigen_space(:, i) = mean(images_eigen_space(:, col_span), 2);
    end
    
    %% compute total, between and within scatter matrices (S, Sb, Sw)
    A_eigen_space = images_eigen_space - repmat(mean_eigen_space, 1, totaltrain);
    S = A_eigen_space * A_eigen_space';
    Sb = zeros(size(S));
    Sw = zeros(size(S));

    %between
    for i=1:SUBJECTS
        a = (class_mean_eigen_space(:, i) - mean_eigen_space);
        Sb = Sb +  trainsize * a * a';
    end

    %within
    img_index = 0;
    for i=1:SUBJECTS
        class_sigma = zeros(size(S));
        for j=1:trainsize
            img_index = img_index + 1;
            a = (images_eigen_space(:, img_index) - class_mean_eigen_space(:, i));
            class_sigma = class_sigma +  a * a';
        end
        Sw = Sw + class_sigma;
    end


    %% optimization
%    [fisher_vec, fisher_lambda] = eig(Sb/Sw); % Cost function J = inv(Sw) * Sb
    [fisher_vec, fisher_lambda] = eig(Sb, Sw);
    fisher_lambda_values = zeros(1, size(fisher_lambda, 2));
    for i=1:size(fisher_lambda,1)
        fisher_lambda_values(i) = fisher_lambda(i,i);
    end
    %sort the vectors according to eigenvalues
    [tr index] = sort(fisher_lambda_values, 'descend');
    fisher_all_faces = fisher_vec(:, index);
    fisher_lambda_values = fisher_lambda_values(index);
    
    %{
    figure
    plot(1:10, fisher_lambda_values(1:10));
    xlabel('Fishervalue index');
    ylabel('Fishervalues');
    pause() 
    %}
    
    %only take SUBJECTS - 1 fisher vectors
    fisher_faces = fisher_all_faces(:, 1: (SUBJECTS-1) );

    fisher_training_projected  = fisher_faces' * images_eigen_space;

    %% Testing
    %required: 
    %mean in original space
    %eigenfaces and fisherfaces (eigen vector and fisher vectors)
    %projected images in the fisher face space (for comparison)

    testimagenumber = 0;
    correct = 0;
    for i=1:SUBJECTS
        for j=trainsize+1:length(person(i).faces)
            testimagenumber = testimagenumber + 1;
            testim = cell2mat(person(i).faces(j));
            %subtract mean image
            testimdouble = double(testim) - reshape(meanimage, row, col);
            
            testimvector = reshape(testimdouble, [], 1);

            fisher_im_projected = fisher_faces' * eigenfacestouse' * testimvector; % Test image feature vector

            distanceToTrain = zeros(totaltrain, 1);
            %find distance between this test image and training images
            for k=1:totaltrain
                distanceToTrain(k) = norm(fisher_training_projected(:,k) - fisher_im_projected, 2);
            end

            [tr index] = sort(distanceToTrain);
            idPersonPredicted = floor((index(1) - 1) / trainsize) + 1;
            if( i == idPersonPredicted)
                correct = correct + 1;
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
            %pause();
            %}
        end
    end
    final_total = final_total + testimagenumber;
    final_correct = final_correct + correct;
    disp(['correct % : ' num2str(correct/testimagenumber * 100)]);
    close all
end
disp(['  final accuracy % ' num2str(final_correct / final_total * 100)]);