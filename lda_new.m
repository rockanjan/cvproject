function lda_new()
    addpath('cvpr');

    % performs eigenface based recognition
    clear;
    close all;

    %load yale_histeq
    load orl_40
    %load person_orl_40
    trainsize = 9;
    final_correct = 0;
    final_total = 0;
    for iter = 1:10
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
        X = []; 
        totaltrain=0;
        for i=1:SUBJECTS
            for j=1:trainsize
                face = cell2mat(person(i).faces(j));
                facecol = reshape(face, [], 1);
                X = [X facecol];
                totaltrain = totaltrain + 1;
                C(totaltrain) = i; %required by cvLda
            end
        end
        %
        X = double(X);

        %call lda
        Wopt = cvLda(X, C);

        Xmean = mean(X, 2);
        %project training images into lda space
        train_projected = Wopt' * (X - repmat(Xmean, 1, totaltrain));
        %train_projected = Wopt' * X;

        %test
        testimagenumber = 0;
        correct = 0;
        for i=1:SUBJECTS
            for j=trainsize+1:length(person(i).faces)
                testimagenumber = testimagenumber + 1;
                testim = cell2mat(person(i).faces(j));
                %occlude
                testim = occlude(testim);
                
                %subtract mean image
                testimcol = double(testim(:)) - Xmean;
                %testimcol = double(testim(:));

                test_projected = Wopt' * testimcol; % Test image feature vector
                %fisher_im_projected'
                distanceToTrain = zeros(totaltrain, 1);
                %find distance between this test image and training images
                for k=1:totaltrain
                    distanceToTrain(k) = norm(train_projected(:,k) - test_projected, 2);
                    %distanceToTrain(k) = acos(dot(fisher_training_projected(:,k), fisher_im_projected)/ (norm(fisher_training_projected(:,k), 2) * norm(fisher_im_projected, 2)));
                end
                [tr index] = sort(distanceToTrain);
                %disp('distance');
                %tr'
                %idPersonPredicted = floor((index(1) - 1) / trainsize) + 1;
                idPersonPredicted = ceil(index(1) / trainsize);
                %disp(['matched image index ' num2str(index(1))]);
                %disp(['matched subject id ' num2str(idPersonPredicted)]); 
                if( i == idPersonPredicted)
                    correct = correct + 1;
                end
                %display
                %{
                subplot(1,4,1);
                imshow(testim)
                title('test image');
                firstmatch = uint8(reshape(X(:,index(1)), row, col));
                secondmatch = uint8(reshape(X(:,index(2)), row, col));
                thirdmatch = uint8(reshape(X(:,index(3)), row, col));
                subplot(1,4,2);
                imshow(firstmatch);
                title('first match');
                subplot(1,4,3);
                imshow(secondmatch);
                title('second match');
                subplot(1,4,4);
                imshow(thirdmatch);
                title('third match');
                if(i == idPersonPredicted)
                    suptitle('correct');
                else
                    suptitle('error !!!');
                end
                pause();
                %}
                
            end
        end
        final_total = final_total + testimagenumber;
        final_correct = final_correct + correct;
        disp(['correct % : ' num2str(correct/testimagenumber * 100)]);
        close all
    end
    disp(['  final accuracy % ' num2str(final_correct / final_total * 100)]);
end

function [image] = occlude(image)
    [row col] = size(image);
    %add some occlusion
    occlusion_percent = 0.1;
    occlusion_size = floor(row*occlusion_percent);
    low=1;
    high=floor(min(row,col) - occlusion_size);
    x = floor(low + (high - low) * rand); %position to keep the black dot
    y = floor(low + (high - low) * rand);
    for k=x:x+occlusion_size
        for l=y:y+occlusion_size
            image(k,l) = 0;
        end
    end
end