function mysift()
    clear;
    close all;
    
    %all variables used here will be available after loading the saved file
    %with descriptors
    %SUBJECTS = 20;
    load descriptor_ioe
    tic
    figure;
    subplot(1, 2, 1);
    %{
    figure
    %%almost similar pair of images
    imshow(cell2mat(person(6).faces(2)) - cell2mat(person(6).faces(7)));
    pause();
    imshow(cell2mat(person(5).faces(1)) - cell2mat(person(5).faces(5)));
    pause();
    imshow(cell2mat(person(5).faces(1)) - cell2mat(person(5).faces(5)));
    pause();
    %}
    distRatio = 0.8;
    final_overall_correct = 0;
    final_overall_total = 0;
    %for distRatio = 0.2:0.2:0.8
    for distRatio = 0.8:0.8
        final_cv_correct = 0;
        final_cv_total = 0;
        for folditer = [10]  
            tic
            fold = folditer;
            crossvalidation_correct_count = 0;
            crossvalidation_total_count = 0;
            fold_size = length(person(1).faces) / fold;
            train_size = (fold-1) * fold_size;
            all_size = train_size + fold_size;
            disp(['folds ' num2str(fold)]);
            for k=1:fold
                test_start = (k-1) * fold_size + 1;
                test_end = test_start + fold_size - 1;
                train_start1 = 1;
                train_end1 = test_start - 1;
                train_start2 = test_end + 1;
                train_end2 = length(person(1).faces);
                %disp(['teststart testend trainstart1 trainend1 trainstart2 trainend2 = ' num2str([test_start test_end train_start1 train_end1 train_start2 train_end2])]);
                total_count = 0;
                correct_count = 0;
                for i=1:SUBJECTS
                    for j= test_start : test_end
                        total_count = total_count + 1;
                        %compare test image's descriptors with training images'
                        image = cell2mat(person(i).faces(j));
                        descriptors = person(i).features(j).descriptors;
                        locs = person(i).features(j).locs;

                        match_count = zeros(SUBJECTS, train_size); %storing number of matches for all subjects and all images
                        for x=1:SUBJECTS
                            train1_count = 0;
                            for y=train_start1:train_end1 %train upper folds
                                current_train_image = cell2mat(person(x).faces(y));
                                train_descriptors = person(x).features(y).descriptors;
                                train_locs = person(x).features(y).locs;
                                match = computematch(image, current_train_image, descriptors, train_descriptors, locs, train_locs, distRatio);
                                train1_count = train1_count + 1;
                                match_count(x,train1_count) = match;
                            end
                        end
                        match_count;
                        for x=1:SUBJECTS
                            train2_count = 0;
                            for y=train_start2:train_end2 %train lower folds
                                current_train_image = cell2mat(person(x).faces(y));
                                train_descriptors = person(x).features(y).descriptors;
                                train_locs = person(x).features(y).locs;
                                match = computematch(image, current_train_image, descriptors, train_descriptors, locs, train_locs, distRatio);
                                train2_count = train2_count + 1;
                                match_count(x,train2_count) = match;
                            end
                        end
                        match_count;
                        index_total_match = sum(match_count, 2); %sum over all columns
                        [val ind] = sort(index_total_match, 'descend');
                        %display the best match
                        best_subject_id = ind(1);
                        [val ind] = sort(match_count( best_subject_id, :), 'descend');
                        %check the test fold and find the appropriate training
                        %image
                        if(ind(1) >= test_start)
                            best_image_id = ind(1) + fold_size;
                        else
                            best_image_id = ind(1);
                        end 
                        bestmatchimage = cell2mat(person(best_subject_id).faces(best_image_id));
                        %{
                        %show result
                        suptitle(['match count = ' num2str(val(1))]);
                        subplot(1,2,1);
                        imshow(image);
                        title(['test: subject id, index = ' num2str(i) ', ' num2str(j)]);
                        subplot(1,2,2);
                        imshow(bestmatchimage);
                        title(['train: subject id, index = ' num2str(best_subject_id) ', ' num2str(best_image_id)] );
                        pause();
                        %}
                        if (i == best_subject_id )
                            correct_count = correct_count + 1;
                        end
                    end
                end
                accuracy = 100 * correct_count / total_count;
                fprintf('\t\tfold: %d -> correct= %d , total = %d, accuracy = %f\n', k, correct_count, total_count, accuracy);
                crossvalidation_correct_count = crossvalidation_correct_count + correct_count;
                crossvalidation_total_count = crossvalidation_total_count + total_count;
            end
            cv_accuracy = 100 * crossvalidation_correct_count / crossvalidation_total_count;
            fprintf('\tcvcorrect = %d, cvtotal = %d, cvaccuracy = %f\n', crossvalidation_correct_count, crossvalidation_total_count, cv_accuracy);
            final_cv_correct = final_cv_correct + crossvalidation_correct_count;
            final_cv_total = final_cv_total + crossvalidation_total_count;
        end
        final_cv_accuracy = 100 * final_cv_correct / final_cv_total;
        disp(['distRatio = ' num2str(distRatio)]);
        fprintf('final cv correct = %d, total = %d, accuracy = %f\n', final_cv_correct, final_cv_total, final_cv_accuracy);
        final_overall_correct = final_overall_correct + final_cv_correct;
        final_overall_total = final_overall_total + final_cv_total;
    end
    final_overall_accuracy = 100 * final_overall_correct / final_overall_total;
    fprintf('final overall correct = %d, total = %d, accuracy = %f\n', final_overall_correct, final_overall_total, final_overall_accuracy);
    toc
end

function [matches] = computematch(test_image, train_image, desc1, desc2, loc1, loc2, distRatio)
    [r c] = size(test_image);
    maxdist = sqrt(r^2 + c^2);
    desc2T = desc2';
    for k = 1:size(desc1, 1)
        %dist = loc1(k)
       dotprods = desc1(k,:) * desc2T;
       [vals,indx] = sort(acos(dotprods));  % Take inverse cosine and sort results
       % Check if nearest neighbor has angle less than distRatio times 2nd.
       %vals(1)/vals(2)
       match(k) = 0;
       if (vals(1) < distRatio * vals(2))
          %if distance within some radius, then only match
          match(k) = indx(1);
       end
    end
    matches = sum(match > 0);
    % Create a new image showing the two images side by side.
    im3 = appendimages(test_image,train_image);
    
    % Show a figure with lines joining the accepted matches.
    %{
    if (matches > 0)
        figure('Position', [100 100 size(im3,2) size(im3,1)]);
        colormap('gray');
        imagesc(im3);
        hold on;
        cols1 = size(test_image,2);
        for i = 1: size(desc1,1)
          if (match(i) > 0)
            line([loc1(i,2) loc2(match(i),2)+cols1], ...
                 [loc1(i,1) loc2(match(i),1)], 'Color', 'g');
          end
        end
        hold off;
        pause();
        close all;
        num = sum(match > 0);
        fprintf('Found %d matches.\n', num);
    end
    %}
end