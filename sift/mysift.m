function mysift()
    clear;
    close all;
    
    %all variables used here will be available after loading the saved file
    %with descriptors
    
    load descriptor
    figure;
    subplot(1, 2, 1);
    total_count = 0;
    correct_count = 0;
    for i=1:SUBJECTS
        for j= teststart : length(person(i).faces)
            total_count = total_count + 1;
            %compare test image's descriptors with training images'
            image = cell2mat(person(i).faces(j));
            descriptors = person(i).features(j).descriptors;
            locs = person(i).features(j).locs;
            
            match_count = zeros(SUBJECTS, trainsize); %storing number of matches for all subjects and all images
            for x=1:SUBJECTS
                for y=1:trainsize
                    train_descriptors = person(x).features(y).descriptors;
                    train_locs = person(x).features(y).locs;
                    match = computematch(descriptors, train_descriptors, locs, train_locs);
                    match_count(x,y) = match;
                end
            end
            match_count
            index_total_match = sum(match_count, 2); %sum over all columns
            [val ind] = sort(index_total_match, 'descend');
            %display the best match
            best_subject_id = ind(1);
            [val ind] = sort(match_count( best_subject_id, :), 'descend');
            best_image_id = ind(1);
            bestmatchimage = cell2mat(person(best_subject_id).faces(best_image_id));
            
            %show result
            %subplot(1, 2, 1);
            %imshow(image);
            %subplot(1,2,2);
            %imshow(bestmatchimage);
            %pause();
            
            if (i == best_subject_id )
                correct_count = correct_count + 1;
            end
        end
    end
    
    accuracy = 100 * (correct_count) / total_count;
    disp(['accuracy = ' num2str(accuracy)]);
end

function [matches] = computematch(desc1, desc2, locs1, locs2)
    distRatio = 0.4;
    matches = 0;
    desc2T = desc2';
    for k = 1:size(desc1, 1)
       dotprods = desc1(k,:) * desc2T;
       [vals,indx] = sort(acos(dotprods));  % Take inverse cosine and sort results
       % Check if nearest neighbor has angle less than distRatio times 2nd.
       if (vals(1) < distRatio * vals(2))
          match(k) = indx(1);
       else
          match(k) = 0;
       end
    end
    matches = sum(match > 0);
end