clear

person = struct('faces', {}, 'features', {});
SUBJECTS = 20; %number of people
DATADIR = '/temp/cvproject/data/orl';
totaltrain = 0;
totalall = 0;
trainsize = 8;
teststart = trainsize+1;

keypoints = 0; %for average keyponit
%read data, store features for all (train/test) images
for i=1:SUBJECTS
    files =  dir(fullfile(DATADIR, ['s' num2str(i)], '*'));
    features = struct('descriptors', {}, 'locs', {});
    faces = {};
    count = 0;
    for j=1:size(files,1) %for all images
        if( ~ ( strcmp(files(j).name, '.') || strcmp(files(j).name, '..') || strcmp(files(j).isdir, '1') ) )
            count = count +1;
            totalall = totalall + 1;
            filename = fullfile(DATADIR, ['s' num2str(i)], files(j).name); 
            [image, descrips, locs] = sift(filename);            
            keypoints = keypoints + size(locs, 1);
            faces{count} = image;
            features(count).descriptors = descrips;
            features(count).locs = locs;
        end
    end
    %store this person's all faces
    person(i).faces = faces;
    person(i).features = features;
end
disp(['total all face images : ' num2str(totalall)]);
disp(['average keypoint: ' num2str(keypoints / totalall)]);

disp(['size person' num2str(size(person))]);
disp(['size person(1)' num2str(size(person(1)))]);

%%verifying
% for i=1:size(person,2)
%     x = cell2mat(person(i).faces(1));
%     imshow(x);
%     pause();
% end


%save everything in descriptor file
save descriptor