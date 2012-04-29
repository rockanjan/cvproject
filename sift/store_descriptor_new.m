clear

person = struct('faces', {}, 'features', {});

%DATADIR = '/temp/CroppedYaleSelected';
%DATADIR = '/temp/lfw_selected';
DATA_DIR = '/temp/cvproject/data/orl40';

totaltrain = 0;
totalall = 0;

keypoints = 0; %for average keyponit
%read data, store features for all (train/test) images

dir_list = dir(fullfile(DATA_DIR));
total_subjects = 0;
for i=1:size(dir(fullfile(DATA_DIR)), 1)
    person_dir = dir_list(i).name
    if( ~ ( strcmp(person_dir, '.') || strcmp(person_dir, '..') ) )
        total_subjects = total_subjects + 1;
        files =  dir(fullfile(DATA_DIR, person_dir, '*.pgm'));
        features = struct('descriptors', {}, 'locs', {});
        faces = {};
        count = 0;
        for j=1:size(files,1) %for all images
            if( ~ ( strcmp(files(j).name, '.') || strcmp(files(j).name, '..') || strcmp(files(j).isdir, '1') ) )
                count = count +1;
                totalall = totalall + 1;
                filename = fullfile(DATA_DIR, person_dir, files(j).name); 
                %{
                %histogram
                tmpimage = imread(filename);
                tmpimage = histeq(tmpimage);
                tmpfile = '/tmp/tmpfile.pgm';
                imwrite(tmpimage, tmpfile, 'pgm');
                filename=tmpfile;
                %histogram done
                %}
                [image, descrips, locs] = sift(filename);            
                keypoints = keypoints + size(locs, 1);
                faces{count} = image;
                features(count).descriptors = descrips;
                features(count).locs = locs;
            end
        end
        person(total_subjects).faces = faces;
        person(total_subjects).features = features;
    end
end
disp(['total all face images : ' num2str(totalall)]);
disp(['average keypoint: ' num2str(keypoints / totalall)]);

disp(['size person' num2str(size(person))]);
disp(['size person(1).faces ' num2str(size(person(1).faces))]);

SUBJECTS = size(person,2); %number of people
clear DATA_DIR
%save everything in descriptor file
save orl_40
%{
%verify
for i=1:SUBJECTS
     for j=1:size(person(i).faces, 2)
         imshow(cell2mat(person(i).faces(j)));
         pause();
     end
end

SUBJECTS = size(person,2);
%}