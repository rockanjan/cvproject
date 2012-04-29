clear;
close all;

person = struct('faces', {});

%DATADIR = '/temp/cvproject/data/orl';
%DATADIR = '/temp/lfw_selected';
%DATADIR = '/temp/CroppedYaleSelected';
DATADIR = '/temp/cvproject/data/orl40';
%DATADIR = '/temp/cvproject/ioepgm';
%DATADIR = '/temp/CroppedYaleSelected';
totaltrain = 0;
totalall = 0;
row = 0;
col = 0;

dir_list = dir(fullfile(DATADIR));
total_subjects = 0;
for i=1:size(dir(fullfile(DATADIR)), 1)
    person_dir = dir_list(i).name
    if( ~ ( strcmp(person_dir, '.') || strcmp(person_dir, '..') ) )
        total_subjects = total_subjects + 1;
        files =  dir(fullfile(DATADIR, person_dir, '*.pgm'));
        features = struct('descriptors', {}, 'locs', {});
        faces = {};
        count = 0;
        for j=1:size(files,1) %for all images
            if( ~ ( strcmp(files(j).name, '.') || strcmp(files(j).name, '..') || strcmp(files(j).isdir, '1') ) )
                count = count +1;
                totalall = totalall + 1;
                filename = fullfile(DATADIR, person_dir, files(j).name); 
                image = imread(filename);
                
                %histogram equalize
                %[row col] = size(image);
                %image = imresize(image, [200 200]); row=200; col=200;
                image = imresize(image, [192 168]); row=192; col=168;
                %image = histeq(image);
                faces{count} = image;
            end
        end
        count
        person(total_subjects).faces = faces;
    end
end
disp(['row col = ' num2str(row) ' ' num2str(col) ]);
disp(['total all face images : ' num2str(totalall)]);

SUBJECTS = size(person,2);
image_per_subject = count;
save orl_40

%verify
for i=1:SUBJECTS
     for j=1:size(person(i).faces, 2)
         imshow(cell2mat(person(i).faces(j)));
         pause();
     end
end