DATADIR = '/temp/cvproject/lfw2/';
OUTDIR = '/temp/lfw_selected/';

faceDetector = vision.CascadeObjectDetector;

folders =  dir(fullfile(DATADIR));
mkdir(OUTDIR);
for j=1:size(folders,1)
    if( ~ ( strcmp(folders(j).name, '.') || strcmp(folders(j).name, '..') || strcmp(folders(j).isdir, '1') ) )
        person_dir = fullfile(DATADIR, folders(j).name);
        temp_str = strtrim(regexp(person_dir, '\/', 'split'));
        person_dir_only =  temp_str{ end };
        person_files = dir([person_dir '/*.jpg']);
        file_count = size(person_files,1);
        if (file_count >= 25  && file_count <=40)
            mkdir(fullfile(OUTDIR, person_dir_only));
            for k=1:file_count
                filename = person_files(k).name;
                temp_str = strtrim(regexp(filename, '\.', 'split'));
                filename_nosuf = temp_str{1};
                im = imread([person_dir '/' filename]);
                %detect
                bbox = step(faceDetector, im);
                %imshow(im);
                %pause();
                for c=1:size(bbox, 1)
                    cropped_face = imcrop(im, bbox(c,:));
                    scaled_face = imresize(cropped_face, [120, 120]);
                    outfilename = fullfile(OUTDIR, person_dir_only, [filename_nosuf '.pgm']);
                    imwrite(scaled_face, outfilename, 'pgm');
                end 
            end
        end
    end
end