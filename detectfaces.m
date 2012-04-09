function detectface(im)
    im = imread('/temp/cvproject/data/YaleBExtended/yaleB11/yaleB11_P00A-005E-10.pgm');
    faceDetector = vision.CascadeObjectDetector;
    bbox = step(faceDetector, im);
    shapeInserter = vision.ShapeInserter;
    i_faces = step(shapeInserter, im, int32(bbox));
    imshow(i_faces);
end