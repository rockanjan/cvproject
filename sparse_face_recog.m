function sparse_face_recog()
    clear all;
    %load person_yale_30sub_40images_histeq
    load orl_40
    [row col] = size(cell2mat(person(1).faces(1)));
    %r = 8;
    final_total = 0;
    final_correct = 0;
    for iter=1:4
        r=12;
        c = floor((col/row) * r);
        disp(['image size : ' num2str(r) 'x' num2str(c)]);
        trainsize = 8;
        %randomize the images
        for i=1:SUBJECTS
            faces = person(i).faces;
            randvalues = rand(1, length(faces));
            [rval rind] = sort(randvalues);
            person(i).faces = faces(rind);
        end

        %append the images in columns of a matrix A
        A = []; 
        totaltrain = 0;
        for i=1:SUBJECTS
            for j=1:trainsize
                face = cell2mat(person(i).faces(j));
                face = imresize(face, [r c]);
                %imshow(face);
                facecol = face(:);
                A = [A facecol];
                totaltrain = totaltrain + 1;
            end
        end
        %testing
        total = 0;
        correct = 0;
        for k=1:SUBJECTS
            for j=trainsize+1:image_per_subject
                %test image
                y = cell2mat(person(k).faces(j));
                test_image = occlude(y);
                y = imresize(y, [r c]);
                y = y(:);
                y = double(y);
                A = double(A);

                n = size(A,2);
                f=ones(2*n,1);
                Aeq=[A -A];
                lb=zeros(2*n,1); %lower bounds

                %{
                X = linprog(f,A,b,Aeq,beq,LB,UB,X0,OPTIONS)

                min f'*x    subject to:   A*x <= b 
                              x
                in addition to Aeq*x = beq, and bounds of x are LB and UB. 
                %}
                %% optimization
                x1 = linprog(f,[],[],Aeq,y,lb,[],[],[]);
                %%
                x1 = x1(1:n)-x1(n+1:2*n);
                %disp([ ' less than 0.2 ' num2str(length( find(x1 < 0.2) )) ' more than 0.2 ' num2str(length(find(x1 > 0.2)))]);
                nn = ones(1, SUBJECTS) * trainsize;
                nn = cumsum(nn);

                tmp_var = 0;

                %finding residuals
                for i = 1:SUBJECTS
                    delta_xi = zeros(length(x1),1);
                    if i == 1
                        delta_xi(1:nn(i)) = x1(1:nn(i));
                    else
                        tmp_var = tmp_var + nn(i-1);
                        begs = nn(i-1)+1;
                        ends = nn(i);
                        delta_xi(begs:ends) = x1(begs:ends);
                    end
                    %delta_xi(1:trainsize)'
                    %delta_xi(end-trainsize: end
                    tmp(i) = norm(y-A*delta_xi,2);
                    tmp1(i) = norm(delta_xi,1)/norm(x1,1);
                end
                clss = find(tmp==min(tmp));
                
                if(clss == k)
                    correct = correct + 1;
                else
                    disp('error!!!');
                end
                total = total + 1;
                
                % To display error
                %{
                bar(tmp);
                pause();
                %}

                
                subplot(1,2,1);
                imshow(test_image);
                title(['test image class : ' num2str(k)] );
                subplot(1,2,2);
                imshow(cell2mat(person(clss).faces(1)));
                title(['matched image class : ' num2str(clss) ]);
                if(clss == k)
                    suptitle('correct :) ');
                else
                    suptitle('error !!!');
                end
                pause(3);
            end
        end
        final_total = final_total + total;
        final_correct = final_correct + correct;
        disp(['correct total accuracy ' num2str(correct) ' ' num2str(total) ' ' num2str(100 * correct / total)]);
    end
    disp(['   final total accuracy ' num2str(100 * final_correct / final_total ) ]);
end

function [image] = occlude(image)
    [row col] = size(image);
    %add some occlusion
    occlusion_percent = 0.4;
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