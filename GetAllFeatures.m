function [Features] = GetAllFeatures(numberOfImages, Directory, names, net, Layers, lengthOfFeats)
    
    D = parallel.pool.DataQueue;
    startTic = tic;
    h = waitbar(0, 'Please wait ... ');
    afterEach(D, @nUpdateWaitbar);

    N = numberOfImages;
    p = 1;
    
    Features = zeros(numberOfImages, lengthOfFeats);
    

    parfor i=1:numberOfImages
        if(mod(i,1000)==0)
            disp(i);
        end
        img           = imread( strcat(Directory, filesep, names{i}) );
        Features(i,:) = GetFeatures(img, net, Layers);
        send(D, i);
    end
    
    function nUpdateWaitbar(~)
        msg = [num2str(p) '/' num2str(N) ' items in ' num2str(round(toc(startTic))) 's.'];
        waitbar(p/N, h, msg);
        p = p + 1;
    end
end