function [sift_coordinates, sift_descriptor] = sift_descriptors(peaks, image, octaves, sigma)
im = image;

%% create Guassian
[height,width] = size(im);
k = 1.6;
DoG_scale = zeros(height, width, octaves);
G_scale = zeros(height, width, octaves+1);
G_scale(:,:,1) = im;
for s = 1:octaves
    sigma1 = (s)*sigma;
    sigma2 = sigma1*k;
    %radii(s) = sigma1*2;
    kn_size = round(sigma2*6);
    G1 = gaussian_filter(sigma1, kn_size, kn_size);
    G2 = gaussian_filter(sigma2, kn_size, kn_size);
    blur1 = (fourier_transform_convolution(im, G1));
    blur2 = (fourier_transform_convolution(im, G2));
    DoG_scale(:,:,s) = blur1 - blur2;
    G_scale(:,:,s+1) = blur1;
end


if 0
    figure();
    for i = 1:octaves+1
       subplot(ceil(octaves/2),2,i);
       imshow(NormaliseImage(G_scale(:,:,i)));
    end
end
%% orientation assignment
% pre calculate magnitudes and orientations
[height, width, octaves] = size(G_scale);
mag = zeros(height, width, octaves);
orientation = zeros(height, width, octaves);
for y = 2:height-1
    for x = 2:width-1
        for o = 1:octaves
            %calculate magintude and orientation at pixel [x,y,s].
            mag(y,x,o) = sqrt((G_scale(y+1,x,o)-G_scale(y-1,x,o)).^2 + ...
                (G_scale(y,x+1,o)-G_scale(y,x-1,o)).^2);
            orientation(y,x,o) = atand((G_scale(y+1,x,o)-G_scale(y-1,x,o)) / ...
                (G_scale(y,x+1,o)-G_scale(y,x-1,o)));
            %convert to degrees from 0 to 360
            if(G_scale(y+1,x,o)-G_scale(y-1,x,o) < 0)
                orientation(y,x,o) = orientation(y,x,o) + 180;
            end
            if(orientation(y,x,o) < 0)
                orientation(y,x,o) = orientation(y,x,o) + 360;
            end
        end
    end
    orientation(isnan(orientation))=0;%NaN from dividing by 0 - change to 0 (no magnitude)
end

nh_size = 16;
orientation_peaks=[];
%create histograms for all keypoints in octave j
[num_of_peaks,~] = size(peaks);
for kp = 1:num_of_peaks
    %round to nearest pixel value (non integer from
    %localisation)for x, y and sigma
    kp_y = round(peaks(kp,1));
    kp_x = round(peaks(kp,2));
    kp_o = round(peaks(kp,3));

    mn_x = kp_x - nh_size/2 +1;
    mx_x = kp_x + nh_size/2;
    mn_y = kp_y - nh_size/2 +1;
    mx_y = kp_y + nh_size/2;
    if((mn_x > 1) && (mx_x < width) && (mn_y > 1) && (mx_y < height))%don't run if keypoint overlaps edge
        %extract the relevant window of magnitude values
        mag_window = mag(mn_y:mx_y, mn_x:mx_x, kp_o);
        g = gaussian_filter(1.5*kp_o, nh_size, nh_size);%gaussian kernel
        g = fspecial('gaussian',[nh_size nh_size],1.5*kp_o);
        guassfilt_mag_window = mag_window.*g;%gaussian weight magnitudes
        %extract the relevant window of orientation values
        ori_window = orientation(mn_y:mx_y, mn_x:mx_x, kp_o);
        %sort orientations into 36 bins (1 to 36)
        bin_window = ceil(ori_window/10);

        if(max(max(bin_window))>0)%no orientation/magnitudes -> don't run
            bin = zeros(36,1);
            for iy = 1:nh_size
                for ix = 1:nh_size
                    if(bin_window(iy,ix)>0)%no orientation/magnitude -> don't run
                        %for each bin, sum the magnitudes of pixels
                        %that fall into that bin
                        bin(bin_window(iy,ix)) = bin(bin_window(iy,ix)) + ...
                            guassfilt_mag_window(iy,ix);
                    end
                end
            end
            %for plotting. NOTE: use breakpoints if plotting - will crash if
            %you try to plot ever single keypoints histogram.
            if 0
                figure();
                bar(0:10:350,bin);
                title(['y = ',num2str(peaks(kp,1)),' x = ',num2str(peaks(kp,2)),' sig = ',num2str(peaks(kp,3))]);
            end
            %add new peak descriptor
            max_bin = max(bin);
            for b = 1:length(bin)
                if(bin(b)>0.8*max_bin)
                    %create a new peaks for orientations with at
                    %least 80% of the strongest orientation
                    orientation_peaks(end+1,:) = [peaks(kp,:),b*10];
                end
            end
        end
    end
end

%% keypoint descriptor
descriptor = [];
coordinates = [];
[num_of_orientation_peaks,~] = size(orientation_peaks);
for kp = 1:num_of_orientation_peaks
    %round to nearest pixel value (non integer from
    %localisation)for x, y and sigma
    kp_y=round(orientation_peaks(kp,1));
    kp_x=round(orientation_peaks(kp,2));
    kp_o= round(orientation_peaks(kp,3));
    peak_orientation=orientation_peaks(kp,5);

    rotation_matrix = [cosd(peak_orientation), sind(peak_orientation);...
        -sind(peak_orientation), cosd(peak_orientation)];
    %extract the relevant window of magnitude values
    mag_window = zeros(16,16);
    ori_window = zeros(16,16);
    for ry = 1:16
        for rx = 1:16
            idx = round([rx-8, ry-8]*rotation_matrix) + [kp_x, kp_y];
            mag_window(ry, rx) = mag(min(max(1,idx(2)),height), min(max(1,idx(1)),width), kp_o);
            ori_window(ry, rx) = mod((orientation(min(max(1,idx(2)),height), min(max(1,idx(1)),width), kp_o) - peak_orientation), 360);
        end
    end
    %mag_window = mag(mn_y:mx_y, mn_x:mx_x, kp_o);
    g = gaussian_filter(0.5*nh_size, nh_size, nh_size);%gaussian kernel
    g = fspecial('gaussian',[nh_size nh_size],0.5*nh_size);
    g_mag_window = mag_window.*g;%weight magnitudes with gaussian
    %extract the relevant window of orientation values and rotates by peaks orientation
    %ori_window = mod((orientation(mn_y:mx_y, mn_x:mx_x, kp_o) - peak_orientation), 360); 
    %fits orientaions into 8 bins
    bin_window = ceil(ori_window/45);
    %splits window into 4x4 grid and calculated descriptors for
    %each keypoint
    kp_descriptor = [];

    for wy = 1:4
        for wx = 1:4
            %current sub window indexes
            y_wind_idx = (wy-1)*4+1:(wy)*4;
            x_wind_idx = (wx-1)*4+1:(wx)*4;
            bin = zeros(8,1);
            for ix = 1:4
                for iy = 1:4
                    if(bin_window(y_wind_idx(iy), x_wind_idx(ix)) > 0)
                        %for each bin, sum the magnitudes of pixels that fall into that bin
                        bin(bin_window(y_wind_idx(iy), x_wind_idx(ix))) = ...
                            bin(bin_window(y_wind_idx(iy), x_wind_idx(ix))) + ...
                            g_mag_window(y_wind_idx(iy), x_wind_idx(ix));
                    end
                end
            end
            %add to the descriptor for the keypoint kp_count
            kp_descriptor = [kp_descriptor; bin];
        end
    end
    %each descriptor in the cell array is a 128 size (4x4x8) vector
    kp_descriptor = NormaliseImage(kp_descriptor);%normalise
    kp_descriptor(kp_descriptor<0.2)=0.2;%theshold (SIFTs attempt at illumination invariance)
    kp_descriptor = NormaliseImage(kp_descriptor);  %normalise 
    descriptor(kp,:) = kp_descriptor;
    coordinates(kp,:) = [kp_y,kp_x];
end
sift_descriptor = descriptor;
sift_coordinates = coordinates;
end