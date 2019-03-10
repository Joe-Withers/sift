function result = DoG_blob_detector(im, scale_size, sigma, octaves)

all_peaks = [];
for octave = 1:octaves
    %% create Difference of Guassian
    im = imresize(im,1/(2^(octave-1)));
    [height,width] = size(im);
    k = 1.6;
    DoG_scale = zeros(height, width, scale_size);
    radii = [];
    figure();
    for s = 1:scale_size
        sigma1 = (s)*sigma;
        sigma2 = sigma1*k;
        radii(s) = sigma1*2;
        kn_size = round(sigma2*6);
        G1 = gaussian_filter(sigma1, kn_size, kn_size);
        G2 = gaussian_filter(sigma2, kn_size, kn_size);
        if 0
            kn_L = G1 - G2;subplot(2,3,s);
            imagesc(kn_L);title("Scale = "+s);
%             r = radii(s); th = 0:pi/50:2*pi;
%             xunit = r * cos(th) + kn_size/2;
%             yunit = r * sin(th) + kn_size/2;
%             plot(xunit, yunit);
        end
        blur1 = (fourier_transform_convolution(im, G1));
        blur2 = (fourier_transform_convolution(im, G2));
        DoG_scale(:,:,s) = blur1 - blur2;
    end

    if 0
        figure();
        for i = 1:scale_size
           subplot(ceil(scale_size/2),2,i);
           imshow(NormaliseImage(imresize(DoG_scale(:,:,i),2)));
        end
    end

    %% find local maxima
    extremas = [];
    for y = 2:height-1
        for x = 2:width-1
            for o = 2:scale_size-1
                %find max and min of surrounding 26 pixels
                surrounding_max = ...
                    max([DoG_scale(y-1,x-1,o-1),DoG_scale(y-1,x,o-1),DoG_scale(y-1,x+1,o-1),...
                    DoG_scale(y,x-1,o-1),DoG_scale(y,x,o-1),DoG_scale(y,x+1,o-1),...
                    DoG_scale(y+1,x-1,o-1),DoG_scale(y+1,x,o-1),DoG_scale(y+1,x+1,o-1),...
                    ...
                    DoG_scale(y-1,x-1,o),DoG_scale(y-1,x,o),DoG_scale(y-1,x+1,o),...
                    DoG_scale(y,x-1,o),                 DoG_scale(y,x+1,o),...
                    DoG_scale(y+1,x-1,o),DoG_scale(y+1,x,o),DoG_scale(y+1,x+1,o),...
                    ...
                    DoG_scale(y-1,x-1,o+1),DoG_scale(y-1,x,o+1),DoG_scale(y-1,x+1,o+1),...
                    DoG_scale(y,x-1,o+1),DoG_scale(y,x,o+1),DoG_scale(y,x+1,o+1),...
                    DoG_scale(y+1,x-1,o+1),DoG_scale(y+1,x,o+1),DoG_scale(y+1,x+1,o+1)]);

                surrounding_min = ...
                    min([DoG_scale(y-1,x-1,o-1),DoG_scale(y-1,x,o-1),DoG_scale(y-1,x+1,o-1),...
                    DoG_scale(y,x-1,o-1),DoG_scale(y,x,o-1),DoG_scale(y,x+1,o-1),...
                    DoG_scale(y+1,x-1,o-1),DoG_scale(y+1,x,o-1),DoG_scale(y+1,x+1,o-1),...
                    ...
                    DoG_scale(y-1,x-1,o),DoG_scale(y-1,x,o),DoG_scale(y-1,x+1,o),...
                    DoG_scale(y,x-1,o),                 DoG_scale(y,x+1,o),...
                    DoG_scale(y+1,x-1,o),DoG_scale(y+1,x,o),DoG_scale(y+1,x+1,o),...
                    ...
                    DoG_scale(y-1,x-1,o+1),DoG_scale(y-1,x,o+1),DoG_scale(y-1,x+1,o+1),...
                    DoG_scale(y,x-1,o+1),DoG_scale(y,x,o+1),DoG_scale(y,x+1,o+1),...
                    DoG_scale(y+1,x-1,o+1),DoG_scale(y+1,x,o+1),DoG_scale(y+1,x+1,o+1)]);

                if((DoG_scale(y,x,o) > surrounding_max)||(DoG_scale(y,x,o) < surrounding_min))
                    extremas(end+1,:) = [y,x,o];
                end
            end
        end
    end

    %% keypoint localisation
    peaks = [];
    [keypoints,~] = size(extremas);
    for k = 1:keypoints
        y = extremas(k,1);
        x = extremas(k,2);
        o = extremas(k,3);

        % find first derivative in each dimension (i.e. [dx; dy; dsigma])
        kn = [-1 0 1]/2;
        DoG_y_window = [0 0 0];
        DoG_x_window = [0 0 0];
        DoG_o_window = [0 0 0];
        DoG_y_window(:) = DoG_scale(y-1:y+1, x, o);
        DoG_x_window(:) = DoG_scale(y, x-1:x+1, o);
        DoG_o_window(:) = DoG_scale(y, x, o-1:o+1);
        d_y = (sum(kn .* DoG_y_window));
        d_x = (sum(kn .* DoG_x_window));
        d_o = (sum(kn .* DoG_o_window));

        first_derivative = [d_y; d_x; d_o];

        % find second derivatives
        DoG_window = DoG_scale(y-1:y+1, x-1:x+1, o-1:o+1);
        % calculate first derivative kernels (3x3x3 - 3-D)

        % calculate the second derivatives
        d_yy = ((DoG_scale(y,x,o) - DoG_scale(max(1,y-2),x,o))/2 - (DoG_scale(min(height,y+2),x,o) - DoG_scale(y,x,o))/2)/2;
        d_yx = ((DoG_scale(y+1,x+1,o) - DoG_scale(y-1,x+1,o))/2 - (DoG_scale(y+1,x-1,o) - DoG_scale(y-1,x-1,o))/2)/2;
        d_yo = ((DoG_scale(y+1,x,o+1) - DoG_scale(y-1,x+1,o+1))/2 - (DoG_scale(y+1,x,o-1) - DoG_scale(y-1,x,o-1))/2)/2;

        d_xy = ((DoG_scale(y+1,x+1,o) - DoG_scale(y+1,x-1,o))/2 - (DoG_scale(y-1,x+1,o) - DoG_scale(y-1,x-1,o))/2)/2;
        d_xx = ((DoG_scale(y,x,o) - DoG_scale(y,max(1,x-2),o))/2 - (DoG_scale(y,min(width,x+2),o) - DoG_scale(y,x,o))/2)/2;
        d_xo = ((DoG_scale(y,x+1,o+1) - DoG_scale(y,x-1,o+1))/2 - (DoG_scale(y,x+1,o-1) - DoG_scale(y,x-1,o-1))/2)/2;

        d_oy = ((DoG_scale(y+1,x,o+1) - DoG_scale(y+1,x,o-1))/2 - (DoG_scale(y-1,x,o+1) - DoG_scale(y-1,x,o-1))/2)/2;
        d_ox = ((DoG_scale(y,x+1,o+1) - DoG_scale(y,x+1,o-1))/2 - (DoG_scale(y,x-1,o+1) - DoG_scale(y,x-1,o-1))/2)/2;
        d_oo = ((DoG_scale(y,x,o) - DoG_scale(y,x,max(1,o-2)))/2 - (DoG_scale(y,x,min(scale_size,o+2)) - DoG_scale(y,x,o))/2)/2;

        %3x3 Hessian
        second_derivative = [d_yy, d_yx, d_yo;...
            d_xy, d_xx, d_xo;...
            d_oy, d_ox, d_oo];
        % calculate offset
        offset = (-inv(second_derivative))*first_derivative;
        %if there isn't another pixel closer to the localised peak
        if(max(abs(offset)) < 0.5)
            peak = extremas(k,:) + offset';
            peak_value = DoG_scale(y,x,o) + 0.5*(first_derivative'*offset);
            %threshold
            if(abs(peak_value) > 0.03)
                %eliminate edge responses
                tr = d_xx + d_yy;%trace
                det = d_xx.*d_yy - d_xy.^2;%determinant

                r = 10;
                edge = (tr.^2 ./ det) > ((r+1)^2 / r);
                if ~edge
                    peaks(end+1,:) = [peak, peak_value, radii(round(peak(3))), octave];
                end
            end

        end
    end

    all_peaks = [all_peaks; peaks];
end
result = all_peaks;
end