classdef image_match < handle
    % This is a matlab class which performs SIFT and pose estimation of 
    % images. This is a just a fun side project I did to learn the SIFT 
    % algorithm and is far from a fast/robust/well-tested implementation.
    
    properties(SetAccess = private)
        hp % "hyper parameters"        
        
        array                    
        gauss_pyramid    
        DoG_pyramid         
        grad_pyramid        
        
        maxima_pyramid   
        feature_points
        descriptors
    end
   
    methods(Access = public)
        function obj = image_match(array)            
            % Store array
            obj.array = array;
            
            % Store "hyper parameters" here
            obj.hp.s = 2;
            obj.hp.num_octaves = 4;         
            obj.hp.contrast_cutoff = 0.03;
            obj.hp.r_cutoff = 10;
            obj.hp.M_eig_ratio_cutoff = 15;
            obj.hp.num_orientation_bins = 36;
            obj.hp.num_descriptor_parts = 4;
            obj.hp.sift_window = 4*obj.hp.num_descriptor_parts;
            obj.hp.num_descriptor_bins = 8;
            obj.hp.RANSAC_it = 5000;
            obj.hp.RANSAC_delta = 5;
            
            % Set array
            % First, convert RGB to grayscale
            if size(obj.array,3) == 3
                obj.array = rgb2gray(im2double(obj.array));
            else
                obj.array = im2double(obj.array);
            end           
            
            % Normalize it
            obj.array = (obj.array - min(obj.array(:)))./(max(obj.array(:)) - min(obj.array(:)));
            
            % Get gaussian pyramid
            k = 2^(1/obj.hp.s);
            num_octaves = obj.hp.num_octaves;
            num_scales = obj.hp.s+3;
            obj.gauss_pyramid = {};
            for i = 1:num_octaves              
                for j = 1:num_scales
                    if i == 1 && j == 1
                        % First scale of first octave; initialize
                        obj.gauss_pyramid{i} = image_match.array_gauss(obj.array,k^-1); 
                    elseif i > 1 && j == 1
                        % This is the first scale after the first octave. 
                        % Take the 2nd from the top image and down sample 
                        % by 2
                        obj.gauss_pyramid{i} = obj.gauss_pyramid{i-1}(1:2:end,1:2:end,end-2);
                    else
                        % These are scales after the first, sigma must be:
                        sigma = sqrt(k^(2*(j-2))-k^(2*(j-2)-2));
                        obj.gauss_pyramid{i} = cat(3, ...
                                                   obj.gauss_pyramid{i}, ...
                                                   image_match.array_gauss(obj.gauss_pyramid{i}(:,:,j-1),sigma));
                    end         
                end
            end   
            
            % Compute difference of gaussian pyramid
            obj.DoG_pyramid = {};
            for i = 1:length(obj.gauss_pyramid)
                obj.DoG_pyramid{i} = obj.gauss_pyramid{i}(:,:,2:end) - obj.gauss_pyramid{i}(:,:,1:end-1);
            end

            % Get gradient of gaussian pyramid - used to compute second
            % moment matrix for affine invariance
            obj.grad_pyramid.x = {};
            obj.grad_pyramid.y = {};
            for i = 1:length(obj.gauss_pyramid)
                obj.grad_pyramid.x{i} = [];
                obj.grad_pyramid.y{i} = [];
                for j = 1:size(obj.gauss_pyramid{i},3)
                    obj.grad_pyramid.x{i} = cat(3,obj.grad_pyramid.x{i},image_match.array_grad(obj.gauss_pyramid{i}(:,:,j),'x'));
                    obj.grad_pyramid.y{i} = cat(3,obj.grad_pyramid.y{i},image_match.array_grad(obj.gauss_pyramid{i}(:,:,j),'y'));
                end
            end            
        end
        
        function get_feature_points(obj)
            % Calculate maxima_pyramid and feature_points
            obj.maxima_pyramid = cell(size(obj.DoG_pyramid));
            obj.feature_points = struct('x',{}, ...
                                        'y',{}, ...
                                        'r',{}, ...
                                        'r1',{}, ...
                                        'r2',{}, ...
                                        'rot',{}, ...
                                        'orientations',{});
            % Set k - gaussians are separated by this constant factor in 
            % scale space.
            k = 2^(1/obj.hp.s);
            for i = 1:length(obj.DoG_pyramid)
                for j = 2:size(obj.DoG_pyramid{i},3)-1 % Skip first and last scales
                    % Assign, to every voxel, the maximum of its neighbors.
                    % Then see if voxel value is greater than this value; 
                    % if this is true, then it's a local maxima (technique 
                    % is from Jonas on stackoverflow).

                    kernel = true(3,3,3);
                    kernel(2,2,2) = false;
                    DoG_dilate = imdilate(abs(obj.DoG_pyramid{i}(:,:,j-1:j+1)),kernel);
                    maxima_pyr = abs(obj.DoG_pyramid{i}(:,:,j)) > DoG_dilate(:,:,2);

                    % Clear out edge values
                    maxima_pyr(1,:) = false;
                    maxima_pyr(end,:) = false;
                    maxima_pyr(:,1) = false;
                    maxima_pyr(:,end) = false;
                    
                    % Store
                    obj.maxima_pyramid{i} = cat(3, ...
                                                obj.maxima_pyramid{i}, ...
                                                maxima_pyr);

                    % Get "pyramid coordinates" of maxima. Use "pyramid 
                    % coordinates" to distinguish these from pixel 
                    % coordinates wrt the original image.
                    [y_maxima_pyr,x_maxima_pyr] = find(maxima_pyr);

                    % Compute sigmas and change in sigmas to get finite
                    % difference approximation of derivatives 
                    octave_factor = (2^(i-1));
                    sigma = (k^(j-2))*octave_factor;
                    dsigma_1 = sigma-(k^(j-3))*octave_factor; 
                    dsigma_2 = (k^(j-1))*octave_factor-sigma;

                    % Go through points and do Lowe's cascade
                    for m = 1:length(y_maxima_pyr)    
                        % Refine location and scale first

                        % Calculate gradient: [ddog/dx_pyr ddog/dy_pyr ddog/dsigma]
                        % where p = [x_pyr y_pyr sigma]
                        dg_dp(1) = (obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)+1,j)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)-1,j))/2;
                        dg_dp(2) = (obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m),j)-obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m),j))/2; 
                        % Take average of both derivatives for sigma
                        dg_dp(3) = ((obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j+1)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j))/dsigma_2 + ...
                                    (obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j-1))/dsigma_1)/2;

                        % Calculate hessian:
                        %   [d^2dog/dx_pyr^2        d^2dog/(dx_pyr*dy_pyr) d^2dog/(dx_pyr*dsigma)
                        %    d^2dog/(dy_pyr*dx_pyr) d^2dog/dy_pyr^2        d^2dog/(dy_pyr*dsigma)
                        %    d^2dog/(dsigma*dx_pyr) d^2dog/(dsigma*dy_pyr) d^2dog/(dsigma^2)]
                        ddg_ddp(1,1) = obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)+1,j) - ...
                                       2*obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j) + ...
                                       obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)-1,j);
                        ddg_ddp(1,2) = ((obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m)+1,j)-obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m)+1,j))/2 - ...
                                        (obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m)-1,j)-obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m)-1,j))/2)/2;
                        ddg_ddp(1,3) = (((obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)+1,j+1)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)+1,j))/dsigma_2 + ...
                                         (obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)+1,j)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)+1,j-1))/dsigma_1)/2 - ...
                                        ((obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)-1,j+1)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)-1,j))/dsigma_2 + ...
                                         (obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)-1,j)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m)-1,j-1))/dsigma_1)/2)/2;
                        ddg_ddp(2,2) = obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m),j) - ...
                                       2*obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j) + ...
                                       obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m),j); 
                        ddg_ddp(2,3) = (((obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m),j+1)-obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m),j))/dsigma_2 + ...
                                         (obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m),j)-obj.DoG_pyramid{i}(y_maxima_pyr(m)+1,x_maxima_pyr(m),j-1))/dsigma_1)/2 - ...
                                        ((obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m),j+1)-obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m),j))/dsigma_2 + ...
                                         (obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m),j)-obj.DoG_pyramid{i}(y_maxima_pyr(m)-1,x_maxima_pyr(m),j-1))/dsigma_1)/2)/2; 
                        ddg_ddp(3,3) = ((obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j+1)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j))/dsigma_2 - ...
                                        (obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j)-obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j-1))/dsigma_1)/((dsigma_2+dsigma_1)/2);

                        % Fill lower half of hessian
                        ddg_ddp(2,1) = ddg_ddp(1,2);
                        ddg_ddp(3,1) = ddg_ddp(1,3);
                        ddg_ddp(3,2) = ddg_ddp(2,3);  

                        % Find incremental parameters
                        delta_p = -pinv(ddg_ddp)*dg_dp';   

                        % Get optimized locations
                        x_maxima_opt_pyr = x_maxima_pyr(m)+delta_p(1);
                        y_maxima_opt_pyr = y_maxima_pyr(m)+delta_p(2);
                        sigma_opt = sigma+delta_p(3);  

                        % Filter out points
                        if(abs(delta_p(1)) < 1 && abs(delta_p(2)) < 1 && abs(delta_p(3)) < dsigma_2 && ...                                  % Points should not move that much
                           abs(obj.DoG_pyramid{i}(y_maxima_pyr(m),x_maxima_pyr(m),j) + (1/2)*dg_dp*delta_p) > obj.hp.contrast_cutoff && ... % Reject extrema with low contrast 
                           trace(ddg_ddp(1:2,1:2))^2/det(ddg_ddp(1:2,1:2)) < (obj.hp.r_cutoff+1)^2/obj.hp.r_cutoff && ...                   % Elminate edge response        
                           x_maxima_opt_pyr >= 1 && ...                                                                                     % Make sure point is within the pyramid
                           x_maxima_opt_pyr <= size(obj.DoG_pyramid{i},2) && ...                                                            % ^
                           y_maxima_opt_pyr >= 1 && ...                                                                                     % ^
                           y_maxima_opt_pyr <= size(obj.DoG_pyramid{i},1) && ...                                                            % ^
                           sigma_opt > 0)                                                                                                   % sigma should be positive
                            % Compute second moment matrix for affine 
                            % invariance. Interpolate gradients at center
                            % specified by x_maxima_pyr and y_maxima_pyr. 
                            % Do this at non-optimized scale for now.

                            % Get radius in pyramid coordinates
                            radius_opt_pyr = (1+(k-1)/2)*sqrt(2)*sigma_opt/octave_factor;

                            % Get gaussian kernel for weights
                            l_kernel = 2*ceil(3*radius_opt_pyr)+1;
                            half_kernel = (l_kernel-1)/2;
                            kernel_gauss = fspecial('gaussian',[l_kernel l_kernel],radius_opt_pyr);

                            % Coordinates
                            [y_pyr,x_pyr] = ndgrid(y_maxima_opt_pyr-half_kernel:y_maxima_opt_pyr+half_kernel, ...
                                                   x_maxima_opt_pyr-half_kernel:x_maxima_opt_pyr+half_kernel);

                            % Interpolate gradients
                            grad_x = image_match.array_interp(obj.grad_pyramid.x{i}(:,:,j), ...
                                                              [x_pyr(:) y_pyr(:)], ...
                                                              'spline');
                            grad_y = image_match.array_interp(obj.grad_pyramid.y{i}(:,:,j), ...
                                                              [x_pyr(:) y_pyr(:)], ...
                                                              'spline');

                            % Only keep in-bounds points
                            idx_in = x_pyr >= 1 & x_pyr <= size(obj.DoG_pyramid{i},2) & ...
                                     y_pyr >= 1 & y_pyr <= size(obj.DoG_pyramid{i},1);
                                  
                            kernel_gauss_vec = kernel_gauss(idx_in);
                            grad_x = grad_x(idx_in);
                            grad_y = grad_y(idx_in);

                            % Get second moment matrix
                            M(1,1) = sum(kernel_gauss_vec.*grad_x.^2);
                            M(1,2) = sum(kernel_gauss_vec.*grad_x.*grad_y);
                            M(2,2) = sum(kernel_gauss_vec.*grad_y.^2);
                            M(2,1) = M(1,2);
                            
                            % Convert to pixel coordinates
                            x_pix = (x_maxima_opt_pyr-1)*octave_factor+1;
                            y_pix = (y_maxima_opt_pyr-1)*octave_factor+1;
                            r_pix = sqrt(2)*(1+(k-1)/2)*sigma_opt;
                            
                            % Get initial guess of ellipse by using second
                            % moment matrix. 
                            % Get major/minor axis and rotation of ellipse
                            [V,D] = eig(M);
                            [D,idx_sorted] = sort(diag(D));
                            
                            % Filter out edge response
                            if D(2)/D(1) < obj.hp.M_eig_ratio_cutoff                            
                                V = V(:,idx_sorted);        
                                % Ellipse sides are proportional to sqrt of eigenvalues
                                ellipse_scale_factor = 2*r_pix/(sqrt(D(1))+sqrt(D(2))); % have minor and major axis sum to diameter of blob
                                r1_pix = sqrt(D(2))*ellipse_scale_factor; % major axis radius
                                r2_pix = sqrt(D(1))*ellipse_scale_factor; % minor axis radius        
                                % Rotation of major axis
                                rot = -atan2(V(1,2),V(2,2));       

                                % Make xform to apply to coordinates of circle: rotation * scaling
                                mat_rot = [cos(rot) -sin(rot);  ...
                                           sin(rot)  cos(rot)];
                                mat_scale = [2*r1_pix/(r2_pix+r1_pix) 0; ...
                                             0                        2*r2_pix/(r2_pix+r1_pix)];
                                mat_affine = mat_rot * mat_scale;

                                % Apply xform to coordinates
                                x_pyr_affine = x_pyr - x_maxima_opt_pyr;
                                y_pyr_affine = y_pyr - y_maxima_opt_pyr;                            
                                p_pyr_affine = mat_affine * vertcat(x_pyr_affine(:)',y_pyr_affine(:)');
                                x_pyr_affine = p_pyr_affine(1,:)' + x_maxima_opt_pyr;
                                y_pyr_affine = p_pyr_affine(2,:)' + y_maxima_opt_pyr;

                                % Interpolate patch
                                patch_pyr = image_match.array_interp(obj.gauss_pyramid{i}(:,:,j), ...
                                                                     [x_pyr_affine(:) y_pyr_affine(:)], ...
                                                                     'spline');
                                patch_pyr = reshape(patch_pyr,l_kernel,l_kernel);

                                % Now perform orientation assignment on patch 
                                % using histogram.
                                patch_pyr_dx = image_match.array_grad(patch_pyr,'x');
                                patch_pyr_dy = image_match.array_grad(patch_pyr,'y');
                                patch_pyr_rot = atan2(patch_pyr_dy,patch_pyr_dx);
                                patch_pyr_mag = kernel_gauss.*sqrt(patch_pyr_dx.^2 + patch_pyr_dy.^2);
                                grad_hist = zeros(1,obj.hp.num_orientation_bins);
                                for n = 1:obj.hp.num_orientation_bins
                                    % Get rot in this range
                                    rot1 = (n-1)*2*pi/obj.hp.num_orientation_bins-pi;
                                    rot2 = n*2*pi/obj.hp.num_orientation_bins-pi;

                                    % Get histogram
                                    if n == obj.hp.num_orientation_bins
                                        grad_hist(n) = sum(patch_pyr_mag(patch_pyr_rot >= rot1 & ...
                                                                         patch_pyr_rot <= rot2));
                                    else
                                        grad_hist(n) = sum(patch_pyr_mag(patch_pyr_rot >= rot1 & ...
                                                                         patch_pyr_rot < rot2));
                                    end
                                end

                                % Find maxima
                                max_idx = find(grad_hist(2:end-1) > grad_hist(1:end-2) & ...
                                               grad_hist(2:end-1) > grad_hist(3:end)) + 1;
                                % Edge cases
                                if grad_hist(1) > grad_hist(end) && grad_hist(1) > grad_hist(2)
                                    max_idx(end+1) = 1; %#ok<AGROW>
                                end
                                if grad_hist(end) > grad_hist(end-1) && grad_hist(end) > grad_hist(1)
                                    max_idx(end+1) = length(grad_hist); %#ok<AGROW>
                                end

                                % Only keep maxima greater than 80% of the max
                                max_idx = max_idx(grad_hist(max_idx) > 0.8*max(grad_hist));                                    

                                if isempty(max_idx)
                                    % If this happens orientation cannot be
                                    % determined, so skip it.
                                    continue
                                end

                                % Create feature point per maxima location
                                orientations = struct('sift_patch',{},'mat_total',{}); 
                                for n = 1:length(max_idx)
                                    % Interpolate value
                                    if max_idx(n) == 1
                                        p1 = grad_hist(end);
                                        p2 = grad_hist(1); 
                                        p3 = grad_hist(2);
                                    elseif max_idx(n) == length(grad_hist)
                                        p1 = grad_hist(end-1);
                                        p2 = grad_hist(end); 
                                        p3 = grad_hist(1);
                                    else
                                        p1 = grad_hist(max_idx(n)-1);
                                        p2 = grad_hist(max_idx(n)); 
                                        p3 = grad_hist(max_idx(n)+1);
                                    end
                                    % Gauss newton update
                                    max_idx_opt = max_idx(n) - ((p3-p1)/2)/(p3-2*p2+p1);

                                    % Ensure idx is within bounds: [0.5 num_bins+0.5]
                                    max_idx_opt = mod(max_idx_opt-0.5,obj.hp.num_orientation_bins)+0.5;

                                    % Obtain angle with linear interpolation
                                    rot_opt = (-pi*(obj.hp.num_orientation_bins+0.5-max_idx_opt)+pi*(max_idx_opt-0.5))/obj.hp.num_orientation_bins;

                                    % Convert normalization angle to matrix tranformation  
                                    mat_rot_grad_hist = [cos(rot_opt) -sin(rot_opt);
                                                         sin(rot_opt)  cos(rot_opt)];

                                    % Get total transformation matrix
                                    mat_total = mat_affine * mat_rot_grad_hist;                                     

                                    % Now get sift patch. This will be used
                                    % to calculate the gradients to form 
                                    % the sift descriptors.

                                    % Coordinates
                                    [y_sp_pyr,x_sp_pyr] = ndgrid(linspace(-2*radius_opt_pyr,2*radius_opt_pyr,obj.hp.sift_window+2), ...
                                                                 linspace(-2*radius_opt_pyr,2*radius_opt_pyr,obj.hp.sift_window+2));

                                    % Apply xform to coordinates                           
                                    p_sp_pyr_total = mat_total * vertcat(x_sp_pyr(:)',y_sp_pyr(:)');
                                    x_sp_pyr_total = p_sp_pyr_total(1,:)' + x_maxima_opt_pyr;
                                    y_sp_pyr_total = p_sp_pyr_total(2,:)' + y_maxima_opt_pyr;

                                    % Interpolate patch
                                    sift_patch = image_match.array_interp(obj.gauss_pyramid{i}(:,:,j), ...
                                                                          [x_sp_pyr_total(:) y_sp_pyr_total(:)], ...
                                                                          'spline');
                                    sift_patch = reshape(sift_patch,obj.hp.sift_window+2,obj.hp.sift_window+2);

                                    % Store
                                    orientations(end+1).sift_patch = sift_patch; %#ok<AGROW>
                                    orientations(end).mat_total = mat_total;
                                end

                                % Store feature points
                                obj.feature_points(end+1).x = x_pix;
                                obj.feature_points(end).y = y_pix;
                                obj.feature_points(end).r = r_pix;
                                obj.feature_points(end).r1 = r1_pix;
                                obj.feature_points(end).r2 = r2_pix;
                                obj.feature_points(end).rot = rot;
                                obj.feature_points(end).orientations = orientations;
                            end
                        end
                    end
                end
            end
        end
                   
        function get_descriptors(obj)
            if isempty(obj.feature_points)
                error(['Feature points have not been computed yet ' ...
                       '--or-- none were found']);
            end
            
            % Make short-hand names
            parts = obj.hp.num_descriptor_parts;
            bins = obj.hp.num_descriptor_bins;
            sub_l = obj.hp.sift_window/obj.hp.num_descriptor_parts;            
            
            % Cycle over sift patches and calculate the sift descriptors. 
            kernel_gauss = fspecial('gaussian', ...
                                    [obj.hp.sift_window obj.hp.sift_window], ...
                                    obj.hp.sift_window/2);                
            for i = 1:length(obj.feature_points)
                orientations = struct('sift_vec',{});
                for j = 1:length(obj.feature_points(i).orientations)
                    grad_x = image_match.array_grad(obj.feature_points(i).orientations(j).sift_patch,'x');
                    grad_y = image_match.array_grad(obj.feature_points(i).orientations(j).sift_patch,'y');
                    grad_x = grad_x(2:end-1,2:end-1).*kernel_gauss;
                    grad_y = grad_y(2:end-1,2:end-1).*kernel_gauss;
                    
                    % Now calculate sift descriptor
                    sift_vec = zeros(1,obj.hp.num_descriptor_parts^2*obj.hp.num_descriptor_bins);
                    for m = 1:obj.hp.num_descriptor_parts
                        for n = 1:obj.hp.num_descriptor_parts
                            sift_vec((n-1+parts*(m-1))*bins+1:(n-1+parts*(m-1)+1)*bins) = sift_hist(grad_x((n-1)*sub_l+1:n*sub_l,(m-1)*sub_l+1:m*sub_l),...
                                                                                                    grad_y((n-1)*sub_l+1:n*sub_l,(m-1)*sub_l+1:m*sub_l));
                        end
                    end        

                    % Normalize -> threshold any values above 0.2 -> renormalize
                    sift_vec = sift_vec./norm(sift_vec);
                    sift_vec(sift_vec > 0.2) = 0.2;
                    sift_vec = sift_vec/norm(sift_vec);

                    % Store
                    orientations(j).sift_vec = sift_vec;
                end
                
                % Store
                obj.descriptors(i).orientations = orientations(j);
            end     
            
            function grad_hist = sift_hist(grad_x,grad_y)                
                % Get sift descriptor histogram
                
                grad_rot = atan2(grad_y,grad_x);
                grad_mag = sqrt(grad_x.^2 + grad_y.^2);
                grad_hist = zeros(1,obj.hp.num_descriptor_bins);
                for o = 1:obj.hp.num_descriptor_bins
                    % Get rot in this range
                    rot1 = (o-1)*2*pi/obj.hp.num_descriptor_bins-pi;
                    rot2 = o*2*pi/obj.hp.num_descriptor_bins-pi;

                    % Get histogram
                    if o == obj.hp.num_descriptor_bins
                        grad_hist(o) = sum(grad_mag(grad_rot >= rot1 & ...
                                                    grad_rot <= rot2));
                    else
                        grad_hist(o) = sum(grad_mag(grad_rot >= rot1 & ...
                                                    grad_rot < rot2));
                    end
                end
            end
        end
        
        function plot_gauss_pyramid(obj)
            % Get total width and height of composite
            height_total = size(obj.gauss_pyramid{1},1)*size(obj.gauss_pyramid{1},3);
            width_total = 0;
            for i = 1:length(obj.gauss_pyramid)
                width_total = width_total + size(obj.gauss_pyramid{i},2);
            end

            % Create composite image
            composite = zeros(height_total,width_total);   
            offset_x = 0;
            for i = 1:length(obj.gauss_pyramid)               
                for j = 1:size(obj.gauss_pyramid{i},3)
                    offset_y = (j-1)*size(obj.gauss_pyramid{i}(:,:,j),1);
                    composite(offset_y+1:offset_y+size(obj.gauss_pyramid{i}(:,:,j),1), ...
                              offset_x+1:offset_x+size(obj.gauss_pyramid{i}(:,:,j),2)) = obj.gauss_pyramid{i}(:,:,j);
                end
                offset_x = offset_x + size(obj.gauss_pyramid{i}(:,:,1),2);
            end

            % Plot
            f = figure;
            a = axes(f);
            imshow(composite,[],'parent',a);
            pause(0.5);
        end
        
        function plot_feature_points(obj)
            if isempty(obj.feature_points)
                error(['Feature points have not been computed yet ' ...
                       '--or-- none were found']);
            end

            % Plot background image
            f = figure;
            a = axes(f);            
            imshow(obj.array,[],'parent',a);
            hold(a,'on');

            % Plot feature points
            for i = 1:length(obj.feature_points)                
                ellipse(obj.feature_points(i).r1, ...
                        obj.feature_points(i).r2, ...
                        obj.feature_points(i).rot, ...
                        obj.feature_points(i).x, ...
                        obj.feature_points(i).y, ...
                        'r');
                text(obj.feature_points(i).x, ...
                     obj.feature_points(i).y, ...
                     num2str(i), ...
                     'parent',a);
            end

            % Remove hold
            hold(a,'off');
            pause(0.5);
        end
    end  
    
    methods(Static)  
        function array_g = array_gauss(array,sigma)
            window = 2*ceil(3*sigma)+1;
            array_g = imfilter(array,fspecial('gaussian',[window window],sigma), ...
                               'same','replicate');
        end
        
        function grad = array_grad(array,direc)
            switch direc
                case 'x'
                    grad = imfilter(array,-fspecial('sobel')'/8,'same','replicate');
                case 'y'
                    grad = imfilter(array,-fspecial('sobel')/8,'same','replicate');
                otherwise 
                    error(['Direction of gradient calculation can either' ...
                           'be: x or y, but: ' direc ' was input']);
            end
        end
        
        function vals = array_interp(array,points,method)
            if exist('method','var')
                vals = interp2(1:size(array,2),1:size(array,1),array,points(:,1),points(:,2),method);
            else
                vals = interp2(1:size(array,2),1:size(array,1),array,points(:,1),points(:,2));
            end
        end
        
        function match_points = get_match_points(ref,cur)
            if isempty(ref.descriptors) || isempty(cur.descriptors)
                error('Both inputs must have descriptors set.');
            end
            
            % Initialize match_points
            match_points = struct('x_ref',{},'y_ref',{},'descriptor_idx_ref',{},'orientation_idx_ref',{}, ...
                                  'x_cur',{},'y_cur',{},'descriptor_idx_cur',{},'orientation_idx_cur',{}, ...
                                  'dist',{});      

            % Do exhaustive search to find best matches
            for i = 1:length(ref.descriptors)
                % Keep track of best and second best matches
                descriptor_idx_ref_1 = i;
                orientation_idx_ref_1 = -1;
                descriptor_idx_cur_1 = -1;
                orientation_idx_cur_1 = -1;
                dist_1 = inf;
                dist_2 = inf;
                for j = 1:length(ref.descriptors(i).orientations)   
                    % Get ref descriptor
                    ref_sift_vec = ref.descriptors(i).orientations(j).sift_vec;

                    % Cycle over all cur descriptors and get the best and 
                    % second best matches      
                    descriptor_idx_cur_1_tmp = -1;
                    orientation_idx_cur_1_tmp = -1;
                    dist_1_tmp = inf;
                    dist_2_tmp = inf;
                    for k = 1:length(cur.descriptors)
                        for l = 1:length(cur.descriptors(k).orientations)
                            % Get cur descriptor
                            cur_sift_vec = cur.descriptors(k).orientations(l).sift_vec;                       
                                
                            dist = sum((ref_sift_vec-cur_sift_vec).^2);
                            if dist < dist_1_tmp
                                % Store previous best distance
                                dist_2_tmp = dist_1_tmp;

                                % Store current best distance
                                descriptor_idx_cur_1_tmp = k;
                                orientation_idx_cur_1_tmp = l;
                                dist_1_tmp = dist;
                            end      
                        end
                    end

                    % See if the best cur descriptor for this ref
                    % orientation is better than the last best
                    if dist_1_tmp < dist_1
                        % Replace previous best
                        orientation_idx_ref_1 = j;  
                        descriptor_idx_cur_1 = descriptor_idx_cur_1_tmp;    
                        orientation_idx_cur_1 = orientation_idx_cur_1_tmp;  
                        dist_1 = dist_1_tmp;  
                        dist_2 = dist_2_tmp;
                    end  
                end

                % Compare best and second match distances. If they are
                % very close, then toss the match (this implies instabilty).                  
                if dist_1/dist_2 < 0.8
                    % This is a good point, but check list and make sure 
                    % current location is unique, and if it's not then pick
                    % the best one
                    x_ref = ref.feature_points(i).x;
                    y_ref = ref.feature_points(i).y;
                    x_cur = cur.feature_points(descriptor_idx_cur_1).x;
                    y_cur = cur.feature_points(descriptor_idx_cur_1).y;

                    add_point = true;
                    for j = 1:length(match_points)
                        if match_points(j).x_cur == x_cur && match_points(j).y_cur == y_cur
                            % Match already exists, pick the current point 
                            % with the lower distance
                            if dist_1 < match_points(j).dist
                                % This new point is better; delete the old 
                                % point
                                match_points(j) = [];                                                                  
                                break
                            else
                                % Old point is better, do not add new point
                                add_point = false;
                                break
                            end
                        end
                    end  
                    
                    if add_point
                        % Add the new point
                        match_points(end+1).dist = dist_1; %#ok<AGROW>
                        match_points(end).x_ref = x_ref;
                        match_points(end).y_ref = y_ref;
                        match_points(end).descriptor_idx_ref = descriptor_idx_ref_1;
                        match_points(end).orientation_idx_ref = orientation_idx_ref_1;                            
                        match_points(end).x_cur = x_cur;
                        match_points(end).y_cur = y_cur;
                        match_points(end).descriptor_idx_cur = descriptor_idx_cur_1;
                        match_points(end).orientation_idx_cur = orientation_idx_cur_1; 
                    end
                end       
            end
        end   
               
        function plot_match_points(ref,cur,match_points)
            % Plot ref on the left and cur on the right
            offset_x = size(ref.array,2);                

            % Create montage
            composite = zeros(max(size(ref.array,1),size(cur.array,1)), ...
                              size(ref.array,2)+size(cur.array,2));
            composite(1:size(ref.array,1),1:size(ref.array,2)) = ref.array;
            composite(1:size(cur.array,1),(1:size(cur.array,2))+offset_x) = cur.array;

            % Plot
            f = figure;
            a = axes(f); 
            imshow(composite,[],'parent',a);
            hold(a,'on');
            c = distinguishable_colors(length(match_points));
            for i = 1:length(match_points)
                line([match_points(i).x_ref match_points(i).x_cur+offset_x], ...
                     [match_points(i).y_ref match_points(i).y_cur], ...
                     'color',c(i,:),'parent',a); 
                plot(match_points(i).x_ref,match_points(i).y_ref, ...
                     'parent',a,'Marker','s','Color',c(i,:),'parent',a);
                plot(match_points(i).x_cur+offset_x,match_points(i).y_cur, ...
                     'parent',a,'Marker','s','Color',c(i,:),'parent',a);
            end
            hold(a,'off');
            pause(0.5);
        end
        
        function [h, match_points_pose] = get_pose(ref,cur,match_points) %#ok<INUSL>
            if length(match_points) < 4
                error('At least four match points are required to compute homography.');
            end
            
            % Perform RANSAC on match points to find homography.  
            num_matches = 0;
            idx_good = [];
            for i = 1:ref.hp.RANSAC_it
                % Get four random points
                match_points_tmp = match_points(randperm(numel(match_points),4));
                
                % Compute homography
                h = homography([match_points_tmp.x_ref; match_points_tmp.y_ref]', ...
                               [match_points_tmp.x_cur; match_points_tmp.y_cur]');
                               
                % Apply homography to all reference coordinates
                p_cur = image_match.apply_homography(h,[match_points.x_ref; match_points.y_ref]');
                
                % Get number of points within delta
                dists = sqrt((p_cur(:,1) - [match_points.x_cur]').^2 + ...
                             (p_cur(:,2) - [match_points.y_cur]').^2);                         
                idx_good_tmp = find(dists < ref.hp.RANSAC_delta);
                
                if length(idx_good_tmp) > num_matches
                    num_matches = length(idx_good_tmp);
                    idx_good = idx_good_tmp;
                end
            end
            
            % Compute homography using best fit
            h = homography([match_points(idx_good).x_ref; match_points(idx_good).y_ref]', ...
                           [match_points(idx_good).x_cur; match_points(idx_good).y_cur]');
                       
            % Store match points used in pose estimation
            match_points_pose = match_points(idx_good);
                                  
            function homography_1_2 = homography(points_1,points_2)
                % Number of points
                num_points = size(points_1,1);

                % Perform normalization first
                T_1 = norm_mat(points_1);
                points_1_aug = [points_1 ones(num_points,1)]';
                points_1_norm = T_1*points_1_aug;

                T_2 = norm_mat(points_2);
                points_2_aug = [points_2 ones(num_points,1)]';
                points_2_norm = T_2*points_2_aug;

                % Compute homography with normalized points
                L = vertcat([points_1_norm' zeros(size(points_1_norm')) -points_2_norm(1,:)'.*points_1_norm'], ...
                            [zeros(size(points_1_norm')) points_1_norm' -points_2_norm(2,:)'.*points_1_norm']);    

                % Solution is the last column of V
                [~,~,V] = svd(L);    
                homography_norm_1_2 = reshape(V(:,end),3,3)';

                % "Undo" normalization to get desired homography
                homography_1_2 = T_2^-1*homography_norm_1_2*T_1;

                % Normalize homography_1_2(3,3) to 1
                homography_1_2 = homography_1_2./homography_1_2(3,3);
            end

            function T_norm = norm_mat(points)
                points_x = points(:,1);
                points_y = points(:,2);
                mean_x = mean(points_x);
                mean_y = mean(points_y);    
                sm = sqrt(2)*size(points,1)./sum(sqrt((points_x-mean_x).^2+(points_y-mean_y).^2));
                T_norm = [sm 0  -mean_x*sm;
                          0  sm -mean_y*sm;
                          0  0   1];
            end            
        end

        function points_2 = apply_homography(homography_1_2,points_1)
            points_2 = homography_1_2 * [points_1 ones(size(points_1,1),1)]';
            points_2 = (points_2(1:2,:)./points_2(3,:))';
        end

        function plot_pose(ref,cur,h)
            % Plot ref on the left and cur on the right
            offset_x = size(ref.array,2);                

            % Create montage
            composite = zeros(max(size(ref.array,1),size(cur.array,1)), ...
                              size(ref.array,2)+size(cur.array,2));
            composite(1:size(ref.array,1),1:size(ref.array,2)) = ref.array;
            composite(1:size(cur.array,1),(1:size(cur.array,2))+offset_x) = cur.array;

            % Get points around reference image
            box_ref = [1 1; ...
                       1 size(ref.array,1); ...
                       size(ref.array,2) size(ref.array,1); ...
                       size(ref.array,2) 1; ...
                       1 1];
                     
            % Apply homography to points
            box_cur = image_match.apply_homography(h,box_ref);

            % Plot
            f = figure;
            a = axes(f); 
            imshow(composite,[],'parent',a);
            hold(a,'on');
            plot(box_ref(:,1),box_ref(:,2),'-b','LineWidth',2,'parent',a);
            plot(offset_x+box_cur(:,1),box_cur(:,2),'-b','LineWidth',2,'parent',a);
            for i = 1:4
                plot([box_ref(i,1) offset_x + box_cur(i,1)], ...
                     [box_ref(i,2) box_cur(i,2)],'-b','LineWidth',1,'parent',a);                
            end
            hold(a,'off');
            pause(0.5);
        end
    end
end