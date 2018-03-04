%% Set "reference" image, get feature points, and get descriptors
ref_img = imread('test.png');

ref = image_match(ref_img(40:250,70:280));
ref.get_feature_points();
ref.get_descriptors();
ref.plot_feature_points();

%% Set "current" image, get feature points, and get descriptors
T = projective2d([0.89   1.24  0.002; 
                  -0.95  1     0.0025;
                  0      0     1]);
cur_img = imwarp(ref_img,T); 

cur = image_match(cur_img);
cur.get_feature_points();
cur.get_descriptors();
cur.plot_feature_points();

%% Match points
match_points = image_match.get_match_points(ref,cur);
image_match.plot_match_points(ref,cur,match_points);

%% Pose estimation
[h_pose, match_points_pose] = image_match.get_pose(ref,cur,match_points);
image_match.plot_match_points(ref,cur,match_points_pose);
image_match.plot_pose(ref,cur,h_pose);