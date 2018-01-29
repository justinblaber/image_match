# image_match
SIFT implementation + pose estimation in MATLAB. It uses the classic DoG blob detector for feature point detection and the SIFT descriptor for feature point correspondence. Pose estimation uses RANSAC to compute the best homography using matched feature points in the reference and current image.

Below is a step by step explanation of test.m:

1) The first step is to compute the feature points and descripts for a "reference" image like so:

```
ref_img = imread('test.png');

ref = image_match(ref_img(40:250,70:280));
ref.get_feature_points();
ref.get_descriptors();
ref.plot_feature_points();
```
![alt text](https://i.imgur.com/Q51lOZo.png)

2) The next step is to do the same thing for a "current" image:

```
T = projective2d([0.89   1.24  0.002; 
                  -0.95  1     0.0025;
                  0      0     1]);
cur_img = imwarp(ref_img,T); 

cur = image_match(cur_img);
cur.get_feature_points();
cur.get_descriptors();
cur.plot_feature_points();
```
![alt text](https://i.imgur.com/67SS1Ij.png)

3) Determine "match points" next:

```
match_points = image_match.get_match_points(ref,cur);
image_match.plot_match_points(ref,cur,match_points);
```

![alt text](https://i.imgur.com/qsUkU3i.png)

4) Use RANSAC to determine the match points to use for pose estimation (which computes a homography):

```
[h_pose, match_points_pose] = image_match.get_pose(ref,cur,match_points);
image_match.plot_match_points(ref,cur,match_points_pose);
```

![alt text](https://i.imgur.com/intPIlM.png)

5) Plot the pose!

```
image_match.plot_pose(ref,cur,h_pose);
```
![alt text](https://i.imgur.com/wIjwlHz.png)





