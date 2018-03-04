# image_match
This is a SIFT implementation + pose estimation in MATLAB. It uses the classic DoG blob detector for feature point detection and the SIFT descriptor for feature point correspondence. Pose estimation uses RANSAC to compute the best homography using matched feature points in the "reference" and "current" images. Note that this was a side project I did for fun and is far from a fast/robust implementation.

# Installation instructions:
```
git clone https://github.com/justinblaber/image_match.git
```
Then, in MATLAB:

```
>> addpath('image_match');
>> test_image_match
```

Below is a step by step explanation of `test_image_match.m`:

1) The first cell:

```
ref_img = imread('test.png');

ref = image_match(ref_img(40:250,70:280));
ref.get_feature_points();
ref.get_descriptors();
ref.plot_feature_points();
```
computes the feature points (using DoG blob detector) and descriptors (SIFT) for the reference image. The resulting plot should look like:

![alt text](https://i.imgur.com/Q51lOZo.png)

2) The next cell does the same thing for the deformed image:

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
The resulting plot should look like:

![alt text](https://i.imgur.com/67SS1Ij.png)

3) The next cell determines the "match points":

```
match_points = image_match.get_match_points(ref,cur);
image_match.plot_match_points(ref,cur,match_points);
```
The resulting plot should look like:

![alt text](https://i.imgur.com/qsUkU3i.png)

4) The last cell uses RANSAC to determine the match points to use for pose estimation (which computes a homography):

```
[h_pose, match_points_pose] = image_match.get_pose(ref,cur,match_points);
image_match.plot_match_points(ref,cur,match_points_pose);
image_match.plot_pose(ref,cur,h_pose);
```
The resulting plots should look like:

![alt text](https://i.imgur.com/intPIlM.png)

![alt text](https://i.imgur.com/wIjwlHz.png)





