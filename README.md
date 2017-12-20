# Images_Stitching
Stitches 2 images into a single bird-view image

# The stitching algorithm requires four steps:
# (1) Detecting keypoints and extracting local invariant descriptors (SIRF)
# (2) Matching descriptors between images (k-NN with k=2 + David Lowe's ratio test)
# (3) Applying RANSAC to estimate the homography transform (3x3 matrix)
# (4) Applying a warping transformation using the homography matrix
#
# References:
# (1) https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching
# (2) https://www.pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv
# (3) https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6
