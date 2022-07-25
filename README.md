# Visual_Odometry
The objective of this project was to predict the camera path of the given video frames without using functions that would directly
solve the problem, such as the Slam libraries to compute the visual odometry.

Algorithm Outline for eight-point-algorithm track
1. Read images into array
2. Extract matches between images using opencv FlannBasedMatcher and opencv ORB
algorithm to detect features.
3. Calculate Fundamental Matrix using eight-point-algorithm
4. Estimate fundamental matrix from the given correspondences using RANSAC
5. Calculate the essential matrix
6. Decomposing the essential matrix into Rotation and translation matrix and find the best
pair of rotation and translation values.
7. Multiply the translation by the scale information provided for each frame id
8. Make the transformation matrix.
9. Run path prediction using frames

