# Image-Mosaicing

#### Description:
This project generates an image mosaic/ panorama from images. The following are the sequence of steps of the process
```
0. Click pictures such that adjacent pictures have atleast 40% overlap. Order the pictures from left to right
1. Generate SIFT features among images
2. Establish correspondence between features
3. Apply RANSAC (Random Sampling and Consensus) to get rid of outliers
4. Generate initial estimate of Homography using inlier correspondence points obtained in step 4
5. Refine the Homography estimate using Levenberg- Marquardt optimization
6. Repeat the above steps for each of the adjacent image pairs 
7. After obtaining Homographies for each of the picture pairs, get the homographies with respect to the central image
8. Project all images (using inverse warping) on to a blank canvas. Use bilinear interpolation
```

#### Dependencies

- OpenCV
- Numpy
- SciPy (if you want to test results from Levenberg-Marquardt)

## Scripts
- [**image_mosiac.py**](./image_mosaic.py): **_MAIN_** file to run. Pass the correct parent folder and image list in the script. Result folder with all intermediate images and final panorama are generated in /<parent_folder>/results
```python
python image mosaic.py
```

###### Supporting scripts
- [**match_features.py**](./match_features.py): Script to generate SIFT features and establish correspondence between features of two images
- [**ransac.py**](./ransac.py): RANSAC algorithm to obtain inliers and discard outliers among correspondences
- [**optimize_fcn.py**](./optimize_fcn.py): Levenberg-Marquardt algorithm for optimization. Generic script and can be used for any Non-Linear Least Squares Estimation
- [**estimate_homography.py**](./estimate_homography.py): Helper functions which help in bilinear interpolation and projecting images to a canvas using Homography matrix

## Results

- [Input set 1 - click click](./input/p2)
    - [Results for Input set 1 - more clicks](./input/p2/results)
    
###### Inputs

|<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/1.jpg" alt="1.jpg" width="234" height="416" /> 1.jpg  |<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/2.jpg" alt="2.jpg" width="234" height="416" />  2.jpg| 
|<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/3.jpg" alt="3.jpg" width="234" height="416" /> 3.jpg  |<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/4.jpg" alt="4.jpg" width="234" height="416" />  4.jpg|
|<img src="https://github.com/aartighatkesar/Image-Mosaicing/blob/master/input/p2/5.jpg" alt="5.jpg" width="234" height="416" /> 5.jpg  |
    
    
        
        










