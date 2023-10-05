# MuseAI_HomeExercise
Detect differences between two similar images to find defects

Code environment:
Python 			3.11.2
matplotlib        	3.8.0
numpy             	1.24.2
opencv-python     	4.8.1.78
scikit-image      	0.21.0
scikit-learn      	1.3.1
scipy             	1.11.3

Assumptions:
- Grayscale images with 3 graded intensities as input
- Images size around 300-500 pixels in each direction
- Uses translation registration: assumes no shearing, 	reflection, scaling or rotation between images
- Should be able to deal with both small and large defects - 	avoids strong filtering
- Favors false positive over false negatives

