# Find defects based on differnces between two images
Detect differences between two images to find defects

## Code environment:
Python 			        3.11.2

See requirements.txt for used libraries

## Assumptions:
- Grayscale images with 3 graded intensities as input
- Images size around 300-500 pixels in each direction
- Uses translation registration: assumes no shearing, 	reflection, scaling or rotation between images
- Should be able to deal with both small and large defects - 	avoids strong filtering
- Favors false positive over false negatives

