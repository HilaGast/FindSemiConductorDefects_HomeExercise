import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_defects(reference: str, inspected: str):
    """detect and return a binary image of any differences appear in inspected and not in reference
    Input:
        reference: path to reference image
        inspected: path to inspected image
    Output:
        defects: binary image of any differences appear in inspected and not in reference
    """
    # Read the images:
    ref_img = read_image(reference)
    insp_img = read_image(inspected)

    # Image normalization for intensity matching - Histogram matching:
    matched_img = histogram_matching(ref_img, insp_img)

    # Image registration:
    moving_keypoints, target_keypoints, top_matches = get_matching_features_orb(
        ref_img, matched_img, nfeatures=1000, threshold=0.5
    )
    translation_matrix = calculate_translation_matrix(
        moving_keypoints, target_keypoints, top_matches
    )
    translated_img = register_image(ref_img, translation_matrix)

    # Image denoising:
    denoised_matched = denoise_image(matched_img)
    denoised_translated = denoise_image(translated_img)
    s = ssim_images(denoised_matched, denoised_translated)

    # Detect images contours:
    matched_contour_mask = detect_contours(
        denoised_matched,
        sigma=1,
        kernel_size=1,
    )
    translated_contur_mask = detect_contours(
        denoised_translated,
        sigma=1,
        kernel_size=1,
    )

    # Image subtraction between registered ref and matched images, including thresholding:
    diff_img = image_differences(
        denoised_matched,
        denoised_translated,
        matched_contour_mask * 100,
        percentileTH=99,
    )

    # Plot the difference image alongside original images
    plot_results(translated_img, matched_img, diff_img)


def read_image(path):
    """read an image from path
    Input:
        path: path to image
    Output:
        img: image"""
    import cv2

    # Read the image and convert to grayscale:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def histogram_matching(reference, inspected):
    """ " perform histogram matching between two images
    Input:
        reference: reference image
        inspected: inspected image
    Output:
        matched: histogram matched image
    """
    from skimage import exposure

    matched = exposure.match_histograms(inspected, reference)
    matched = matched.astype("uint8")
    return matched


def gaussian_filter(img, kernel_size=3):
    """apply gaussian filter on an image
    Input:
        img: image
        kernel_size: size of the kernel
    Output:
        img_filtered: filtered image
    """
    from scipy.ndimage import gaussian_filter

    img_filtered = gaussian_filter(img, kernel_size)

    return img_filtered


def low_pass_filter(img, kernel_size=3):
    """apply low pass filter on an image
    Input:
        img: image
        kernel_size: size of the kernel
    Output:
        img_filtered: filtered image
    """
    import cv2

    img_filtered = cv2.blur(img, (kernel_size, kernel_size))
    return img_filtered


def local_thresholding(img, block_size=3, offset=0):
    """apply local thresholding on an image
    Input:
        img: image
        block_size: size of the block
        offset: offset
    Output:
        img_thresholded: thresholded image
    """
    from skimage.filters import threshold_local

    threshold = threshold_local(img, block_size, offset=offset)
    img_thresholded = img > threshold
    return img_thresholded


def detect_contours(img, sigma=3.0, kernel_size=3, low_threshold=10, high_threshold=20):
    """Detect contours in an image using Canny edge detection
    Input:
        img: image
    Output:
        contours: contours found in the image
    """
    from skimage import feature
    from skimage.morphology import dilation, disk

    img_contour = feature.canny(img, sigma, low_threshold, high_threshold)
    img_contour = img_contour.astype("uint8")
    img_contour = dilation(img_contour, disk(kernel_size))

    return img_contour


def get_matching_features_orb(moving, target, nfeatures, threshold=0.75):
    """get matching features between two images using ORB
    Input:
        target: reference image for registration
        moving: moving image to register
    Output:
        kp1: keypoints of the moving image
        kp2: keypoints of the target image
        matches: matching features
    """
    # Initiate ORB detector:
    orb = cv2.ORB_create(nfeatures)

    # Find the keypoints and descriptors with ORB:
    moving_keypoints, des1 = orb.detectAndCompute(moving, None)
    target_keypoints, des2 = orb.detectAndCompute(target, None)

    # Create BFMatcher object:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors:
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance:
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the best top_matches% of the matches:
    top_matches = matches[: int(len(matches) * threshold)]

    return moving_keypoints, target_keypoints, top_matches


def calculate_translation_matrix(moving_keypoints, target_keypoints, matches):
    """calculate the translation matrix between two images
    Input:
        moving_keypoints: keypoints of the moving image
        target_keypoints: keypoints of the target image
        matches: matching features
    Output:
        translation_matrix: translation matrix between the two images
    """
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((len(matches), 2))
    p2 = np.zeros((len(matches), 2))

    for i in range(len(matches)):
        p1[i, :] = moving_keypoints[matches[i].queryIdx].pt
        p2[i, :] = target_keypoints[matches[i].trainIdx].pt

    # Find the homography matrix:
    # homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    translation_matrix, inliers = cv2.estimateAffinePartial2D(p1, p2)

    return translation_matrix


def register_image(img, translation_matrix):
    """register an image
    Input:
        image: image to register
        traslation_matrix: translation matrix
    Output:
        registered: registered image
    """
    x, y = img.shape

    # use the homography matrix to align the images
    registered = cv2.warpAffine(img, translation_matrix, (y, x))

    return registered


def denoise_image(img):
    """denoise an image using non-local means denoising
    Input:
        img: image
    Output:
        denoised_img: denoised image
    """
    denoised_img = cv.fastNlMeansDenoising(img, None, 7, 14, 21)
    return denoised_img


def ssim_images(img1, img2):
    """calculate the structural similarity index between two images
    Input:
        img1: image
        img2: image
    Output:
        ssim: structural similarity index
    """
    from skimage.metrics import structural_similarity

    score, diff_img = structural_similarity(img1, img2, full=True)
    return diff_img


def image_differences(inspected, reference, ref_contour, percentileTH=90):
    """ " find the differences between two images
    Input:
        reference: reference image
        inspected: inspected image
    Output:
        diff: image of the differences between the two images
    """
    inspected[reference == 0] = 0
    # remove the median of the reference contour from each image:
    inspected = inspected.astype("float64")
    # inspected = (inspected - np.min(inspected)) / (np.max(inspected) - np.min(inspected))
    reference = reference.astype("float64")
    # reference = (reference - np.min(reference)) / (np.max(reference) - np.min(reference))
    ref_contour = ref_contour.astype("float64")
    diff = np.sqrt((inspected - reference) ** 2)
    diff = diff - ref_contour
    diff[diff <= 0] = 0
    diff[diff < np.percentile(diff, percentileTH)] = 0

    diff[diff > 0] = 1
    diff = diff.astype("uint8")
    _, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff = diff.astype("bool")
    return diff


def plot_results(reference, inspected, diff_img):
    """plot the results of the defect detection
    Input:
        reference: reference image
        inspected: inspected image
        diff_img: image of the differences between the two images
    Output:
        None
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(reference, cmap="gray")
    ax[0].set_title("Reference image")
    ax[1].imshow(inspected, cmap="gray")
    ax[1].set_title("Inspected image")
    ax[2].imshow(diff_img, cmap="gray")
    ax[2].set_title("Defects Found")
    plt.show()


def main():
    find_defects(
        r"C:\Users\hilag\PycharmProjects\pythonProject\MuseAI_HomeExercise\home_exercise\defective_examples\case2_reference_image.tif",
        r"C:\Users\hilag\PycharmProjects\pythonProject\MuseAI_HomeExercise\home_exercise\defective_examples\case2_inspected_image.tif",
    )


if __name__ == "__main__":
    main()
