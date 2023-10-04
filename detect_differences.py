from typing import Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
import skimage
from cv2 import (
    DMatch,
    KeyPoint,
    imread,
    IMREAD_GRAYSCALE,
    NORM_HAMMING,
    BFMatcher,
    estimateAffinePartial2D,
    fastNlMeansDenoising,
    connectedComponentsWithStats,
    warpAffine,
)
import cv2
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, disk
from sklearn.cluster import KMeans
import argparse


def find_defects(reference_path: str, inspected_path: str, should_plot: bool = True) -> None:
    """
    Find defects in an inspected image compared to a reference image
    Input:
        reference: path to reference image
        inspected: path to inspected image
    Output:
        None
    """

    # Read the images
    ref_img = read_image(reference_path)
    insp_img = read_image(inspected_path)

    # Pre-processing
    preprocessed_insp, preprocessed_ref = preprocess(insp_img, ref_img)

    # Create 1st mask using the differences between kmeans clusters of the two images
    kmeans_mask = create_kmeans_mask(preprocessed_insp, preprocessed_ref, object_min_size=30)
    if should_plot:
        plot_results(
            ref_img, insp_img, kmeans_mask, labels=("reference", "inspected", "Kmeans mask"), title="Kmeans mask result"
        )

    # Create 2nd mask using the contours difference between the two images
    contour_mask = create_contours_mask(
        preprocessed_insp,
        preprocessed_ref,
        sigma_contour=1,
        sigma_gaussian=2,
        kernel_size=3,
        object_min_size=100,
    )

    if should_plot:
        plot_results(
            ref_img,
            insp_img,
            contour_mask,
            labels=("reference", "inspected", "contour mask"),
            title="Contour mask result",
        )

    # Combine masks, keeping only the pixels that are different in both masks:
    mask = kmeans_mask * contour_mask

    if should_plot:
        plot_results(
            ref_img,
            insp_img,
            mask,
            labels=("reference", "inspected", "mask"),
            title="Masks",
        )


def read_image(path: str) -> np.ndarray:
    # Read the image and convert to grayscale:
    img = imread(path, IMREAD_GRAYSCALE)
    return img


def preprocess(
    insp_img: np.ndarray,
    ref_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Match the intensity of the inspected image using Histogram matching:
    hist_matched_insp_img = match_histogram(ref_img, insp_img)

    # Image registration - we will move the reference image to match the inspected image:
    # Step 1: Feature matching
    moving_keypoints, target_keypoints, top_matches = get_matching_features_orb(
        ref_img,
        hist_matched_insp_img,
        nfeatures=1000,
        top_matches_to_use=0.5,  # use only the top 50% of the features
    )

    # Step 2: Calculate the translation matrix
    translation_matrix = calculate_translation_matrix(
        moving_keypoints,
        target_keypoints,
        top_matches,
    )

    # Step 3: Register the reference image
    translated_ref_img = register_image(ref_img, translation_matrix)
    hist_matched_insp_img[translated_ref_img == 0] = 0

    # Image denoise:
    denoised_insp_img = denoise_image(hist_matched_insp_img, h=7, template_window_size=7, search_window_size=21)
    denoised_ref_img = denoise_image(translated_ref_img, h=7, template_window_size=7, search_window_size=21)

    return denoised_insp_img, denoised_ref_img


def match_histogram(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    # Perform histogram matching between two images:
    matched = skimage.exposure.match_histograms(moving, reference)
    matched = matched.astype("uint8")
    return matched


def get_matching_features_orb(
    moving: np.ndarray,
    target: np.ndarray,
    nfeatures: int,
    top_matches_to_use: float = 0.5,
) -> tuple[Sequence[KeyPoint], Sequence[KeyPoint], list[DMatch]]:
    """get matching features between two images using ORB
    Input:
        target: reference image for registration
        moving: moving image to register
    Output:
        kp1: key points of the moving image
        kp2: key points of the target image
        matches: matching features
    """
    # Initiate ORB detector:
    orb = cv2.ORB_create(nfeatures)

    # Find the key points and descriptors with ORB:
    moving_keypoints, moving_descriptors = orb.detectAndCompute(moving, mask=None)
    target_keypoints, target_descriptors = orb.detectAndCompute(target, mask=None)

    # Create BFMatcher object:
    bf = BFMatcher(NORM_HAMMING, crossCheck=True)

    # Match descriptors:
    matches = bf.match(moving_descriptors, target_descriptors)

    # Sort descriptors by their distance:
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the best matches % of the matches:
    top_matches = matches[: int(len(matches) * top_matches_to_use)]

    return moving_keypoints, target_keypoints, top_matches


def calculate_translation_matrix(
    moving_keypoints: Sequence[KeyPoint],
    target_keypoints: Sequence[KeyPoint],
    matches: list[DMatch],
) -> np.ndarray:
    """calculate the translation matrix between two images
    Input:
        moving_keypoints: keypoints of the moving image
        target_keypoints: keypoints of the target image
        matches: matching features
    Output:
        translation_matrix: translation matrix between the two images
    """
    matches_count = len(matches)
    p1 = np.zeros((matches_count, 2))
    p2 = np.zeros((matches_count, 2))

    for i in range(matches_count):
        p1[i, :] = moving_keypoints[matches[i].queryIdx].pt
        p2[i, :] = target_keypoints[matches[i].trainIdx].pt

    # Find the translation matrix, ignore inliers
    translation_matrix, _ = estimateAffinePartial2D(p1, p2)

    return translation_matrix


def register_image(img: np.ndarray, translation_matrix: np.ndarray) -> np.ndarray:
    # Register an image using a translation matrix:
    x, y = img.shape
    registered = warpAffine(img, translation_matrix, (y, x))

    return registered


def denoise_image(
    img: np.ndarray,
    h: int = 7,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """denoise an image using non-local means denoising
    Input:
        img: image
        h: level of denoising
        template_window_size: template window size for the denoising
        search_window_size: search window size for the denoising
    Output:
        denoised_img: denoised image
    """
    denoised_img = fastNlMeansDenoising(
        img, None, h, templateWindowSize=template_window_size, searchWindowSize=search_window_size
    )

    return denoised_img


def create_kmeans_mask(
    img1: np.ndarray,
    img2: np.ndarray,
    object_min_size: int = 30,
) -> np.ndarray:
    """Create a mask using kmeans for three instensity groups and the differences in classes between the two images
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image
        min_size: minimum size of objects to keep in the mask
    Output:
        mask: mask of the differences between the two images
    """

    # Create kmeans mask where k=3 for both images:
    mask_img1 = get_sorted_labels_kmeans(img1, n_clusters=3)
    mask_img2 = get_sorted_labels_kmeans(img2, n_clusters=3)

    # Find the differences between the two masks in each class
    cluster_0_mask = mask_img1[:, :, 0] != mask_img2[:, :, 0]
    cluster_1_mask = mask_img1[:, :, 1] != mask_img2[:, :, 1]
    cluster_2_mask = mask_img1[:, :, 2] != mask_img2[:, :, 2]

    # Add masks together:
    combined_mask = cluster_0_mask.astype("uint8") + cluster_1_mask.astype("uint8") + cluster_2_mask.astype("uint8")

    # Remove small objects:
    combined_mask = find_and_remove_small_objects(combined_mask, threshold=object_min_size)

    # Binarize for final mask:
    combined_mask[combined_mask > 0] = 1

    return combined_mask


def get_sorted_labels_kmeans(
    img: np.ndarray,
    n_clusters: int = 3,
    init: tuple[list[int], ...] = ([10], [127], [220]),
) -> np.ndarray:
    """Create a mask using kmeans for n instensity groups:
    Input:
        img: numpy array of the image
        n_clusters: number of clusters for the kmeans
        init: initial centers for the kmeans. Default is [[10], [127], [220]] for three clusters
    Output:
        mask: mask for the different clusters, sorted by centroids values"""

    # Perform kmeans clustering:
    model = KMeans(n_clusters=n_clusters, init=init, max_iter=1000, n_init="auto", tol=1e-5).fit(img.reshape(-1, 1))
    labels = model.labels_.reshape(img.shape)
    centers = model.cluster_centers_.reshape(-1)

    # Sort the clusters by their centroids values:
    sorted_idx = np.argsort(centers)
    mask = np.zeros([img.shape[0], img.shape[1], n_clusters])
    for i in range(n_clusters):
        mask[:, :, i] = labels == sorted_idx[i]

    return mask


def create_contours_mask(
    img1: np.ndarray,
    img2: np.ndarray,
    sigma_contour: int = 1,
    sigma_gaussian: int = 2,
    kernel_size: int = 3,
    object_min_size: int = 30,
) -> np.ndarray:
    """Create a mask using the differences contours between the two images
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image
        sigma_contour: sigma value for the contour detection
        sigma_gaussian: sigma value for the gaussian filter
        kernel_size: size of the kernel for the contour detection
        min_size: minimum size of objects to keep in the mask
    Output:
        mask: mask of the differences between the two images"""

    # Find contours in both images:
    img1_contour_mask = detect_contours(
        img1,
        sigma=sigma_contour,
        kernel_size=kernel_size,
    )
    img2_contour_mask = detect_contours(
        img2,
        sigma=sigma_contour,
        kernel_size=kernel_size,
    )

    # Create image masked by contours and apply gaussian filter:
    img1_contour = gaussian_filter(img1_contour_mask * img1, sigma=sigma_gaussian)
    img2_contour = gaussian_filter(img2_contour_mask * img2, sigma=sigma_gaussian)

    # Calculate the structural similarity index between the two masked contour images:
    ssim = ssim_images(img1_contour, img2_contour)

    # change range values to 0-1 and invert the image:
    ssim = (ssim - np.min(ssim)) / (np.max(ssim) - np.min(ssim))
    ssim = 1 - ssim

    # Threshold the ssim image:
    threshold = skimage.filters.threshold_otsu(ssim)
    ssim[ssim < threshold] = 0
    contours_mask = ssim * 255
    contours_mask = contours_mask.astype("uint8")

    # Remove small objects:
    contours_mask = find_and_remove_small_objects(contours_mask, threshold=object_min_size)

    # Binarize for final mask:
    contours_mask[contours_mask > 0] = 1

    return contours_mask


def detect_contours(
    img: np.ndarray,
    sigma: float = 3.0,
    kernel_size: int = 3,
    low_threshold: int = 10,
    high_threshold: int = 20,
) -> np.ndarray:
    """Detect contours in an image using Canny edge detection
    Input:
        img: image
        sigma: sigma value for Canny edge detection
        kernel_size: size of the kernel for dilation of edge mask
        low_threshold: low threshold for Canny edge detection
        high_threshold: high threshold for Canny edge detection

    Output:
        img_contour: image with contours
    """
    img_contour = skimage.feature.canny(img, sigma, low_threshold, high_threshold)
    img_contour = img_contour.astype("uint8")
    img_contour = dilation(img_contour, disk(kernel_size))

    return img_contour


def ssim_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    # Calculate the structural similarity index between two images
    score, diff_img = skimage.metrics.structural_similarity(img1, img2, full=True)

    return diff_img


def find_and_remove_small_objects(img: np.ndarray, threshold: int = 20) -> np.ndarray:
    """
    Find and remove small objects in an image
    Input:
        img: np.ndarray of the image
        threshold: minimum size of objects to keep in the image
    Output:
        img: np.ndarray of the image with the small objects removed
    """
    # find all connected components (white blobs in the image):
    blobs_count, img_with_separated_blobs, stats, _ = connectedComponentsWithStats(img)

    # Remove background from the list:
    sizes = stats[:, -1]
    sizes = sizes[1:]
    blobs_count -= 1

    # For every component in the image, keep it only if it's above threshold:
    cleaned_img = np.zeros_like(img_with_separated_blobs)
    for blob in range(blobs_count):
        if sizes[blob] >= threshold:
            # see description of im_with_separated_blobs above
            cleaned_img[img_with_separated_blobs == blob + 1] = 255

    return cleaned_img


def plot_results(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    labels: Tuple[str, ...] = ("img1", "img2", "img3"),
    title: str = "",
) -> None:
    # Plot the results comparing 3 images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img1, cmap="gray")
    ax[0].set_xlabel(labels[0], fontsize=16)
    ax[1].imshow(img2, cmap="gray")
    ax[1].set_xlabel(labels[1], fontsize=16)
    ax[2].imshow(img3, cmap="gray")
    ax[2].set_xlabel(labels[2], fontsize=16)
    fig.suptitle(title, fontsize=24)
    plt.show()


def main() -> None:
    # Parse arguments:
    parser = argparse.ArgumentParser(description="Find defects in an inspected image compared to a reference image")
    parser.add_argument("--reference", type=str, required=True, help="path to reference image")
    parser.add_argument("--inspected", type=str, required=True, help="path to inspected image")
    args = parser.parse_args()

    # Find defects in the inspected image compared to the reference image:
    find_defects(args.reference, args.inspected)


if __name__ == "__main__":
    main()
