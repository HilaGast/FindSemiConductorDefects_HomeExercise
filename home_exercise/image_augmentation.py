import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image

from MuseAI_HomeExercise.detect_differences import read_image


def create_new_images(image, n=10):
    """create n new images from an image
    Input:
        image: image file path
        n: number of new images to create
    """
    org_img = read_image(image)
    image_file_parts = image.split(".")
    image_name = image_file_parts[0]
    images = []
    images.append(org_img)

    for i in range(n):
        new_img_name = image_name + "_new_" + str(i) + "." + image_file_parts[1]
        new_img = org_img.copy()
        new_img = gamma_correction(new_img, gamma=random.uniform(0.8, 1.2))
        new_img = gaussian_noise(
            new_img, mean=random.uniform(0, 0.1), var=random.uniform(0, 0.05)
        )
        new_img = salt_and_pepper_noise(new_img, prob=random.uniform(0, 0.02))
        new_img = random_translation(
            new_img, x=random.randint(-10, 10), y=random.randint(-5, 5)
        )
        new_img = intensity_scaling(new_img, factor=random.uniform(0.8, 1.2))
        images.append(new_img)
        save_image(new_img, new_img_name)

    plot_images(images, 2, int(np.ceil((n + 1) / 2)))


def gamma_correction(image, gamma=1.0):
    """gamma correction on image
    Input:
        image: image to perform gamma correction on
        gamma: gamma value for gamma correction
    Output:
        gamma_corrected_image: gamma corrected image
    """
    gamma_corrected_image = np.array(255 * (image / 255) ** gamma, dtype="uint8")
    return gamma_corrected_image


def gaussian_noise(image, mean=0, var=0.1):
    """add gaussian noise to image
    Input:
        image: image to add noise to
        mean: mean value for gaussian noise
        var: variance value for gaussian noise
    Output:
        noisy_image: image with added gaussian noise
    """
    noisy_image = skimage.util.random_noise(image, mode="gaussian", mean=mean, var=var)
    noisy_image = np.array(255 * noisy_image, dtype="uint8")
    return noisy_image


def salt_and_pepper_noise(image, prob=0.1):
    """add salt and pepper noise to image
    Input:
        image: image to add noise to
        prob: probability for salt and pepper noise
    Output:
        noisy_image: image with added salt and pepper noise
    """
    noisy_image = skimage.util.random_noise(image, mode="s&p", amount=prob)
    noisy_image = np.array(255 * noisy_image, dtype="uint8")
    return noisy_image


def random_translation(image, x=0, y=0):
    """translate image
    Input:
        image: image to translate
        x: x translation value
        y: y translation value
    Output:
        translated_image: translated image
    """
    rows, cols = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    return translated_image


def intensity_scaling(image, factor=1.0):
    """scale image intensity
    Input:
        image: image to scale
        factor: scaling factor
    Output:
        scaled_image: scaled image
    """
    scaled_image = np.array(factor * image, dtype="uint8")
    return scaled_image


def plot_images(images, rows, cols):
    """plot images
    Input:
        images: list of images to plot
        rows: number of rows
        cols: number of columns
    Output:
        None
    """
    fig, ax = plt.subplots(rows, cols, figsize=(15, 5))
    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(images[i * rows + j], cmap="gray")
            ax[i, j].set_title("Image " + str(i * cols + j))
    plt.show()


def save_image(image, image_name):
    """save image
    Input:
        image: image to save
        image_name: image file path
    """
    im = Image.fromarray(image)
    im.save(image_name)


def main():
    create_new_images(
        r"C:\Users\hilag\PycharmProjects\pythonProject\MuseAI_HomeExercise\home_exercise\defective_examples\case2_reference_image.tif",
        n=10,
    )


if __name__ == "__main__":
    main()
