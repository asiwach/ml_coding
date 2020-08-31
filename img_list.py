import glob
import os


def img_list(input_dir):
    images = [img for img in glob.glob(os.path.join(input_dir, "*.jpg"))]
    images = images + [img for img in glob.glob(os.path.join(input_dir, "*.jpeg"))]
    images = images + [img for img in glob.glob(os.path.join(input_dir, "*.png"))]
    images = images + [img for img in glob.glob(os.path.join(input_dir, "*.JPG"))]
    return images
