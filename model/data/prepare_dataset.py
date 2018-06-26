import cv2
import glob
import numpy as np

def process_images(images_paths, save_path, set_prefix):
    targets = []
    labels = []
    for image_path in glob.glob(images_paths):
        image = cv2.imread(image_path).astype(np.float32)
        b, g, r = cv2.split(image)
        image = cv2.merge((r, g, b))
        height, width, channels = image.shape
        targets.append(image[:,:width//2,:])
        labels.append(image[:, width//2:, :])
    targets = np.array(targets, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    np.save(save_path.format("target", set_prefix), targets)
    np.save(save_path.format("label", set_prefix), labels)


if __name__ == '__main__':
    folder_path_train = './facades/train/*.jpg'
    folder_path_val = './facades/val/*.jpg'
    folder_path_test = './facades/test/*.jpg'
    save_path = './facades/{0}_{1}_dataset.np'
    process_images(folder_path_train, save_path, "train")
    process_images(folder_path_val, save_path, "val")
    process_images(folder_path_test, save_path, "test")