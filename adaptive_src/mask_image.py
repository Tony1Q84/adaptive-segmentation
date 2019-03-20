import numpy as np
import os
import cv2
from face_alignment.api import FaceAlignment, LandmarksType

source_dataset_path = '../datasets/RAF/Image/aligned/'
cant_crop_path = '../datasets/RAF/Image/RAF_mask/cant_crop/'
more_face_path = '../datasets/RAF/Image/RAF_mask/more_face/'
Coor_2D = FaceAlignment(LandmarksType._2D, flip_input = False)


def crop_by_landmls(image, landmarks):
    out_face = np.zeros_like(image)
    # landmarks = landmarks.astype(int)

    # remapped_shape = np.zeros_like(landmarks)
    # feature_mask = np.zeros((image.shape[0], image.shape[1]))
    remapped_shape = cv2.convexHull(landmarks)
    points = remapped_shape.astype(int)
    cv2.fillConvexPoly(out_face, points, (255, 255, 255))
    # feature_mask = feature_mask.astype(np.bool)
    # out_face[feature_mask] = image[feature_mask]

    return out_face


image_paths = []
for folder, subfolders, filenames in os.walk(source_dataset_path):
    for filename in filenames:
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(folder, filename))

num = len(image_paths)
more_face = 0
non_face = 0

for image_arg, image_path in enumerate(image_paths):
    print('Processing : {}/{}'.format(image_arg+1, num))
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = Coor_2D.get_landmarks(gray_image)
    if landmarks is not None:
        landmarks = np.asarray(landmarks)
        if landmarks.shape == (1, 68, 2):
            landmarks = np.squeeze(landmarks)
            landmarks = landmarks.astype(int)
            out_face = crop_by_landmls(image, landmarks)
            source_path, filename = os.path.split(image_path)
            save_path = source_path.replace('aligned', 'RAF_mask')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, filename), out_face)

        else:
            more_face += 1
            source_path, filename = os.path.split(image_path)
            save_path = os.path.join(more_face_path, filename)
            cv2.imwrite(save_path, gray_image)

    else:
        non_face += 1
        source_path, filename = os.path.split(image_path)
        save_path = os.path.join(cant_crop_path, filename)
        cv2.imwrite(save_path, gray_image)

print('There ara {} non_faces, and {} more_faces'.format(non_face, more_face))
print('Finished!')

# image = cv2.imread("/home/tony/lvhui/face-alignment/test/assets/4.jpg")
# # image = imutils.resize(image, width=500)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# landmarks= Coor_2D.get_landmarks(gray)
# landmaks = np.asarray(landmarks)
# landmarks = np.squeeze(landmarks)
#
# landmarks = landmarks.astype(int)
# out_face = crop_by_landmls(image, landmarks)
# cv2.imshow("mask_inv", out_face)
# cv2.waitKey()