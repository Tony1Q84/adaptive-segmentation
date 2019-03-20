import numpy as np
import os
import cv2
from FaceAligner import FaceAlign
from face_alignment.api import FaceAlignment, LandmarksType


source_dataset_path = '../datasets/fer2013/'
cant_crop_path = '../datasets/crop_plus_rotate/cant_crop/'
more_face_path = '../datasets/crop_plus_rotate/more_face/'
Coor_2D = FaceAlignment(LandmarksType._2D, flip_input = False)
Rotate = FaceAlign()

def cut_range(landmarks):
   a = landmarks.max(axis = 0)
   # print(a)
   b = landmarks.min(axis = 0)
   # print(b)
   low_max = np.clip(a, 0, 48)
   low_min = np.clip(b, 0, 48)
   # width = low_max[0] - low_min[0]
   # height = low_max[1] - low_min[1]
   # x_min = low_min[0]
   # y_min =
   return low_min[0], low_min[1], low_max[0], low_max[1]

def crop_by_landmls(image, landmarks):
    out_face = np.zeros_like(image)
    # landmarks = landmarks.astype(int)

    # remapped_shape = np.zeros_like(landmarks)
    feature_mask = np.zeros((image.shape[0], image.shape[1]))
    remapped_shape = cv2.convexHull(landmarks)
    points = remapped_shape.astype(int)
    cv2.fillConvexPoly(feature_mask, points, 1)
    feature_mask = feature_mask.astype(np.bool)
    out_face[feature_mask] = image[feature_mask]

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
    # gray_image = HistEqualize(gray_image)
    landmarks = Coor_2D.get_landmarks(gray_image)
    if landmarks is not None:
        landmarks = np.asarray(landmarks)
        if landmarks.shape == (1, 68, 2):
            landmarks = np.squeeze(landmarks)
            landmarks = landmarks.astype(int)
            # x_min, y_min, x_max, y_max = cut_range(landmarks)
            out_face = crop_by_landmls(image, landmarks)
            faceAligned = Rotate.align(out_face, landmarks)
            faceAligned = cv2.resize(faceAligned, (48, 48))
            # faceAligned = Rotate.align(out_face, landmarks)
            source_path, filename = os.path.split(image_path)
            save_path = source_path.replace('fer2013', 'crop_plus_rotate')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path, filename), faceAligned)

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