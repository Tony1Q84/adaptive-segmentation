import numpy as np
import os
import cv2
from FaceAligner import FaceAlign
from face_alignment.api import FaceAlignment, LandmarksType


source_dataset_path = '../datasets/fer2013/'
cant_crop_path = '../datasets/fer2013_crop/cant_crop/'
more_face_path = '../datasets/fer2013_crop/more_face/'
Coor_2D = FaceAlignment(LandmarksType._2D, flip_input = False)
Rotate = FaceAlign()

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
            faceAligned = Rotate.align(image, landmarks)
            source_path, filename = os.path.split(image_path)
            save_path = source_path.replace('fer2013', 'fer2013_crop')
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


# image_path = '../images/PrivateTest_1369768_3.jpg'
# image = cv2.imread(image_path)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# landmarks = Coor_2D.get_landmarks(gray_image)
# landmarks = np.asarray(landmarks)
# landmarks = np.squeeze(landmarks)
# faceAligned = Rotate.align(image, landmarks)
# res = np.hstack((image, faceAligned))
# cv2.imwrite('res4.png', res)