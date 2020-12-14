# coding=utf-8
"""
This file contains main code for images clustering
"""
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import MiniBatchKMeans


def main(args_):
    """Split images into clusters"""
    model = InceptionV3(include_top=False)

    features_list = []
    files = os.listdir(args_.images)
    for f in files:
        pth = os.path.join(args_.images, f)
        img = image.load_img(pth, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        feature = np.array(model.predict(img_data))
        features_list.append(feature.flatten())
    features_list = np.array(features_list)

    kmeans = MiniBatchKMeans(n_clusters=args_.clusters,
                             random_state=0).fit_predict(features_list)

    output_dirs = [os.path.join(args_.destination, str(i))
                   for i in range(args_.clusters)]
    for d in output_dirs:
        os.mkdir(d)

    for i, file in enumerate(files):
        pth = os.path.join(args_.images, file)
        img = cv2.imread(pth)
        if img is not None:
            output_pth = os.path.join(output_dirs[kmeans[i]], file)
            cv2.imwrite(output_pth, img)
        else:
            print(f'Can\'t read image f{pth}')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument(
        '--images', '-I', type=str, required=True,
        help='Path to directory with images to be clustered')
    argparser.add_argument('--clusters', type=int, default=2,
                           help='Number of clusters for KMeans')
    argparser.add_argument('--destination', '-D', type=str, required=True,
                           help='Path to directory where clustered images'
                                'will be stored')

    args = argparser.parse_args()
    main(args)
