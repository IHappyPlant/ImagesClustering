"""This file contains main code for images clustering"""
import os
from argparse import ArgumentParser
from time import time

import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import MiniBatchKMeans


def main(args_):
    """Split images into clusters"""
    start = time()
    model = InceptionV3(include_top=False)  # load model

    features_list = []
    files = os.listdir(args_.images)
    print('Started features extraction')
    features_extr_start = time()
    for f in files:
        pth = os.path.join(args_.images, f)  # get path to image

        # load and preprocess image
        img = image.load_img(pth, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        # extract features from image
        feature = np.array(model.predict(img_data))
        features_list.append(feature.flatten())

    features_list = np.array(features_list)
    features_extr_end = time()
    print('Features extracted')
    print(f'Features extraction time: '
          f'{features_extr_end - features_extr_start}')

    # run KMeans for extracted features
    print('Started clustering')
    clustering_start = time()
    kmeans = MiniBatchKMeans(n_clusters=args_.clusters,
                             random_state=0).fit_predict(features_list)
    clustering_end = time()
    print('Clustering finished')
    print(f'Clustering time: {clustering_end - clustering_start}')

    # generate output directory for each cluster
    output_dirs = [os.path.join(args_.destination, str(i))
                   for i in range(args_.clusters)]
    for d in output_dirs:
        os.mkdir(d)

    # write files to corresponding cluster directories
    for i, file in enumerate(files):
        pth = os.path.join(args_.images, file)
        img = cv2.imread(pth)
        if img is not None:
            output_pth = os.path.join(output_dirs[kmeans[i]], file)
            cv2.imwrite(output_pth, img)
        else:
            print(f'Can\'t read image f{pth}')

    end = time()
    print(f'Full time: {end - start}')


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
