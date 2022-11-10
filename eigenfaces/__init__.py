from PIL import Image
import numpy as np
import sys
import glob
import os
from math import sqrt
import func
import pca

class EigenFaces(object):

    def train(self, training_images_folder):
        self.projected_classes = []

        self.list_of_arrays_of_images, self.labels_list, list_of_flattened_classs = read_images(training_images_folder)

        images_matrix = np.array([np.array(Image.fromarray(img)).flatten()
                                  for img in self.list_of_arrays_of_images], 'f')

        self.eigenface_matrix, variance, self.mean_image = pca.pca(images_matrix)

        for sample_class in list_of_flattened_classs:
            class_weights_vertex = self.projectImage(sample_class)
            self.projected_classes.append(class_weights_vertex.mean(0))

    def projectImage(self, X):
        X = X - self.mean_image
        return np.dot(X, func.Transpose(self.eigenface_matrix))

    def facePredict(self, X):
        min_class = -1
        min_distance = np.finfo('float').max
        projected_target = self.projectImage(X)
        projected_target = np.delete(projected_target, -1)
        for i in range(len(self.projected_classes)):
            distance = func.Norm(projected_target - np.delete(self.projected_classes[i], -1))
            if distance < min_distance:
                min_distance = distance
                min_class = self.labels_list[i]

        return min_class

    def __repr__(self):
        return "PCA (num_components=%d)" % (self._num_components)

def read_images(path, sz=None):

    class_samples_list = []
    class_matrices_list = []
    images, image_labels = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            class_samples_list = []
            for filename in os.listdir(subject_path):
                if filename != ".DS_Store":
                    try:
                        im = Image.open(os.path.join(subject_path, filename))
                        if (sz is not None):
                            im = im.resize(sz, Image.ANTIALIAS)
                        images.append(np.asarray(im, dtype=np.uint8))

                    except IOError as e:
                        errno, strerror = e.args
                        print("I/O error({0}): {1}".format(errno, strerror))
                    except:
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                    class_samples_list.append(np.asarray(im, dtype=np.uint8))

            class_samples_matrix = np.array([img.flatten()
                                             for img in class_samples_list], 'f')

            class_matrices_list.append(class_samples_matrix)

            image_labels.append(subdirname)

    return images, image_labels, class_matrices_list