# https://github.com/bearsprogrammer/real-time-deep-face-recognition

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import cv2
import math
import pickle
from sklearn.svm import SVC



MIN_SIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709  # scale factor
MARGIN = 44
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
RANDOM_KEY = np.random.randint(0, high=99999)
MODEL_PRETRAIN_DIR = 'data/20170511-185253/20170511-185253.pb'
FACE_DETECT_PRETRAIN = 'data'


def CreateAlignFolder(raw_data_dir = 'human_data/unalign', output_dir = 'human_data/align'):
    output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = facenet.get_dataset(raw_data_dir)

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, FACE_DETECT_PRETRAIN)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 182

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    print('Goodluck')

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                        print('read data dimension: ', img.ndim)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                            print('to_rgb data dimension: ', img.ndim)
                        img = img[:, :, 0:3]
                        print('after data dimension: ', img.ndim)

                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('detected_face: %d' % nrof_faces)
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                det = det[index, :]
                            det = np.squeeze(det)
                            bb_temp = np.zeros(4, dtype=np.int32)

                            bb_temp[0] = det[0]
                            bb_temp[1] = det[1]
                            bb_temp[2] = det[2]
                            bb_temp[3] = det[3]

                            cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                            scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')

                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, scaled_temp)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

def Training(data_align_dir = 'human_data/align', classifier_filename = 'my_classifier.pkl'):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            dataset = facenet.get_dataset(data_align_dir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            labels = np.array(labels) - len(set(labels)) 
            print(labels)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            print('Loading feature extraction model')
            modeldir = 'data/20170511-185253/20170511-185253.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            batch_size = 1000
            image_size = 160
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            print('Goodluck')




def PredictFaceByImage(face_path, classifier_filename = 'my_classifier.pkl'):

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, FACE_DETECT_PRETRAIN)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(MODEL_PRETRAIN_DIR)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            img = misc.imread(face_path)
            if img.ndim < 2:
                print('Unable to align "%s"' % face_path)
            if img.ndim == 2:
                img = facenet.to_rgb(img)
                print('to_rgb data dimension: ', img.ndim)
            img = img[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(img, MIN_SIZE, pnet, rnet, onet, THRESHOLD , FACTOR)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(img[0]) or bb[i][3] >= len(img):
                        print('face is inner of range!')
                        continue

                    cropped.append(img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[0] = facenet.flip(cropped[0], False)
                    scaled.append(misc.imresize(cropped[0], (IMAGE_SIZE, IMAGE_SIZE), interp='bilinear'))
                    scaled[0] = cv2.resize(scaled[0], (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                            interpolation=cv2.INTER_CUBIC)
                    scaled[0] = facenet.prewhiten(scaled[0])
                    scaled_reshape.append(scaled[0].reshape(-1,INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE,3))
            else:
                print('Unable to align "%s"' % image_path)
            emb_array = np.zeros((1, embedding_size))
            feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

            model = pickle.load(open(classifier_filename, 'rb'))[0]
            print(model.predict(emb_array))

# CreateAlignFolder()
Training()
PredictFaceByImage('human_data/test/test/2.JPG')
