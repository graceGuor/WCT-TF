# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

	mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import os
import time

# This is a placeholder for a Google-internal import.
import tensorflow as tf
import grpc
import numpy
import random
import scipy
import threading

from utils import get_img, save_img

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


tf.app.flags.DEFINE_integer('concurrency', 10,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_float('alpha', 1.0, 'alpha')
tf.app.flags.DEFINE_integer('num_tests', 10, 'Number of test images')
tf.app.flags.DEFINE_integer('content-size', 512, '')
# tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('server', '10.11.11.31:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('out_path', 'res/stylized_img_client', 'Output folder path. ')
FLAGS = tf.app.flags.FLAGS


def resize_to(img, resize=512):
    '''Resize long side to target size and preserve aspect ratio'''
    height, width = img.shape[0], img.shape[1]
    if height > width:
        ratio = height / resize
        long_side = round(width / ratio)
        resize_shape = (resize, long_side, 3)
    else:
        ratio = width / resize
        long_side = round(height / ratio)
        resize_shape = (long_side, resize, 3)

    return scipy.misc.imresize(img, resize_shape, interp='bilinear')


def postprocess(image):
    return numpy.uint8(numpy.clip(image, 0, 1) * 255)


def preprocess(image, size=512):
    image = resize_to(image, size)
    if len(image.shape) == 3:  # Add batch dimension
        image = numpy.expand_dims(image, 0)
    return image / 255.  # Range [0,1]


class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._active = 0
    self._condition = threading.Condition()

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def do_inference(hostport, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.

    Args:
      hostport: Host:port address of the PredictionService.
      work_dir: The full path of working directory for test data set.
      concurrency: Maximum number of concurrent requests.
      num_tests: Number of test images to use.

    Returns:
      The classification error rate.

    Raises:
      IOError: An error occurred processing test data set.
    """

    content_image_path_prefix = "../../data/style_transfer/cases/content"
    style_image_path_prefix = "../../data/style_transfer/cases/styles/"
    contents = [
        "cat.jpg",
        #         "golden_gate.jpg",
        #         "man-playing-tennis.jpg",
        #         "tubingen.jpg"
                ]
    styles = [
        # "colorful-girl.png",
        #       "tiger.png",
        #       "the-starry-night.jpg",
              "water-droplets.png",
              # "the-scream.jpg",
              # "seated-nude.jpg"
        ]


    size = 512
    print("size:", size)
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    sum_time = 0
    result_counter = _ResultCounter(num_tests, concurrency)
    for i in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'wct'  # 与target name相同
        request.model_spec.signature_name = 'predict_words'

        random_content = random.randint(0, len(contents) - 1)
        random_style = random.randint(0, len(styles) - 1)
        # random_content = 2
        content_image_path = os.path.join(content_image_path_prefix, contents[random_content])
        style_image_path = os.path.join(style_image_path_prefix, styles[random_style])
        print(contents[random_content], styles[random_style])

        # 读取测试图片content_img和style_img
        content_img = get_img(content_image_path)
        style_img = get_img(style_image_path)


        content_img = preprocess(content_img, size).astype(numpy.float32)
        style_img = preprocess(style_img, size).astype(numpy.float32)

        request.inputs['input_content'].CopyFrom(tf.make_tensor_proto(content_img))

        # style_image = numpy.array(Image.open(style_image_path), dtype=numpy.float32)[:, :, :3][numpy.newaxis, :, :, :]
        request.inputs['input_style'].CopyFrom(tf.make_tensor_proto(style_img))
        result_counter.throttle()
        result = stub.Predict.future(request, 500.0)  # 5 seconds

        start_time = time.time()
        result = result.result().outputs['output_stylized']
        end_time = time.time()
        sum_time += (end_time - start_time)

        result_shape = [d.size for d in list(result.tensor_shape.dim)]
        result_arr = numpy.array(result.float_val).reshape(result_shape[1:])
        result_arr = postprocess(result_arr)
        # print(result_arr, result_arr.shape, type(result_arr))

        content_prefix, content_ext = os.path.splitext(content_image_path)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext
        style_prefix, _ = os.path.splitext(style_image_path)
        style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

        # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
        out_f = os.path.join(FLAGS.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))

        save_img(out_f, result_arr)
        print(i, "Wrote stylized output image to {} at {}".format(out_f, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


    average_time = sum_time / FLAGS.num_tests
    print("stylized", FLAGS.num_tests, "images in average_time:", average_time)

    return result_arr


def do_inference2(hostport, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.
    test throughout.

    Args:
      hostport: Host:port address of the PredictionService.
      work_dir: The full path of working directory for test data set.
      concurrency: Maximum number of concurrent requests.
      num_tests: Number of test images to use.

    Returns:
      The classification error rate.

    Raises:
      IOError: An error occurred processing test data set.
    """

    content_image_path_prefix = "../../data/style_transfer/cases/content"
    style_image_path_prefix = "../../data/style_transfer/cases/styles/"
    contents = [
        # "han.jpg",
        # "avatar.jpg",
        # "xiaonvjing.jpg",
        # "yangmi.jpg",
        # "yangmi2.jpg",
        # "zhoujielun.jpg",
        "nanhai.png",
        # "han_avatar.jpg",
        # "cat.jpg",
        #         "golden_gate.jpg",
        #         "man-playing-tennis.jpg",
        #         "tubingen.jpg"
                ]
    styles = [
        # "colorful-girl.png",
        #       "tiger.png",
        #       "the-starry-night.jpg",
        # "the-scream.jpg",
        # "seated-nude.jpg",
        # "water-droplets.png",
        #       "panda.jpg",
        #       "dongmannv.jpg",
        #       "dongmannv2.jpeg",
        #       "dongmannv3.jpeg",
              "heibai.png",
              # "dongmannan.png",
        #       "chimuqingzi.jpg",
              # "xiaonvjing.jpg",
              # "dolphin.jpg"

        ]

    size = 512
    print("size:", size)
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    sum_time = 0
    result_counter = _ResultCounter(num_tests, concurrency)
    content_imgs = []
    style_imgs = []
    for i in range(num_tests):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'wct'  # 与target name相同
        request.model_spec.signature_name = 'predict_words'

        random_content = random.randint(0, len(contents) - 1)
        random_style = random.randint(0, len(styles) - 1)
        # random_content = 2
        content_image_path = os.path.join(content_image_path_prefix, contents[random_content])
        style_image_path = os.path.join(style_image_path_prefix, styles[random_style])
        print(contents[random_content], styles[random_style])

        # 读取测试图片content_img和style_img
        content_img = get_img(content_image_path)
        style_img = get_img(style_image_path)


        content_img = preprocess(content_img, size).astype(numpy.float32)
        style_img = preprocess(style_img, size).astype(numpy.float32)

        content_imgs.append(content_img)
        style_imgs.append(style_img)

        content_prefix, content_ext = os.path.splitext(content_image_path)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext
        style_prefix, _ = os.path.splitext(style_image_path)
        style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext


        request.inputs['input_content'].CopyFrom(tf.make_tensor_proto(content_img))
        # request.inputs['input_content'].CopyFrom(tf.make_tensor_proto(numpy.array(content_imgs)))

        # style_image = numpy.array(Image.open(style_image_path), dtype=numpy.float32)[:, :, :3][numpy.newaxis, :, :, :]
        print("style_img.shape:", style_img.shape)
        print("style_imgs.shape:", len(style_imgs))
        print("style_imgs.shape:", numpy.array(style_imgs).shape)
        request.inputs['input_style'].CopyFrom(tf.make_tensor_proto(style_img))
        # request.inputs['input_style'].CopyFrom(tf.make_tensor_proto(numpy.array(style_imgs)))
        # alpha = FLAGS.alpha
        alpha = (i + 1) / num_tests * FLAGS.alpha
        print(i, "alpha:", alpha)
        request.inputs['input_alpha'].CopyFrom(tf.make_tensor_proto(alpha))
        result_counter.throttle()
        result = stub.Predict.future(request, 500.0)  # 5 seconds

        start_time = time.time()
        result = result.result().outputs['output_stylized']
        end_time = time.time()
        sum_time += (end_time - start_time)

        result_shape = [d.size for d in list(result.tensor_shape.dim)]
        result_arr = numpy.array(result.float_val).reshape(result_shape[1:])
        result_arr = postprocess(result_arr)
        # print(result_arr, result_arr.shape, type(result_arr))
        # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
        out_f = os.path.join(FLAGS.out_path, '{}_{}_{}{}'.format(content_prefix, style_prefix, alpha, content_ext))

        save_img(out_f, result_arr)
        print(i, "Wrote stylized output image to {} at {}".format(out_f, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


    average_time = sum_time / FLAGS.num_tests
    print("stylized", FLAGS.num_tests, "images in average_time:", average_time)

    return result_arr


def main(_):
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return
    if not FLAGS.server:
        print('please specify server host:port')
        return
    # do_inference(FLAGS.server, FLAGS.concurrency, FLAGS.num_tests)
    do_inference2(FLAGS.server, FLAGS.concurrency, FLAGS.num_tests)


if __name__ == '__main__':
    tf.app.run()
