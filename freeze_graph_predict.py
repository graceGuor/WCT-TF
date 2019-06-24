#! /usr/bin/env python

import argparse
import tensorflow as tf
from tensorflow import graph_util as graph_util
import os
import numpy as np
import scipy
import shutil
from utils import get_files, get_img, get_img_crop, save_img, resize_to, center_crop
from tensorflow.python.tools import inspect_checkpoint as chkp


# Parameters
parser = argparse.ArgumentParser()

# Data Parameters
tf.flags.DEFINE_string("label_id_file", "../resource/vocab_out_emojis", "Data source for emoji label.")
tf.flags.DEFINE_string("output_file", "../resource/score_output", "output score file.")
tf.flags.DEFINE_string("data_file", "../resource/eval", "Data source for eval phase.")
tf.flags.DEFINE_string("eval_label", "../resource/eval_label", "eval data with label file.")
# Build pb model
tf.flags.DEFINE_boolean("is_build_pb", True, "if build pb language model")
tf.flags.DEFINE_string("pb_save_path", "models/pbs", "pb save file.")
tf.flags.DEFINE_integer("model_version", 1, "emoji prediction model version")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "models/relu5_1", "Checkpoint directory from training run")
tf.flags.DEFINE_string("relu_target", "relu5_1", "Relu target layers corresponding to decoder checkpoints")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_fraction", 0.9, "gpu fraction (default: 0.5)")


parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path', default="res/freeze_graph_predict")

parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512", default=0)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=0)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)

args = parser.parse_args()

FLAGS = tf.flags.FLAGS


def preprocess(image):
    if len(image.shape) == 3:  # Add batch dimension
        image = np.expand_dims(image, 0)
    return image / 255.  # Range [0,1]


def freeze_graph_test(pb_file, content_path, style_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    content_prefix, content_ext = os.path.splitext(content_path)
    content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext
    style_prefix, _ = os.path.splitext(style_path)
    style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

    # 读取测试图片content_img和style_img
    content_img = get_img(content_path)
    if args.content_size > 0:
        content_img = resize_to(content_img, args.content_size)

    style_img = get_img(style_path)

    if args.style_size > 0:
        style_img = resize_to(style_img, args.style_size)
    if args.crop_size > 0:
        style_img = center_crop(style_img, args.crop_size)

    # if args.keep_colors:
    #     style_img = preserve_colors_np(style_img, content_img)

    content_img = preprocess(content_img)
    style_img = preprocess(style_img)

    # 打印检查点所有的变量
    chkp.print_tensors_in_checkpoint_file("models/inference/model.ckpt", tensor_name='', all_tensors=False)

    # for i in range(5):
    #     print("=" * 10, i + 1)
    #     chkp.print_tensors_in_checkpoint_file("models/relu" + str(i+1) + "_1/model.ckpt-15002", tensor_name='', all_tensors=True)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        # with open(pb_file, "rb") as f:
        #     x = f.read()
        #     # print(x)
        #     output_graph_def.ParseFromString(x)
        #     output_graph_def.ParseFromString(f.read())
        #     tf.import_graph_def(output_graph_def, name="")
        #     for i, n in enumerate(output_graph_def.node):
        #         print("Name of the node - %s" % n.name)
        #         print(n)

        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            # tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)


            # 定义输入的张量名称,对应网络结构的输入张量
            content_input_tensor = sess.graph.get_tensor_by_name(
                "encoder_decoder_relu5_1/content_encoder_relu5_1/content_imgs:0")
            style_input_tensor = sess.graph.get_tensor_by_name("style_img:0")

            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name(
                "encoder_decoder_relu5_1/decoder_relu5_1/decoder_model_relu5_1/relu5_1_16/relu5_1_16/BiasAdd:0")


            # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
            stylized_rgb = sess.run(output_tensor_name, feed_dict={content_input_tensor: content_img,
                                                                   style_input_tensor: style_img})


            # Stitch the style + stylized output together, but only if there's one style image
            if args.concat:
                # Resize style img to same height as frame
                style_img_resized = scipy.misc.imresize(style_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                stylized_rgb = np.hstack([style_img_resized, stylized_rgb])

            # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
            out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))

            save_img(out_f, stylized_rgb)


if __name__ == '__main__':
    model_version = FLAGS.model_version
    version_export_path = os.path.join(FLAGS.pb_save_path + "/" + str(model_version))
    pb_file = os.path.join(version_export_path, 'saved_model.pb')
    print("Importing trained model from ", pb_file)
    # content_image_path = "/Users/grace/data/style_transfer_test_cases/content/cat.jpg"
    # style_image_path = "/Users/grace/data/style_transfer_test_cases/styles/tiger.png"
    content_image_path = "../../data/style_transfer/cases/content/cat.jpg"
    style_image_path = "../../data/style_transfer/cases/styles/tiger.png"

    freeze_graph_test(pb_file, content_image_path, style_image_path)

    print("Finished!")
