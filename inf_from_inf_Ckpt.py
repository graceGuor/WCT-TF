from __future__ import division, print_function

import os
import argparse
import numpy as np
import tensorflow as tf
from utils import preserve_colors_np
from utils import get_files, get_img, get_img_crop, save_img, resize_to, center_crop
import scipy
import time
from wct import WCT

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_from', type=str,
                    help='Checkpoint save dir in inference phase, restore graph from this path',
                    default="models/inf")
parser.add_argument('--checkpoint_to', type=str,
                    help='save checkpoint to this path',
                    default="models/inf_inf")
# parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
# parser.add_argument('--relu-targets', nargs='+', type=str,
#                     help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--vgg-path', type=str, help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7')
parser.add_argument('--content-path', type=str, dest='content_path', help='Content image or folder of images')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image or folder of images')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/gpu:0')
parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512",
                    default=0)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=0)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)
parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
parser.add_argument('-r', '--random', type=int, help="Choose # of random subset of images from style folder", default=0)
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
parser.add_argument('--adain', action='store_true', help="Use AdaIN instead of WCT", default=False)

## Style swap args
parser.add_argument('--swap5', action='store_true', help="Swap style on layer relu5_1", default=False)
parser.add_argument('--ss-alpha', type=float, help="Style swap alpha blend", default=0.6)
parser.add_argument('--ss-patch-size', type=int, help="Style swap patch size", default=3)
parser.add_argument('--ss-stride', type=int, help="Style swap stride", default=1)

# Misc Parameters
parser.add_argument('--allow_soft_placement', help="Allow device soft device placement", default=True)
parser.add_argument('--log_device_placement', help="Log placement of ops on devices", default=False)
parser.add_argument('--gpu_fraction', type=float, help="gpu fraction", default=0.9)

args = parser.parse_args()


def postprocess(image):
    return np.uint8(np.clip(image, 0, 1) * 255)


def preprocess(image):
    if len(image.shape) == 3:  # Add batch dimension
        image = np.expand_dims(image, 0)
    return image / 255.  # Range [0,1]


def main():
    start = time.time()

    session_conf = tf.ConfigProto(
        allow_soft_placement=args.allow_soft_placement,
        log_device_placement=args.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction

    graph = tf.Graph()
    with tf.Session(config=session_conf, graph=graph) as sess:
        # Load the saved meta graph and restore variables
        checkpoint_dir = tf.train.latest_checkpoint(args.checkpoint_from)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_dir))
        saver.restore(sess, checkpoint_dir)
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.tables_initializer())

        content_image_path = "/Users/grace/data/style_transfer_test_cases/content/cat.jpg"
        content_image_path = "/Users/grace/data/style_transfer_test_cases/content/golden_gate.jpg"  # error
        content_image_path = "/Users/grace/data/style_transfer_test_cases/content/man-playing-tennis.jpg"
        style_image_path = "/Users/grace/data/style_transfer_test_cases/styles/the-starry-night.jpg"
        style_image_path = "/Users/grace/data/style_transfer_test_cases/styles/tiger.png"

        content_image_path = "../../data/style_transfer/cases/content/tubingen.jpg"
        style_image_path = "../../data/style_transfer/cases/styles/the-starry-night.jpg"
        style_image_path = "../../data/style_transfer/cases/styles/tiger.png"
        style_image_path = "../../data/style_transfer/cases/styles/colorful-girl.png"

        # 读取测试图片content_img和style_img
        content_img = get_img(content_image_path)
        style_img = get_img(style_image_path)

        content_img = preprocess(content_img)
        style_img = preprocess(style_img)

        content_prefix, content_ext = os.path.splitext(content_image_path)
        content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext
        style_prefix, _ = os.path.splitext(style_image_path)
        style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

        # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
        # out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))

        # stylized = sess.run(["encoder_decoder_relu5_1/decoder_relu5_1/decoder_model_relu5_1/relu5_1_16/relu5_1_16/BiasAdd:0"],
        #          feed_dict={"encoder_decoder_relu5_1/content_encoder_relu5_1/content_imgs:0": content_img,
        #                     "style_img:0": style_img})
        # stylized = postprocess(stylized.reshape(stylized.shape[1:]))
        # save_img(out_f, stylized)

        stylized_imgs = sess.run(
            ["encoder_decoder_relu5_1/decoder_relu5_1/decoder_model_relu5_1/relu5_1_16/relu5_1_16/BiasAdd:0",
             "encoder_decoder_relu4_1/decoder_relu4_1/decoder_model_relu4_1/relu4_1_11/relu4_1_11/BiasAdd:0",
             "encoder_decoder_relu3_1/decoder_relu3_1/decoder_model_relu3_1/relu3_1_6/relu3_1_6/BiasAdd:0",
             "encoder_decoder_relu2_1/decoder_relu2_1/decoder_model_relu2_1/relu2_1_3/relu2_1_3/BiasAdd:0",
             "encoder_decoder_relu1_1/decoder_relu1_1/decoder_model_relu1_1/relu1_1_1/relu1_1_1/BiasAdd:0"],
            feed_dict={"encoder_decoder_relu5_1/content_encoder_relu5_1/content_imgs:0": content_img,
                       "style_img:0": style_img})
        print(content_prefix, style_prefix, content_ext)
        for i, stylized in enumerate(stylized_imgs):
            out_f = os.path.join(args.out_path,
                                 '{}_{}_{}{}'.format(content_prefix, style_prefix, int(len(stylized_imgs) - i), content_ext))
            print(type(stylized), stylized.dtype)
            print(stylized.shape)
            stylized = postprocess(stylized.reshape(stylized.shape[1:]))
            save_img(out_f, stylized)

        print("Wrote stylized output image to {} at {}".format(out_f,
                                                               time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))



if __name__ == '__main__':
    main()
