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

parser.add_argument('--checkpoint', type=str,
                    help='Checkpoint save dir in inference phase',
                    default="models/inf")
parser.add_argument('--log-path', type=str,
                    dest='log_path', help='Logging dir path')
parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str,
                    help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
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


def main():
    start = time.time()

    session_conf = tf.ConfigProto(
        allow_soft_placement=args.allow_soft_placement,
        log_device_placement=args.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction

    with tf.Graph().as_default():
        # with tf.Session(config=session_conf) as sess:
        # Load the WCT model
        wct_model = WCT(checkpoints=args.checkpoints,
                        relu_targets=args.relu_targets,
                        vgg_path=args.vgg_path,
                        device=args.device,
                        ss_patch_size=args.ss_patch_size,
                        ss_stride=args.ss_stride)

        with wct_model.sess as sess:
            # 训练的时候不需要style_img，所以在inference的时候重新保存一次checkpoint
            saver = tf.train.Saver()
            log_path = args.log_path if args.log_path is not None else os.path.join(args.checkpoint, 'log')
            summary_writer = tf.summary.FileWriter(log_path, sess.graph)

            # 部分node没有保存在checkpoint中，需要重新初始化
            # sess.run(tf.global_variables_initializer())

            # Get content & style full paths
            if os.path.isdir(args.content_path):
                content_files = get_files(args.content_path)
            else:  # Single image file
                content_files = [args.content_path]
            if os.path.isdir(args.style_path):
                style_files = get_files(args.style_path)
                if args.random > 0:
                    style_files = np.random.choice(style_files, args.random)
            else:  # Single image file
                style_files = [args.style_path]

            os.makedirs(args.out_path, exist_ok=True)

            count = 0

            # Apply each style to each content image
            for content_fullpath in content_files:
                content_prefix, content_ext = os.path.splitext(content_fullpath)
                content_prefix = os.path.basename(content_prefix)  # Extract filename prefix without ext

                content_img = get_img(content_fullpath)
                if args.content_size > 0:
                    content_img = resize_to(content_img, args.content_size)

                for style_fullpath in style_files:
                    style_prefix, _ = os.path.splitext(style_fullpath)
                    style_prefix = os.path.basename(style_prefix)  # Extract filename prefix without ext

                    # style_img = get_img_crop(style_fullpath, resize=args.style_size, crop=args.crop_size)
                    # style_img = resize_to(get_img(style_fullpath), content_img.shape[0])

                    style_img = get_img(style_fullpath)

                    if args.style_size > 0:
                        style_img = resize_to(style_img, args.style_size)
                    if args.crop_size > 0:
                        style_img = center_crop(style_img, args.crop_size)

                    if args.keep_colors:
                        style_img = preserve_colors_np(style_img, content_img)

                    # if args.noise:  # Generate textures from noise instead of images
                    #     frame_resize = np.random.randint(0, 256, frame_resize.shape, np.uint8)
                    #     frame_resize = gaussian_filter(frame_resize, sigma=0.5)

                    # Run the frame through the style network
                    stylized_rgb = wct_model.predict(content_img, style_img, args.alpha, args.swap5, args.ss_alpha, args.adain)

                    if args.passes > 1:
                        for _ in range(args.passes - 1):
                            stylized_rgb = wct_model.predict(stylized_rgb, style_img, args.alpha, args.swap5, args.ss_alpha,
                                                             args.adain)

                    save_path = saver.save(sess, os.path.join(args.checkpoint, 'model.ckpt'))
                    print("Model saved in file: %s" % save_path)

                    # Stitch the style + stylized output together, but only if there's one style image
                    if args.concat:
                        # Resize style img to same height as frame
                        style_img_resized = scipy.misc.imresize(style_img, (stylized_rgb.shape[0], stylized_rgb.shape[0]))
                        # margin = np.ones((style_img_resized.shape[0], 10, 3)) * 255
                        stylized_rgb = np.hstack([style_img_resized, stylized_rgb])

                    # Format for out filename: {out_path}/{content_prefix}_{style_prefix}.{content_ext}
                    out_f = os.path.join(args.out_path, '{}_{}{}'.format(content_prefix, style_prefix, content_ext))
                    # out_f = f'{content_prefix}_{style_prefix}.{content_ext}'

                    # print(stylized_rgb, stylized_rgb.shape, type(stylized_rgb))
                    # print(out_f)
                    save_img(out_f, stylized_rgb)

                    count += 1
                    print("{}: Wrote stylized output image to {} at {}".format(count, out_f, time.time()))
                    print("breaking...")
                    break
                break

            print("Finished stylizing {} outputs in {}s".format(count, time.time() - start))


if __name__ == '__main__':
    main()
