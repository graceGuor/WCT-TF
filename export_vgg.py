#! /usr/bin/env python

import tensorflow as tf
import os
import shutil


# Parameters
# ==================================================

# Build pb model
tf.flags.DEFINE_boolean("is_build_pb", True, "if build pb language model")
tf.flags.DEFINE_string("pb_save_path", "models/pbs", "pb save file.")
tf.flags.DEFINE_integer("model_version", 1, "emoji prediction model version")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "models/vgg", "Checkpoint directory from training run")
tf.flags.DEFINE_string("relu_target", "relu5_1", "Relu target layers corresponding to decoder checkpoints")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_fraction", 0.9, "gpu fraction (default: 0.5)")


FLAGS = tf.flags.FLAGS


def freeze_graph_tfserving(checkpoint_file, version_export_path):
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction

    graph = tf.Graph()

    with tf.Session(config=session_conf, graph=graph) as sess:
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # for n in tf.get_default_graph().as_graph_def().node:
        #     print("---" * 3, n.name)
        # 定义输入的张量名称,对应网络结构的输入张量
        input_tensor = sess.graph.get_tensor_by_name("vgg_input:0")

        # 定义输出的张量名称
        output_tensor = sess.graph.get_tensor_by_name("relu4_1/Relu:0")

        input_info = tf.saved_model.utils.build_tensor_info(input_tensor)
        output_info = tf.saved_model.utils.build_tensor_info(output_tensor)

        builder = tf.saved_model.builder.SavedModelBuilder(version_export_path)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input': input_info},
                outputs={'output': output_info},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_words': prediction_signature
            }
        )
        builder.save()
        print('Done exporting!')


def freeze_graph_tfserving_simpleSave(checkpoint_file, version_export_path):
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction

    graph = tf.Graph()

    with tf.Session(config=session_conf, graph=graph) as sess:
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # for n in tf.get_default_graph().as_graph_def().node:
        #     print("---" * 3, n.name)
        # 定义输入的张量名称,对应网络结构的输入张量
        input_tensor = sess.graph.get_tensor_by_name("vgg_input:0")

        # 定义输出的张量名称
        output_tensor = sess.graph.get_tensor_by_name("relu4_1/Relu:0")

        inputs_dict = {
            'input': input_tensor
        }
        outputs_dict = {
            'output': output_tensor
        }
        tf.saved_model.simple_save(
            sess, version_export_path, inputs_dict, outputs_dict
        )

        print('Done exporting!')


def freeze_graph_test(version_export_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        '''
        You can provide 'tags' when saving a model,
        in my case I provided, 'serve' tag 
        '''

        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], version_export_path)
        print("version_export_path:", version_export_path)

        # print your graph's ops, if needed
        print("-" * 100)
        print(graph.get_operations())


def save():
    checkpoint_dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    model_version = FLAGS.model_version
    version_export_path = os.path.join(FLAGS.pb_save_path + "/" + str(model_version))
    if os.path.exists(version_export_path):
        shutil.rmtree(version_export_path)

    print("Exporting trained model to ", version_export_path)
    freeze_graph_tfserving(checkpoint_dir, version_export_path)
    # freeze_graph_tfserving_simpleSave(checkpoint_dir, version_export_path)


def inference():
    model_version = FLAGS.model_version
    version_export_path = os.path.join(FLAGS.pb_save_path + "/" + str(model_version))
    freeze_graph_test(version_export_path)


if __name__ == '__main__':
    # save()
    inference()

    print("Finished!")
