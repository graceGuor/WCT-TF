#! /usr/bin/env python

import tensorflow as tf
from tensorflow import graph_util as graph_util
import os
import shutil


# Parameters
# ==================================================
# Build pb model
tf.flags.DEFINE_string("pb_save_path", "models/pbs", "pb save file.")
tf.flags.DEFINE_integer("model_version", 2, "emoji prediction model version")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 200, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "models/inf", "Checkpoint directory from training run")
tf.flags.DEFINE_string("relu_target", "relu5_1", "Relu target layers corresponding to decoder checkpoints")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_fraction", 0.9, "gpu fraction (default: 0.5)")


FLAGS = tf.flags.FLAGS


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = "encoder_decoder_relu5_1/decoder_relu5_1/decoder_model_relu5_1/relu5_1_16/relu5_1_16/BiasAdd," \
                        "style_encoder/model_2/relu5_1/Relu"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    for n in tf.get_default_graph().as_graph_def().node:
        print("---"*3, n.name)

    with tf.Session() as sess:
        for n in sess.graph_def.node:
            print('**', n.name)
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据

        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        output_file = os.path.join(output_graph, 'wct.pb')
        with tf.gfile.GFile(output_file, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点


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
        # print(graph.get_operations())
        # sess.run(tf.tables_initializer())

        # Get the placeholders from the graph by name

        # Tensors we want to evaluate
        # 定义输入的张量名称,对应网络结构的输入张量
        content_input_tensor = sess.graph.get_tensor_by_name(
            "encoder_decoder_relu5_1/content_encoder_relu5_1/content_imgs:0")
        style_input_tensor = sess.graph.get_tensor_by_name("style_img:0")
        alpha_input_tensor = sess.graph.get_tensor_by_name("alpha:0")

        # 定义输出的张量名称
        stylized_image_tensor = sess.graph.get_tensor_by_name(
            "encoder_decoder_relu1_1/decoder_relu1_1/decoder_model_relu1_1/relu1_1_1/relu1_1_1/BiasAdd:0")

        with tf.device('/cpu:0'):
            builder = tf.saved_model.builder.SavedModelBuilder(version_export_path)

            tensor_input_content = tf.saved_model.utils.build_tensor_info(content_input_tensor)
            tensor_input_style = tf.saved_model.utils.build_tensor_info(style_input_tensor)
            tensor_input_alpha = tf.saved_model.utils.build_tensor_info(alpha_input_tensor)

            tensor_output_stylized = tf.saved_model.utils.build_tensor_info(stylized_image_tensor)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input_content': tensor_input_content,
                            'input_style': tensor_input_style},
                            # 'input_alpha': tensor_input_alpha},
                    outputs={'output_stylized': tensor_output_stylized},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_words': prediction_signature
                },
                clear_devices=True
            )
            builder.save()
        print('Done exporting!')


if __name__ == '__main__':

    checkpoint_dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    model_version = FLAGS.model_version
    version_export_path = os.path.join(FLAGS.pb_save_path + "/" + str(model_version))
    if os.path.exists(version_export_path):
        shutil.rmtree(version_export_path)
    # os.makedirs(version_export_path)

    print("Exporting trained model from ", checkpoint_dir)
    print("Exporting trained model to ", version_export_path)

    # freeze_graph(checkpoint_dir, version_export_path)
    freeze_graph_tfserving(checkpoint_dir, version_export_path)

    print("Finished!")

    print()
