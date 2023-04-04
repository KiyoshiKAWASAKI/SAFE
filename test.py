import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import xlwt
import sys
import os

def test(x_test_head, x_test_body, x_test_image,  y_test, model_dir, model):

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.compat.v1.Session(config=session_conf)

        with sess.as_default():
            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())

            saver = tf.compat.v1.train.import_meta_graph(model)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

            all_vectors = graph.get_operation_by_name("prediction/combine_all").outputs[0]
            double_img = graph.get_operation_by_name("calculate_cos_simi/combine_image").outputs[0]
            input_head = graph.get_operation_by_name("input_headline").outputs[0]
            input_body = graph.get_operation_by_name("input_body").outputs[0]
            input_image = graph.get_operation_by_name("input_image").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            batch_size = graph.get_operation_by_name("batch_size").outputs[0]


            all_vectors, double_img  = \
                sess.run([all_vectors, double_img], feed_dict={input_head: x_test_head,
                                                            input_body: x_test_body,
                                                            input_image: x_test_image,
                                                            input_y: y_test,
                                                            dropout_keep_prob: 1.0,
                                                            batch_size: len(y_test)})

            body_vector = all_vectors[:, 32:64]
            img_vector = all_vectors[:, 64:]

            print(body_vector.shape)
            print(img_vector.shape)



if __name__ == '__main__':
    modelfolder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/04012023"
    outdir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/04012023"

    model = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/" \
            "04012023/model_99.meta"

    input_npy_folder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                       "safe_data/invasion_triplets_safe_input_npy"
    save_feature_folder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                          "safe_data/invasion_triplets_safe_output_feat"

    # TODO: test each triplet separately

    all_triplets = os.listdir(input_npy_folder)

    for one_triplet in all_triplets:
        print("*" * 50)
        print("Processing: ", one_triplet)

        # Real data:
        x_body = np.load(input_npy_folder + "/" + one_triplet + "/all_words.npy")
        x_image = np.load(input_npy_folder + "/" + one_triplet + "/all_imgs.npy")
        x_names = np.load(input_npy_folder + "/" + one_triplet + "/all_names.npy")

        nb_samples = x_body.shape[0]

        # Dummy data
        # Load training data but just use as place holders because the model needs these input
        x_head = np.load('/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/all_headlines.npy')
        y = np.load('/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/all_labels.npy')

        x_head = x_head[:nb_samples, :, :]
        y = y[:nb_samples, :]

        test(x_head, x_body, x_image, y, modelfolder, model)





