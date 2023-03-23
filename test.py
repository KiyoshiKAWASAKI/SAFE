import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import xlwt
import sys

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

            print(model)

            saver = tf.compat.v1.train.import_meta_graph(model)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))

            predictions = graph.get_operation_by_name("loss/predictions").outputs[0]
            accuracy = graph.get_operation_by_name("loss/accuracy").outputs[0]
            input_head = graph.get_operation_by_name("input_headline").outputs[0]
            input_body = graph.get_operation_by_name("input_body").outputs[0]
            input_image = graph.get_operation_by_name("input_image").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            batch_size = graph.get_operation_by_name("batch_size").outputs[0]


            all_predictions_fake, acc_fake  = \
                sess.run([predictions, accuracy], feed_dict={input_head: x_test_head,
                                                            input_body: x_test_body,
                                                            input_image: x_test_image,
                                                            input_y: y_test,
                                                            dropout_keep_prob: 1.0,
                                                            batch_size: len(y_test)})

            predictionss_click = tf.convert_to_tensor(all_predictions_fake)
            actuals_click = tf.argmax(y_test, 1)

            #  ----------------------------------------------------------------

            # for clickbait detection
            actuals = actuals_click
            predictionss = predictionss_click

            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predictions = tf.ones_like(predictionss)
            zeros_like_predictions = tf.zeros_like(predictionss)

            tp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictionss, ones_like_predictions)
                    ),
                    "float"
                )
            )

            tn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictionss, zeros_like_predictions)
                    ),
                    "float"
                )
            )

            fp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictionss, ones_like_predictions)
                    ),
                    "float"
                )
            )

            fn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictionss, zeros_like_predictions)
                    ),
                    "float"
                )
            )

            tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op])


            tpr = float(tp) / (float(tp) + float(fn))

            accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
            print('Clickbait: ')
            print('ACC. = ' + str(accuracy))

            precision = float(tp) / (float(tp) + float(fp))
            print('precision = ' + str(precision))

            recall = tpr
            print('recall = ' + str(recall))

            f1_score = (2 * (precision * recall)) / (precision + recall)
            print('f1_score = ' + str(f1_score))

            print('tp:' + str(tp))
            print("tn: " + str(tn))
            print('fp:' + str(fp))
            print("fn: " + str(fn))

            return [accuracy, precision, recall, f1_score, tp, tn, fp, fn, all_predictions_fake]


if __name__ == '__main__':
    selected_epoch = 20

    modelfolder = '/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/safe_model_FakeNewsNet_Dataset'
    outdir = '/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/safe_test_results_FakeNewsNet_Dataset'

    model = modelfolder + "/models" + str(selected_epoch) + ".meta"

    print('===============================================')
    print('load vectors and labels ... ')

    # Use our private data for test
    x_body = np.load("/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/"
                     "safe_data/jan01_jan02_2023_triplets_npy/all_words.npy")
    x_image = np.load("/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/"
                      "safe_data/jan01_jan02_2023_triplets_npy/all_imgs.npy")
    x_names = np.load("/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/"
                      "safe_data/jan01_jan02_2023_triplets_npy/all_names.npy")

    print("x_body: ", x_body.shape)
    print("x_image: ", x_image.shape)
    print("x_names: ", x_names.shape)

    nb_samples = x_body.shape[0]

    # Load training data but just use as place holders because the model needs these input
    x_head = np.load('/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/all_headlines.npy')
    y = np.load('/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/all_labels.npy')

    x_head = x_head[:nb_samples, :, :]
    y = y[:nb_samples, :]

    print("x_head (dummy): ", x_head.shape)
    print("y (dummy): ", y.shape)

    print('===============================================')
    print('Running test...')


    acc, pre, rec, f1, tp, tn, fp, fn, all_predictions_fake = test(x_head,
                                                                   x_body,
                                                                   x_image,
                                                                   y,
                                                                   modelfolder,
                                                                   model)












