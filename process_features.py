import os
import numpy as np
import csv
import sys
import json


all_triplet_feature_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                           "safe_data/invasion_triplets_safe_output_feat"
result_save_path = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                   "safe_data/result.json"


def calculate_median(text_feature):
    """

    :param input_feature:
    :return:
    """

    return np.median(text_feature, axis=0)



if __name__ == '__main__':
    all_triplets = os.listdir(all_triplet_feature_path)

    result_dict = {}

    for one_triplet in all_triplets:
        result_dict[one_triplet] = None

    for one_triplet in all_triplets:
        print("Processing one triplet: ", one_triplet)
        triplet_contents_path = os.path.join(all_triplet_feature_path, one_triplet)

        text_features = np.load(os.path.join(triplet_contents_path, "text_feature.npy"))
        img_features = np.load(os.path.join(triplet_contents_path, "img_feature.npy"))
        entry_names = np.load(os.path.join(triplet_contents_path, "file_names.npy"))

        # Skip one triplet if it has less than 3 samples
        if text_features.shape[0] < 3:
            continue

        else:
            median_vec = calculate_median(text_features)

            dist_vector = []

            # calculate distance for each sample
            for i in range(text_features.shape[0]):
                one_feature = text_features[i, :]
                dist = np.linalg.norm(one_feature - median_vec)
                dist_vector.append(dist)

            result_dict[one_triplet] = {"samples": entry_names.tolist(),
                                        "median": median_vec.tolist(),
                                        "distances": dist_vector}

    with open(result_save_path, 'w') as file:
        json.dump(result_dict, file)
        print("Saving file to %s" % result_save_path)



