"""
Prepare data for training:
   x_head = np.load('../pf_embedding/case_headline.npy')
   x_body = np.load('../pf_embedding/case_body.npy')
   x_image = np.load('../pf_embedding/case_image.npy')
"""

import numpy as np
import json
import os
import sys
from tqdm import tqdm

# For FakeNewsNet
# article_folder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
#                  "safe_data/FakeNewsNet_Dataset_processed"
# article_subfolder = ["gossipcop_fake",
#                      "gossipcop_real",
#                      "politifact_fake",
#                      "politifact_real"]
#
# image_sif_folder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
#                    "safe_data/FakeNewsNet_Dataset_captions_sif"
# image_sif_subfolder = ["gossipcop",
#                        "politifact"]

# original_img_folder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/NewsImages"
# original_img_subfolder = ["gossipcop_images", "politifact_images"]

# result_save_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data"

# For ICWSM
article_sif_folder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                     "safe_data/jan01_jan02_2023_triplets_sif/"
image_sif_folder = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                   "safe_data/jan01_jan02_2023_triplets_captions_sif"
result_save_dir = "/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/" \
                  "safe_data/jan01_jan02_2023_triplets_npy"




def process_one_embedding(embedding,
                          required_size,
                          embedding_length=300):
    """
    Process a give embedding to the required shape

    :param embedding:
    :return:
    """
    # Check whether the size of the embedding is the same as required
    if embedding.shape[0] > required_size:
        # Case 1: Caption is longer than required length - only take first n
        embedding = embedding[:required_size, :]

    elif embedding.shape[0] < required_size:
        # Case 2: Caption is shorter than required length - fill up to n with zeroes
        filler = np.zeros((required_size - embedding.shape[0], embedding_length))
        embedding = np.concatenate((embedding, filler))

    assert (embedding.shape[0] == required_size) and (embedding.shape[1] == embedding_length)

    return embedding




def generate_safe_training_npy(article_folder,
                          article_subfolder,
                          image_caption_folder,
                          head_size=30,
                          body_size=100,
                          image_size=20,
                          embedding_length=300):
    """
    Preprocess FakeNewNet dataset
    Jointly process headline, body and image files.
    There are more articles than images, so use the article to find img captions

    :param article_folder:
    :param article_subfolder:
    :param image_caption_folder:
    :param image_caption_subfoler:
    :param image_folder:
    :return: headline.npy, body.npy, image.npy
    """

    all_headline = None
    all_body = None
    all_img = None
    all_labels = None

    # TODO: process 4 categories, respectively
    for one_subfolder in article_subfolder:
        dir_full_path = os.path.join(article_folder, one_subfolder)
        print("Processing: ", dir_full_path)

        dataset_name = one_subfolder.split("_")[0]
        label_name = one_subfolder.split("_")[1]

        # List all samples in this category
        all_samples = os.listdir(dir_full_path)

        # TODO: Process each sample
        for i in tqdm(range(len(all_samples))):
            one_sample = all_samples[i]
            # 0. Get full path for all files
            one_sample_dir = os.path.join(dir_full_path, one_sample)

            # 1. Check whether it is gossipcop or politifact
            if dataset_name == "gossipcop":
                img_id = one_sample.split("-")[-1]
            else:
                img_id = one_sample[10:]

            # 2. Check whether this sample has corresponding img caption embedding
            target_img_sif_dir = image_caption_folder + "/" + dataset_name + "/" + dataset_name
            target_img_sif_path = target_img_sif_dir + "/" + img_id + "_caption_sif_embedding.npy"

            if os.path.isfile(target_img_sif_path):
                # TODO: Process img caption embedding
                img_embedding = np.load(target_img_sif_path)

                img_embedding = process_one_embedding(embedding=img_embedding,
                                                      required_size=image_size)
                img_embedding = np.reshape(img_embedding,
                                           (1, img_embedding.shape[0], img_embedding.shape[1]))

                if all_img is None:
                    all_img = img_embedding
                else:
                    all_img = np.concatenate((all_img, img_embedding))


                # TODO: process headline embeddings to the desired length
                headline_embedding = np.load(os.path.join(one_sample_dir, "headline.npy"))
                headline_embedding = process_one_embedding(embedding=headline_embedding,
                                                           required_size=head_size)
                headline_embedding = np.reshape(headline_embedding,
                                                (1, headline_embedding.shape[0], headline_embedding.shape[1]))

                if all_headline is None:
                    all_headline = headline_embedding
                else:
                    all_headline = np.concatenate((all_headline, headline_embedding))

                # TODO: process body embeddings to the desired length
                body_embedding = np.load(os.path.join(one_sample_dir, "body.npy"))
                body_embedding = process_one_embedding(embedding=body_embedding,
                                                       required_size=body_size)
                body_embedding = np.reshape(body_embedding,
                                            (1, body_embedding.shape[0], body_embedding.shape[1]))

                if all_body is None:
                    all_body = body_embedding
                else:
                    all_body = np.concatenate((all_body, body_embedding))

                # TODO: process data label
                if label_name == "real":
                    one_label = np.array([[1, 0]])
                else:
                    one_label = np.array([[0, 1]])

                if all_labels is None:
                    all_labels = one_label
                else:
                    all_labels = np.concatenate((all_labels, one_label))

            # Go to the next sample if there is no img embedding
            else:
                continue

    print("Headline: ", all_headline.shape)
    print("Body: ", all_body.shape)
    print("Image caption: ", all_img.shape)
    print("Labels: ", all_labels.shape)

    # Save all npy to disk
    np.save(os.path.join(result_save_dir, "all_headlines.npy"), all_headline)
    np.save(os.path.join(result_save_dir, "all_body.npy"), all_body)
    np.save(os.path.join(result_save_dir, "all_imgs.npy"), all_img)
    np.save(os.path.join(result_save_dir, "all_labels.npy"), all_labels)




def prepare_icwsm_data(img_caption_sif_dir,
                       word_sif_dir,
                       save_dir,
                       body_size=100,
                       image_size=20):
    """
    Preprocess our private data

    :param img_caption_sif_dir:
    :param word_sif_dir:
    :return:
    """
    all_img = None
    all_words = None
    all_names = []

    # TODO: Find all subfolders of triplets
    all_triplets = os.listdir(img_caption_sif_dir)
    all_triplets.sort()

    for one_triplet in all_triplets:
        print("*" * 50)
        print("Processing one triplet: ", one_triplet)
        one_img_caption_folder = os.path.join(img_caption_sif_dir, one_triplet)
        one_word_folder = os.path.join(word_sif_dir, one_triplet)

        # TODO: Find all the files in a triplet folder
        all_img_files = os.listdir(one_img_caption_folder)
        all_word_files = os.listdir(one_word_folder)

        # Process files into dictionary
        all_img_dict = {}
        all_word_dict = {}

        for one_file in all_img_files:
            one_file_id = one_file.split("_")[1]
            all_img_dict[one_file_id] = one_file

        for one_file in all_word_files:
            one_file_id = one_file.split("_")[1]
            all_word_dict[one_file_id] = one_file

        for common_id in set(all_img_dict).intersection(set(all_word_dict)):
            # Get the name/ID of the post first
            all_names.append([os.path.join(one_img_caption_folder, all_img_dict[common_id]),
                              os.path.join(one_word_folder, all_word_dict[common_id])])

            # Process image caption embeddings
            img_embedding = np.load(os.path.join(one_img_caption_folder, all_img_dict[common_id]))
            img_embedding = process_one_embedding(embedding=img_embedding,
                                                  required_size=image_size)
            img_embedding = np.reshape(img_embedding,
                                        (1, img_embedding.shape[0],img_embedding.shape[1]))

            if all_img is None:
                all_img = img_embedding
            else:
                all_img = np.concatenate((all_img, img_embedding))

            # Process post content embeddings
            word_embedding = np.load(os.path.join(one_word_folder, all_word_dict[common_id]))
            word_embedding = process_one_embedding(embedding=word_embedding,
                                                  required_size=body_size)
            word_embedding = np.reshape(word_embedding,
                                       (1, word_embedding.shape[0], word_embedding.shape[1]))

            if all_words is None:
                all_words = word_embedding
            else:
                all_words = np.concatenate((all_words, word_embedding))


    # Save everything to file
    print("Image embedding: ", all_img.shape)
    print("Word embedding: ", all_words.shape)
    print("Triplet IDs: ", len(all_names))

    np.save(os.path.join(save_dir, "all_imgs.npy"), all_img)
    np.save(os.path.join(save_dir, "all_words.npy"), all_words)

    all_names = np.asarray(all_names)
    np.save(os.path.join(save_dir, "all_names.npy"), all_names)





if __name__ == '__main__':
    # generate_safe_training_npy(article_folder=article_folder,
    #                       article_subfolder=article_subfolder,
    #                       image_caption_folder=image_sif_folder)

    prepare_icwsm_data(img_caption_sif_dir=image_sif_folder,
                       word_sif_dir=article_sif_folder,
                       save_dir=result_save_dir)