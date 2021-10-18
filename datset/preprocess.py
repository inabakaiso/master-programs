import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
import shutil
import sys, os
import string

sys.path.append(os.path.join('../', 'src'))
import config
import csv


def convert_csv(read_file, *out_file):
    """
    :param read_file:
    :param out_file:
    :return:
    """
    f = open(read_file, "r", encoding="utf-8_sig")
    json_data = json.load(f)
    print("completed load json's file")

    id_file = {}
    for i in range(len(json_data["images"])):
        file_name = json_data["images"][i]["file_name"]
        id = json_data["images"][i]["id"]
        id_file[id] = file_name

    annotation_size = len(json_data["annotations"])
    split_num = len(out_file)
    length = annotation_size // split_num
    df = pd.DataFrame(np.zeros((annotation_size, 3)), columns=['image_id', 'file_name', 'image_caption'])
    print("created dataset")
    for j in range(annotation_size):
        image_id = json_data["annotations"][j]["image_id"]
        file_name = id_file[image_id]
        caption = json_data["annotations"][j]["tokenized_caption"]
        df.loc[j, 'image_id'] = image_id
        df.loc[j, 'file_name'] = file_name
        df.loc[j, 'image_caption'] = caption
    df.sort_values('image_id')
    df = df.reset_index(drop=True)
    if not os.path.exists(config.annotation_dir):
        os.makedirs(config.annotation_dir)
    print("annotation")
    for i in range(split_num):
        df[length * i:length * (i + 1)].to_csv(out_file[i], index=False)
    print("convert finish")


def image_preprocess(df_path, out_file, origin_image, new_dir):
    """
    :param df_path: caption dataset {file_list, caption}
    :param out_file:
    :param origin_image:
    :param new_dir:
    :return:
    """
    df = pd.read_csv(df_path)

    image_list = []
    for idx, data_row in df.iterrows():
        image_id, image_desc = data_row['file_name'], data_row['image_caption']
        if image_id not in image_list:
            image_list.append(image_id)
    print(f"finish list up : {image_list[0]}")

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    print("maked new directory")

    ## image_list に散在する画像ファイルを new dirにcopyしている
    for list in image_list:
        shutil.copy(origin_image.format(list), new_dir)

    data = pd.Series(image_list)
    data.to_csv(out_file, index=False)
    print("finish")


def calc_max_length(df):
    return max(len(t) for t in df['image_caption'].split())


def total_num(df):
    total = 0
    for caption in df['image_caption']:
        total += len(caption.split()) - 1
    return total


def remove_few_words(tokenizer, freq):
    """
    :param df: dataset
    :param tokenizer: Tokenizer
    :param freq: 文字の出現回数
    :return:
    """

    remove_list = list()
    for word in tokenizer.word_index().keys():
        word_count = tokenizer.word_counts[word]
        if word_count <= freq:
            remove_list.append(word)
    return remove_list



