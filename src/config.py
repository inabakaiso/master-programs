"""

Path, HyperParam

"""
import torch
import torch.backends.cudnn as cudnn

## 基となるjson file
json_train = "../input/annotations/stair_captions_v1.2_train_tokenized.json"
json_valid = "../input/annotations/stair_captions_v1.2_val_tokenized.json"

##　基となるイメージ画像
train_path = "../input/train_original/train2014"
valid_path = "../input/train_original/val2014"

## 画像のID(COCO...00029832.jpg)と,それにマッチするキャプションを格納するデータセット
caption_train = '../input/annotation_csv/train_caption.csv'
caption_valid = '../input/annotation_csv/valid_caption.csv'
caption_test = '../input/annotation_csv/test_caption.csv'
annotation_dir = '../input/annotation_csv'

## インストールした元画像は使わないデータが多いため、上記の整理したデータセットを基に使うdataをそれぞれ仕分ける
data_train = '../input/data/data_train.csv'
data_valid = '../input/data/data_valid.csv'
data_test = '../input/data/data_test.csv'
train_image = '../input/train2014'
val_image = '../input/val2014'
test_image = '../input/test2014'
train_image_path = '../input/train_original/train2014/{}'
valid_image_path = '../input/train_original/val2014/{}'
test_image_path = '../input/train_original/val2014/{}'

## キャプションに対して前処理をしたものを格納
cleaned_train = "../input/cleaned_caption/clean_train.csv"
cleaned_valid = "../input/cleaned_caption/clean_valid.csv"
cleaned_test = "../input/cleaned_caption/clean_test.csv"


## Hyper Paramなど
class CFG:
    # Data parameters
    data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
    data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Training parameters
    start_epoch = 0
    epochs = 120  # number of epochs to train_original for (if early stopping is not triggered)
    epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
    batch_size = 16
    num_workers = 12  # for data-loading; right now, only 1 works with h5py
    encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
    decoder_lr = 4e-4  # learning rate for decoder
    grad_clip = 5.  # clip gradients at an absolute value of
    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
    best_bleu4 = 0.  # BLEU-4 score right now
    print_freq = 100  # print training/validation stats every __ batches
    fine_tune_encoder = False  # fine-tune encoder?
    checkpoint = None  # path to checkpoint, None if none
    size = 224
