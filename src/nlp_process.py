from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch
from pickle import load
import pandas as pd
import config
import re
import mojimoji
import string

class Clean_text:
    """
    文字の正規化等
    """
    def __init__(self, file_path, output_file):

        self.dataset = pd.read_csv(file_path)
        self.output = output_file

    def clean_preprocess(self):
        df = self.dataset
        df["image_caption"] = df["image_caption"].apply(self.lower_text)
        df["image_caption"] = df["image_caption"].apply(self.normalize_number)
        df["image_caption"] = df["image_caption"].apply(self.translate_pure_word)
        df["image_caption"] = df["image_caption"].apply(self.lower_text)
        df["image_caption"] = df["image_caption"].apply(self.translate_punctual)

        df.to_csv(self.output, index=False)


    def lower_text(self, text):
        return text.lower()

    def normalize_number(self, text):
        tmp = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text)
        replaced_text = re.sub(r'\d+', '0', tmp)
        return replaced_text

    def translate_pure_word(self, text):
        return mojimoji.zen_to_han(text)

    def translate_punctual(self, text):
        table = str.maketrans('', '', string.punctuation)  # 記号をリストアップ
        text = text.split()
        text = [w.translate(table) for w in text]
        return ' '.join(text)


class Tokenizer(object):
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')

        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}

    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def texts_to_sequences(self, texts):
        sequence = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequence.append(sequence)
        return sequence

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.stoi[i], sequence)))

    def sequence_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


def bms_collate(batch):
    tokenizer = torch.load('../model/tokenizer/tokenizer.pth')
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first = True, padding_value = tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)

if __name__ == '__main__':
    caption = pd.read_csv('../input/cleaned_caption/clean_train.csv')['image_caption']
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(caption)
    torch.save(tokenizer, '../model/tokenizer/tokenizer.pth')
    print('Saved tokenizer')




