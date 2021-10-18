import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
import random
from matplotlib import pyplot as plt
import config
import model
import nlp_process
import pandas as pd
from torch.utils.data import DataLoader
from utils import *
from dataset import *
from model import *
from nlp_process import *
import time
torch.backends.cudnn.benchmark = True
CFG = config.CFG()

def epoch_per_loss(epoch, tra_loss, val_loss):
    plt.plot(range(epoch), tra_loss, 'b', label='training_loss')
    plt.plot(range(epoch), val_loss, 'r', label='validation_loss')
    plt.title("Train And Valid Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    :param train_loader: DataLoader for training model
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weight
    :param decoder_optimizer: optimizer to update decoder's weight
    :param epoch: epoch number
    :return:
    """
    decoder.train()
    encoder.train()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(scores, targets)

        loss += CFG.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backword()

        if CFG.grad_clip is not None:
            clip_gradient(decoder_optimizer, CFG.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, CFG.grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

            # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % CFG.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    :param val_loader:
    :param encoder:
    :param decoder:
    :param criterion:
    :return:
    """

    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()
    predictions = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in val_loader:
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_length, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores_copy = scores.copy()
            scores, _ = pack_padded_sequence(scores, decode_length, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_length, batch_first=True)

            loss = criterion(scores, targets)

            loss += CFG.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_length))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_length))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % CFG.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
    return


if __name__ == "__main__":
    if CFG.checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=CFG.attention_dim,
                                       embed_dim=CFG.emb_dim,
                                       decoder_dim=CFG.decoder_dim,
                                       vocab_size=300,
                                       dropout=CFG.dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=CFG.decoder_lr)

        encoder = Encoder(encoded_image_size=14)
        encoder.fine_tune(fine_tune=CFG.fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=CFG.encoder_lr) if CFG.fine_tune_encoder else None

    else:
        checkpoint = torch.load(CFG.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epoch_since_improvement = checkpoint['epoch_since_improvement']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if CFG.fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune=CFG.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=CFG.encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    # train_loader = torch.utils.data.DataLoader(
    #     CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
    #     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(
    #     CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
    #     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    df = pd.read_csv('../input/cleaned_caption/clean_train.csv')

    tokenizer = torch.load('../model/tokenizer/tokenizer.pth')

    train_dataset = CaptionDataset(df, tokenizer, transform=get_transforms(data='train'))
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=bms_collate)

    train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, CFG.epochs)
