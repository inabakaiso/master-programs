import torch
import torch.nn as nn
import torchvision
import torch.functional as F
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.encoded_image_size = encoded_image_size

        resnet = torchvision.models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        CNN Encoder
        :param images:
        :return:
        """
        out = self.cnn(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)

        return out

    def fine_tune(self, fine_tune=True):
        """
        train_original or valid and test_original
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune:
        :return:
        """
        for p in self.cnn.parameters():
            p.requires_grad = False

        for c in list(self.cnn.children())[5:]:
            p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: features size of encoded_images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """

        :param encoder_out:
        :param decoder_hidden:
        :return:
        """
        att1 = self.encoder_att(encoder_out)  # out (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # out (batch_size, attention_dim)
        ## squeeze => 次元を作る. unsqueeze =>　次元削減
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encodeing = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encodeing, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim,
                 vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim:
        :param embed_dim:
        :param decoder_dim:
        :param vocab_size:
        :param encoder_dim:
        :param dropout:
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(attention_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        ## LSTMCell(input size: float, hidden size: float, bias: boolean)
        ## output h_1, c_1
        self.decoder_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embedding(self, embedding):
        """
        :param embedding: pretrained embedding
        :return:
        """
        self.embedding.weight = nn.Parameter(embedding)

    def fine_tune_embedding(self, fine_tune=True):
        """
        :param fine_tune:
        :return:
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out:
        :return:
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)

        return h, c

    def forward(self, encoder_out, encoded_captions, caption_length):
        """
        Forward propagation.
        :param encoder_out:
        :param encoded_captions:
        :param caption_length:
        :return:
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten Image # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing length
        caotion_lengths, sort_ind = caption_length.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_length = (caotion_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_length), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_length), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_length)):
            batch_size_t = sum([l > t for l in decode_length])
            attention_weighted_encodeing, alpha = self.attention(encoder_out[:batch_size_t],
                                                                 h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encodeing = gate + attention_weighted_encodeing
            h, c = self.decoder_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encodeing], dim=1),
                (h[:batch_size_t], c[batch_size_t])
            )
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_length, alphas, sort_ind

    def predict(self, encoder_out, decode_length, tokenizer):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1) ## the number of pixels; 600000
        start_tokens = torch.ones(batch_size, dtype=torch.long, device=device) * tokenizer.stoi["<sos>"]
        embeddings = self.embedding(start_tokens)
        h, c = self.init_hidden_state(encoder_out)
        predictions = torch.zeros(batch_size, decode_length, vocab_size).to(device)
        end_condition = torch.zeros(batch_size, dtype=torch.long).to(device)
        for t in range(decode_length):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            ## LSTMCell
            ## torch.cat dimに沿ってベクトルを結合
            h, c = self.decoder_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1)
            )

class MultiLayerLstmWithAttention:
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout, num_layers):
        """
        :param attention_dim:
        :param embed_dim:
        :param decoder_dim:
        :param vocab_size:
        :param encoder_dim:
        :param dropout:
        :param num_layers:
        """
        super(MultiLayerLstmWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decoder_step = nn.ModuleList([nn.LSTMCell(embed_dim + encoder_dim
                                                       if layer == 0
                                                       else (embed_dim, embed_dim))
                                           for layer in range(self.num_layers)])
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = [self.init_h(mean_encoder_out) for i in range(self.num_layers)]
        c = [self.init_c(mean_encoder_out) for i in range(self.num_layers)]
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        :param self:
        :param encoder_out:
        :param encoded_captions:
        :param caption_lengths:
        :return:
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]

        # embedding transformed sequence for vector
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state, initialize cell_vector and hidden_vector
        prev_h, prev_c = self.init_hidden_state(encoder_out)

        # set decode length by caption length - 1 because of omitting start token
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size, device=self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels, device=self.device)

        # predict sequence
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[batch_size_t],
                                                                prev_h[-1][:batch_size_t])
            gate = self.sigmoid(self.f_beta(prev_h[-1][:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            for i, rnn in enumerate(self.decode_step):
                # hidden vector and cell vector
                h, c = rnn(input, (prev_h[i][:batch_size_t], prev_c[i][:batch_size_t]))
                input = self.dropout
                # save state for next time step
                prev_h[i] = h
                prev_c[i] = c

            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def predict(self, encoder_out, decode_lengths, tokenizer):

        # size variables
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # embed start tocken for LSTM input
        start_tockens = torch.ones(batch_size, dtype=torch.long, device=self.device) * tokenizer.stoi['<sos>']
        embeddings = self.embedding(start_tockens)

        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        predictions = torch.zeros(batch_size, decode_lengths, vocab_size, device=self.device)

        # predict sequence
        end_condition = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for t in range(decode_lengths):
            awe, alpha = self.attention(encoder_out, h[-1])  # (s, encoder_dim), (s, num_pixels)
            gate = self.sigmoid(self.f_beta(h[-1]))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            input = torch.cat([embeddings, awe], dim=1)

            for j, rnn in enumerate(self.decode_step):
                at_h, at_c = rnn(input, (h[j], c[j]))  # (s, decoder_dim)
                input = self.dropout(at_h)
                h[j] = at_h
                c[j] = at_c

            preds = self.fc(self.dropout(h[-1]))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            end_condition |= (torch.argmax(preds, -1) == tokenizer.stoi["<eos>"])
            if end_condition.sum() == batch_size:
                break
            embeddings = self.embedding(torch.argmax(preds, -1))

        return predictions

    # beam search
    def forward_step(self, prev_tokens, hidden, encoder_out, function):

        h, c = hidden
        # h, c = h.squeeze(0), c.squeeze(0)
        h, c = [hi.squeeze(0) for hi in h], [ci.squeeze(0) for ci in c]

        embeddings = self.embedding(prev_tokens)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)

        awe, alpha = self.attention(encoder_out, h[-1])  # (s, encoder_dim), (s, num_pixels)
        gate = self.sigmoid(self.f_beta(h[-1]))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        input = torch.cat([embeddings, awe], dim=1)
        for j, rnn in enumerate(self.decode_step):
            at_h, at_c = rnn(input, (h[j], c[j]))  # (s, decoder_dim)
            input = self.dropout(at_h)
            h[j] = at_h
            c[j] = at_c

        preds = self.fc(self.dropout(h[-1]))  # (batch_size_t, vocab_size)

        # hidden = (h.unsqueeze(0), c.unsqueeze(0))
        hidden = [hi.unsqueeze(0) for hi in h], [ci.unsqueeze(0) for ci in c]
        predicted_softmax = function(preds, dim=1)

        return predicted_softmax, hidden, None
