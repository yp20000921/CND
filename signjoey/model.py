# coding: utf-8
import tensorflow as tf

# tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from lunanlp import torch_seed

import random

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder_at,TransformerDecoder_nat
from signjoey.search import beam_search, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import freeze_params
from torch import Tensor
from typing import Union

from torch.autograd import Variable
import torch



class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        nat_decoder,
        gloss_output_layer: nn.Module,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()

        self.encoder = encoder
        self.nat_decoder = nat_decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation


    def new_arange(self,x, *size):
        """
        Return a Tensor of `size` filled with a range function on the device of x.
        If size is empty, using the size of the variable x.
        """
        if len(size) == 0:
            size = x.size()
        return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

    # pylint: disable=arguments-differ
    def forward(
        self,
        rd,
        curriculum_type,
        nat_target,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        nat_txt_input = None,
        nat_txt_mask = None

    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """


        encoder_output2, encoder_hidden2 = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )

        x = encoder_output2

        predicted_lengths_logits = torch.matmul(x[:, 0, :], self.encoder.length_embed.weight.transpose(0, 1)).float()
        predicted_lengths_logits[:, 0] += float('-inf')  # Cannot predict the len_token
        predicted_lengths2 = F.log_softmax(predicted_lengths_logits, dim=-1)
        encoder_output2 = x[:, 1:, :]

        unroll_steps = nat_txt_input.size(1)



        rand_seed = random.randint(0, 19260817)

        with torch.no_grad():
            with torch_seed(rand_seed):
                decoder_outputs = self.decode(
                    encoder_output=encoder_output2,
                    encoder_hidden=encoder_hidden2,
                    rd=rd,
                    curriculum_type=curriculum_type,
                    sgn_mask=sgn_mask,
                    txt_input=nat_txt_input,
                    unroll_steps=unroll_steps,
                    txt_mask=nat_txt_mask,
                    )

                word_outputs_nat = decoder_outputs[0][-1]

                pred_tokens = F.softmax(word_outputs_nat, dim=-1).argmax(dim=-1)

                nonpad_positions = nat_txt_mask.squeeze(1)
                same_num = ((pred_tokens == nat_target) & nonpad_positions).sum(1)
                seq_lens = (nonpad_positions).sum(1)
                #glat = 0.5 - 0.4 * train_ratio  #####
                keep_prob = ((seq_lens - same_num) / seq_lens * 0.5).unsqueeze(-1)
                keep_word_mask = (torch.rand(nat_txt_input.shape,
                                             device=nat_txt_input.device) < keep_prob).bool()
                glat_prev_output_tokens = nat_txt_input.masked_fill(keep_word_mask,0) + nat_target.masked_fill(~keep_word_mask, 0)
                glat_tgt_tokens = nat_target.masked_fill(keep_word_mask, 1)

                nat_txt_input, tgt_tokens = glat_prev_output_tokens, glat_tgt_tokens


        with torch_seed(rand_seed):
            decoder_outputs = self.decode(
                encoder_output=encoder_output2,
                encoder_hidden=encoder_hidden2,
                rd=rd,
                curriculum_type=curriculum_type,
                sgn_mask=sgn_mask,
                txt_input=nat_txt_input,
                unroll_steps=unroll_steps,
                txt_mask=nat_txt_mask,
            )






        return decoder_outputs, None,predicted_lengths2,tgt_tokens

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(
            embed_src=self.sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        rd,
        curriculum_type,
        test = False,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        num_layers = len(self.nat_decoder.common_layers)
        if curriculum_type == "linear":
            presv_ratio_for_each_layer = [i / num_layers + 0.1 for i in range(num_layers)]
        elif curriculum_type == "log":
            presv_ratio_for_each_layer = [math.log(i) / math.log(num_layers) for i in range(1, num_layers + 1)]



        return self.nat_decoder(
            test=test,
            txt_input=txt_input,
            presv_ratio_for_each_layer = presv_ratio_for_each_layer,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            rd = rd,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden, )


    def one_hot(self, seq_batch, depth):
        out = Variable(torch.zeros(seq_batch.size()+torch.Size([depth]))).cuda()
        dim = len(seq_batch.size())
        index = seq_batch.view(seq_batch.size()+torch.Size([1]))# 16 32 1
        return out.scatter_(dim, index, 1)


    def get_loss_for_batch(
        self,
        rd,
        curriculum_type,
        batch: Batch,
        translation_loss_function: nn.Module,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable




        decoder_outputs_nat,_,predicted_lengths,nat_target= self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            rd=rd,
            curriculum_type=curriculum_type,
            sgn_lengths=batch.sgn_lengths,
            nat_txt_input=batch.nat_txt_input,
            nat_txt_mask=batch.nat_txt_mask,
            nat_target = batch.nat_target,
        )




        word_outputs_nat,_, _, nat_loss_mask = decoder_outputs_nat




        length_target = batch.txt_lengths
        criterion = nn.NLLLoss(reduction="sum")
        length_target = length_target.contiguous().view(-1)
        length_loss = criterion(
            predicted_lengths.contiguous().view(-1, predicted_lengths.size(-1)), length_target) # true





        translation_loss_nat1 = 0
        layer_xe = []

        nat_loss_mask.append(nat_target.ne(1))


        for idx in range(len(word_outputs_nat)):
            layer_out = word_outputs_nat[idx]
            txt_log_probs_nat = F.log_softmax(layer_out, dim=-1)

            nat_non_pad_mask = nat_loss_mask[idx]
            nat_masked_words_probs = txt_log_probs_nat[nat_non_pad_mask]
            nat_target_words = nat_target[nat_non_pad_mask]

            if nat_target_words.size(0) !=0:
                loss = translation_loss_function(nat_masked_words_probs, nat_target_words)  # 1不会计算loss
                loss = loss / nat_target_words.size(0)
            else:
                loss = 0


            layer_xe.append(loss)
            translation_loss_nat1 += loss

        nat_non_pad_mask = batch.nat_target.ne(1)
        nat_target_words = batch.nat_target[nat_non_pad_mask]

        translation_loss_nat1 = translation_loss_nat1 * nat_target_words.size(0)



        translation_loss =  translation_loss_nat1 + length_loss



        return torch.tensor(0), translation_loss,0,translation_loss_nat1,length_loss


    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.nat_decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param gls_vocab: sign gloss vocabulary
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    """

    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    gloss_output_layer = nn.Linear(encoder.output_size, len(txt_vocab))

    # build decoder and word embeddings
    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
        if cfg["decoder"].get("type", "recurrent") == "transformer":


            nat_decoder = TransformerDecoder_nat(
                **cfg["decoder"],
                encoder=encoder,
                txt_embed=txt_embed,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )



    model: SignModel = SignModel(
        encoder=encoder,
        gloss_output_layer=gloss_output_layer,
        nat_decoder=nat_decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    if do_translation:
        # tie softmax layer with txt embeddings
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    return model
