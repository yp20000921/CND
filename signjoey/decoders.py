# coding: utf-8

"""
Various decoders
"""
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from signjoey.attention import BahdanauAttention, LuongAttention
from signjoey.encoders import Encoder
from signjoey.helpers import freeze_params, subsequent_mask
from signjoey.transformer_layers import PositionalEncoding, TransformerDecoderLayer

import torch.nn.functional as F
import numpy as np
from signjoey.da_utils import *

import math
random = np.random.RandomState(0)


# pylint: disable=abstract-method
class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class RecurrentDecoder(Decoder):
    """A conditional RNN decoder with attention."""

    def __init__(
        self,
        rnn_type: str = "gru",
        emb_size: int = 0,
        hidden_size: int = 0,
        encoder: Encoder = None,
        attention: str = "bahdanau",
        num_layers: int = 1,
        vocab_size: int = 0,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        init_hidden: str = "bridge",
        input_feeding: bool = True,
        freeze: bool = False,
        **kwargs
    ) -> None:
        """
        Create a recurrent decoder with attention.

        :param rnn_type: rnn type, valid options: "lstm", "gru"
        :param emb_size: target embedding size
        :param hidden_size: size of the RNN
        :param encoder: encoder connected to this decoder
        :param attention: type of attention, valid options: "bahdanau", "luong"
        :param num_layers: number of recurrent layers
        :param vocab_size: target vocabulary size
        :param hidden_dropout: Is applied to the input to the attentional layer.
        :param dropout: Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param init_hidden: If "bridge" (default), the decoder hidden states are
            initialized from a projection of the last encoder state,
            if "zeros" they are initialized with zeros,
            if "last" they are identical to the last encoder state
            (only if they have the same size)
        :param input_feeding: Use Luong's input feeding.
        :param freeze: Freeze the parameters of the decoder during training.
        :param kwargs:
        """

        super(RecurrentDecoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        if self.input_feeding:  # Luong-style
            # combine embedded prev word +attention vector before feeding to rnn
            self.rnn_input_size = emb_size + hidden_size
        else:
            # just feed prev word embedding
            self.rnn_input_size = emb_size

        # the decoder RNN
        self.rnn = rnn(
            self.rnn_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True
        )

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self._output_size = vocab_size

        if attention == "bahdanau":
            self.attention = BahdanauAttention(
                hidden_size=hidden_size,
                key_size=encoder.output_size,
                query_size=hidden_size,
            )
        elif attention == "luong":
            self.attention = LuongAttention(
                hidden_size=hidden_size, key_size=encoder.output_size
            )
        else:
            raise ValueError(
                "Unknown attention mechanism: %s. "
                "Valid options: 'bahdanau', 'luong'." % attention
            )

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # to initialize from the final encoder state of last layer
        self.init_hidden_option = init_hidden
        if self.init_hidden_option == "bridge":
            self.bridge_layer = nn.Linear(encoder.output_size, hidden_size, bias=True)
        elif self.init_hidden_option == "last":
            if encoder.output_size != self.hidden_size:
                if encoder.output_size != 2 * self.hidden_size:  # bidirectional
                    raise ValueError(
                        "For initializing the decoder state with the "
                        "last encoder state, their sizes have to match "
                        "(encoder: {} vs. decoder:  {})".format(
                            encoder.output_size, self.hidden_size
                        )
                    )
        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward_step(
        self,
        prev_embed: Tensor,
        prev_att_vector: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        hidden: Tensor,
    ) -> None:
        """
        Make sure the input shapes to `self._forward_step` are correct.
        Same inputs as `self._forward_step`.

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        """
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size([1, self.hidden_size])
        assert prev_att_vector.shape[0] == prev_embed.shape[0]
        assert encoder_output.shape[0] == prev_embed.shape[0]
        assert len(encoder_output.shape) == 3
        assert src_mask.shape[0] == prev_embed.shape[0]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[2] == encoder_output.shape[1]
        if isinstance(hidden, tuple):  # for lstm
            hidden = hidden[0]
        assert hidden.shape[0] == self.num_layers
        assert hidden.shape[1] == prev_embed.shape[0]
        assert hidden.shape[2] == self.hidden_size

    def _check_shapes_input_forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        src_mask: Tensor,
        hidden: Tensor = None,
        prev_att_vector: Tensor = None,
    ) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param trg_embed:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param hidden:
        :param prev_att_vector:
        """
        assert len(encoder_output.shape) == 3
        assert len(encoder_hidden.shape) == 2
        assert encoder_hidden.shape[-1] == encoder_output.shape[-1]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[0] == encoder_output.shape[0]
        assert src_mask.shape[2] == encoder_output.shape[1]
        assert trg_embed.shape[0] == encoder_output.shape[0]
        assert trg_embed.shape[2] == self.emb_size
        if hidden is not None:
            if isinstance(hidden, tuple):  # for lstm
                hidden = hidden[0]
            assert hidden.shape[1] == encoder_output.shape[0]
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == encoder_output.shape[0]
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(
        self,
        prev_embed: Tensor,
        prev_att_vector: Tensor,  # context or att vector
        encoder_output: Tensor,
        src_mask: Tensor,
        hidden: Tensor,
    ) -> (Tensor, Tensor, Tensor):
        """
        Perform a single decoder step (1 token).

        1. `rnn_input`: concat(prev_embed, prev_att_vector [possibly empty])
        2. update RNN with `rnn_input`
        3. calculate attention and context/attention vector

        :param prev_embed: embedded previous token,
            shape (batch_size, 1, embed_size)
        :param prev_att_vector: previous attention vector,
            shape (batch_size, 1, hidden_size)
        :param encoder_output: encoder hidden states for attention context,
            shape (batch_size, src_length, encoder.output_size)
        :param src_mask: src mask, 1s for area before <eos>, 0s elsewhere
            shape (batch_size, 1, src_length)
        :param hidden: previous hidden state,
            shape (num_layers, batch_size, hidden_size)
        :return:
            - att_vector: new attention vector (batch_size, 1, hidden_size),
            - hidden: new hidden state with shape (batch_size, 1, hidden_size),
            - att_probs: attention probabilities (batch_size, 1, src_len)
        """

        # shape checks
        self._check_shapes_input_forward_step(
            prev_embed=prev_embed,
            prev_att_vector=prev_att_vector,
            encoder_output=encoder_output,
            src_mask=src_mask,
            hidden=hidden,
        )

        if self.input_feeding:
            # concatenate the input with the previous attention vector
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.emb_dropout(rnn_input)

        # rnn_input: batch x 1 x emb+2*enc_size
        _, hidden = self.rnn(rnn_input, hidden)

        # use new (top) decoder layer as attention query
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

        # compute context vector using attention mechanism
        # only use last layer for attention mechanism
        # key projections are pre-computed
        context, att_probs = self.attention(
            query=query, values=encoder_output, mask=src_mask
        )

        # return attention vector (Luong)
        # combine context with decoder hidden state before prediction
        att_vector_input = torch.cat([query, context], dim=2)
        # batch x 1 x 2*enc_size+hidden_size
        att_vector_input = self.hidden_dropout(att_vector_input)

        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x hidden_size
        return att_vector, hidden, att_probs

    def forward(
        self,
        trg_embed: Tensor,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        src_mask: Tensor,
        unroll_steps: int,
        hidden: Tensor = None,
        prev_att_vector: Tensor = None,
        **kwargs
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
         Unroll the decoder one step at a time for `unroll_steps` steps.
         For every step, the `_forward_step` function is called internally.

         During training, the target inputs (`trg_embed') are already known for
         the full sequence, so the full unrol is done.
         In this case, `hidden` and `prev_att_vector` are None.

         For inference, this function is called with one step at a time since
         embedded targets are the predictions from the previous time step.
         In this case, `hidden` and `prev_att_vector` are fed from the output
         of the previous call of this function (from the 2nd step on).

         `src_mask` is needed to mask out the areas of the encoder states that
         should not receive any attention,
         which is everything after the first <eos>.

         The `encoder_output` are the hidden states from the encoder and are
         used as context for the attention.

         The `encoder_hidden` is the last encoder hidden state that is used to
         initialize the first hidden decoder state
         (when `self.init_hidden_option` is "bridge" or "last").

        :param trg_embed: emdedded target inputs,
            shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states from the encoder,
            shape (batch_size, src_length, encoder.output_size)
        :param encoder_hidden: last state from the encoder,
            shape (batch_size x encoder.output_size)
        :param src_mask: mask for src states: 0s for padded areas,
            1s for the rest, shape (batch_size, 1, src_length)
        :param unroll_steps: number of steps to unrol the decoder RNN
        :param hidden: previous decoder hidden state,
            if not given it's initialized as in `self.init_hidden`,
            shape (num_layers, batch_size, hidden_size)
        :param prev_att_vector: previous attentional vector,
            if not given it's initialized with zeros,
            shape (batch_size, 1, hidden_size)
        :return:
            - outputs: shape (batch_size, unroll_steps, vocab_size),
            - hidden: last hidden state (num_layers, batch_size, hidden_size),
            - att_probs: attention probabilities
                with shape (batch_size, unroll_steps, src_length),
            - att_vectors: attentional vectors
                with shape (batch_size, unroll_steps, hidden_size)
        """

        # shape checks
        self._check_shapes_input_forward(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            hidden=hidden,
            prev_att_vector=prev_att_vector,
        )

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self._init_hidden(encoder_hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(keys=encoder_output)

        # here we store all intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []

        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size]
                )

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden,
            )
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, unroll_steps, src_length
        outputs = self.output_layer(att_vectors)
        # outputs: batch, unroll_steps, vocab_size
        return outputs, hidden, att_probs, att_vectors

    def _init_hidden(self, encoder_final: Tensor = None) -> (Tensor, Optional[Tensor]):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state of the last encoder layer.

        In case of `self.init_hidden_option == "bridge"`
        and a given `encoder_final`, this is a projection of the encoder state.

        In case of `self.init_hidden_option == "last"`
        and a size-matching `encoder_final`, this is set to the encoder state.
        If the encoder is twice as large as the decoder state (e.g. when
        bi-directional), just use the forward hidden state.

        In case of `self.init_hidden_option == "zero"`, it is initialized with
        zeros.

        For LSTMs we initialize both the hidden state and the memory cell
        with the same projection/copy of the encoder hidden state.

        All decoder layers are initialized with the same initial values.

        :param encoder_final: final state from the last layer of the encoder,
            shape (batch_size, encoder_hidden_size)
        :return: hidden state if GRU, (hidden state, memory cell) if LSTM,
            shape (batch_size, hidden_size)
        """
        batch_size = encoder_final.size(0)

        # for multiple layers: is the same for all layers
        if self.init_hidden_option == "bridge" and encoder_final is not None:
            # num_layers x batch_size x hidden_size
            hidden = (
                torch.tanh(self.bridge_layer(encoder_final))
                .unsqueeze(0)
                .repeat(self.num_layers, 1, 1)
            )
        elif self.init_hidden_option == "last" and encoder_final is not None:
            # special case: encoder is bidirectional: use only forward state
            if encoder_final.shape[1] == 2 * self.hidden_size:  # bidirectional
                encoder_final = encoder_final[:, : self.hidden_size]
            hidden = encoder_final.unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:  # initialize with zeros
            with torch.no_grad():
                hidden = encoder_final.new_zeros(
                    self.num_layers, batch_size, self.hidden_size
                )

        return (hidden, hidden) if isinstance(self.rnn, nn.LSTM) else hidden

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (self.rnn, self.attention)





class TransformerDecoder_at(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        hidden_size: int = 512,
        ff_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        vocab_size: int = 1,
        freeze: bool = False,
        **kwargs
    ):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder_at, self).__init__()

        self._hidden_size = hidden_size
        self._output_size = vocab_size

        # create num_layers decoder layers and put them in a list
        self.common_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(
        self,
        trg_embed: Tensor = None,
        encoder_output: Tensor = None,
        encoder_hidden: Tensor = None,
        src_mask: Tensor = None,
        unroll_steps: int = None,
        hidden: Tensor = None,
        trg_mask: Tensor = None,
        **kwargs
    ):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)


        trg_mask = trg_mask & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)

        # for layer in self.common_layers:
        #     x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.common_layers),
            8,
        )


class TransformerDecoder_nat(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(
            self,
            num_layers: int = 4,
            num_heads: int = 8,
            hidden_size: int = 512,
            ff_size: int = 2048,
            dropout: float = 0.1,
            emb_dropout: float = 0.1,
            vocab_size: int = 1,
            freeze: bool = False,
            txt_embed=None,
            **kwargs
    ):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder_nat, self).__init__()

        self._hidden_size = hidden_size
        self._output_size = vocab_size

        self.txt_embed = txt_embed
        self.max_relative_positions = 200

        self.num_layers = num_layers

        # create num_layers decoder layers and put them in a list
        self.common_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    max_relative_positions=self.max_relative_positions,
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )


        self.pos_embedding = nn.Embedding(self.max_relative_positions, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.relative_attention = True

        self.position_buckets = -1
        pos_ebd_size = self.max_relative_positions * 2
        if self.position_buckets > 0:
            pos_ebd_size = self.position_buckets * 2
        self.rel_embeddings = nn.Embedding(pos_ebd_size, hidden_size)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self.PEfc = nn.Linear(2 * hidden_size, hidden_size)
        self.reduce_concat = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)
             for _ in range(num_layers - 1)])

        self.softcopy_learnable = True

        if self.softcopy_learnable:
            self.para_softcopy_temp = torch.nn.Parameter(torch.tensor(1.0))

        self.copy_attn = nn.Linear(hidden_size, hidden_size, bias=False)

        if freeze:
            freeze_params(self)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight  # 200*256
        rel_embeddings = self.layer_norm(rel_embeddings)
        return rel_embeddings

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)  # len
            relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size=self.position_buckets,
                                                   max_position=self.max_relative_positions)
        return relative_pos

    def new_arange(self, x, *size):
        """
        Return a Tensor of `size` filled with a range function on the device of x.
        If size is empty, using the size of the variable x.
        """
        if len(size) == 0:
            size = x.size()
        return torch.arange(size[-1], device=x.device).expand(*size).contiguous()




    def forward(
            self,
            rd,
            presv_ratio_for_each_layer,
            test = False,
            txt_input=None,
            trg_embed: Tensor = None,
            encoder_output: Tensor = None,
            src_mask: Tensor = None,
            trg_mask: Tensor = None,
            **kwargs
    ):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:

        """

        assert trg_mask is not None, "trg_mask required for Transformer"

        trg_len = trg_embed.shape[1]
        batch_size = trg_embed.shape[0]
        bsz = trg_embed.size(0)

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).cuda()
        pe = self.pos_embedding(pos)

        attn_score = torch.bmm(self.copy_attn(pe), encoder_output.transpose(1, 2))  # 16 32 177
        if src_mask is not None:
            attn_score = attn_score.masked_fill(~src_mask.expand(-1, trg_len, -1), float('-inf'))
        attn_weight = F.softmax(attn_score, dim=-1)
        x = torch.bmm(attn_weight, encoder_output)  # 16 32 512
        # # mask_target_x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)  #没有pos
        # # output_mask = prev_output_tokens.eq(self.unk)  等于mask == 不为pad
        output_mask = txt_input.eq(5)
        cat_x = torch.cat([trg_embed.unsqueeze(2), x.unsqueeze(2)], dim=2).view(-1, x.size(2))
        # #index = torch.arange(batch_size * trg_len).cuda() * 2 + output_mask.view(-1).long()
        x = cat_x.index_select(dim=0,
                               index=torch.arange(batch_size * trg_len).cuda() * 2 +  # 如果是mask就选x 否则trg_embed
                                     output_mask.view(-1).long()).reshape(batch_size, trg_len, x.size(2))


        x = torch.cat((x, pe), -1)
        x = self.PEfc(x)
        x = self.emb_dropout(x)

        relative_pos = self.get_rel_pos(x, None, None)
        rel_embeddings = self.get_rel_embedding()

        nat_all_layer_outputs = []
        loss_cpt_mask = []
        all_yhat_tokens = []

        nat_mask = txt_input.eq(5)

        attention_mask = trg_mask.unsqueeze(1)

        trg_mask_causal = trg_mask & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)








        for i, layer in enumerate(self.common_layers):
            presv_ratio = presv_ratio_for_each_layer[i]
            if i == 0:
                new_x = x
            else:
                # x = self.layer_norm(x) #########
                layer_out = self.output_layer(x)  # 3458
                nat_all_layer_outputs.append(layer_out)  # 512
                layer_out_logits = F.softmax(layer_out, dim=-1)

                prob_for_trg_tokens0, _ = layer_out_logits.max(-1)

                prob_for_trg_tokens2, top2 = layer_out_logits.topk(2, dim=-1)
                top2 = top2[:, :, -1]  #############

                prob_for_trg_tokens = prob_for_trg_tokens0.masked_fill(~nat_mask, 0)  # 把不是5的地方mask

                y_hat_tokens = layer_out_logits.argmax(dim=-1)

                ####

                all_yhat_tokens.append(y_hat_tokens)

                unk_mask = (y_hat_tokens == -1)
                unk_mask2 = (y_hat_tokens == -1)
                for j in range(bsz):
                    lenth = torch.nonzero(nat_mask[j] == True).size(0)
                    num_of_presv_tokens = int(lenth * presv_ratio)
                    if num_of_presv_tokens == 0:
                        num_of_presv_tokens += 1

                    _, index = prob_for_trg_tokens[j].topk(num_of_presv_tokens, dim=-1)

                    if rd > 0 and not test: ##
                        ind = random.choice(index.size(0), size=math.ceil(rd*index.size(0)), replace=False)
                        index2 = index[ind]  # 替换gt的那些下标
                        unk_mask2[j][index2] = True

                    unk_mask[j][index] = True

                y_hat_tokens = y_hat_tokens * unk_mask + txt_input * ~unk_mask


                if rd > 0 and not test: #
                    y_hat_tokens = top2 * unk_mask2 + y_hat_tokens * ~unk_mask2  ##

                y_hat_embed = self.txt_embed(y_hat_tokens, trg_mask)  #

                ####
                new_x = torch.cat((x, y_hat_embed), dim=-1)
                new_x = self.reduce_concat[-1](new_x)

                loss_cpt_mask.append(unk_mask)  #

            x = layer(x=new_x, memory=encoder_output, src_mask=src_mask, trg_mask=attention_mask,
                      relative_pos=relative_pos, rel_embeddings=rel_embeddings,
                      trg_mask_casual=trg_mask_causal.unsqueeze(1))

        x = self.layer_norm(x)  # grad True
        output = self.output_layer(x)  # grad True

        y_hat_tokens = output.argmax(dim=-1)
        all_yhat_tokens.append(y_hat_tokens)

        nat_all_layer_outputs.append(output)



        return nat_all_layer_outputs, x, None, loss_cpt_mask




    def __repr__(self):
        return "%s(total_layers=%r,common=%r, num_heads=%r)" % (
            self.__class__.__name__,
            len(self.common_layers),
            len(self.common_layers),
            8,
        )




