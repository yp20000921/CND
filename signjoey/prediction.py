#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True
from itertools import groupby
import torch.nn.functional as F

import logging
import numpy as np
import pickle as pickle
import time
import torch.nn as nn

from typing import List
from torchtext.data import Dataset
from signjoey.loss import XentLoss
from signjoey.helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
)
from signjoey.metrics import bleu, chrf, rouge, wer_list
from signjoey.model import build_model, SignModel
from signjoey.batch import Batch
from signjoey.data import load_data, make_data_iter
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN
from signjoey.phoenix_utils.phoenix_cleanup import (
    clean_phoenix_2014,
    clean_phoenix_2014_trans,
)

def predict_length_beam(predicted_lengths, length_beam_size):  #直接计算topk
    beam = predicted_lengths.topk(length_beam_size, dim=1)[1]  #10,5
    beam[beam < 2] = 2
    return beam

def duplicate_encoder_out(encoder_out,mask,bsz, beam_size):
    encoder_out = encoder_out.transpose(0,1)  #len bsz 256
    mask = mask.squeeze(1)
    encoder_ = encoder_out.unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1, bsz * beam_size,encoder_out.size(-1))
    mask = mask.unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, -1)
    encoder_out = encoder_.transpose(0,1)
    mask = mask.unsqueeze(1)
    return encoder_out,mask

def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y  #nonzero 不为0的下标   不为0 就是pad 把原本pad的部位置1


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)  #加上等差数列
    x.view(-1)[i.view(-1)] = y


def assign_multi_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]

def select_worst(token_probs, num_mask):
    bsz, seq_len = token_probs.size()
    masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]  #需要msk的id
    masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
    return torch.stack(masks, dim=0)

# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel,
    data: Dataset,
    rd,
    curriculum_type,
    batch_size: int,
    use_cuda: bool,
    sgn_dim: int,
    do_recognition: bool,
    recognition_loss_function: torch.nn.Module,
    recognition_loss_weight: int,
    do_translation: bool,
    translation_loss_function: torch.nn.Module,
    translation_loss_weight: int,
    translation_max_output_length: int,
    level: str,
    txt_pad_index: int,
    recognition_beam_size: int = 1,
    translation_beam_size: int = 1,
    translation_beam_alpha: int = -1,
    batch_type: str = "sentence",
    dataset_version: str = "phoenix_2014_trans",
    frame_subsampling_ratio: int = None,
) -> (
    float,
    float,
    float,
    List[str],
    List[List[str]],
    List[str],
    List[str],
    List[List[str]],
    List[np.array],
):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param translation_max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param translation_loss_function: translation loss function (XEntropy)
    :param recognition_loss_function: recognition loss function (CTC)
    :param recognition_loss_weight: CTC loss weight
    :param translation_loss_weight: Translation loss weight
    :param txt_pad_index: txt padding token index
    :param sgn_dim: Feature dimension of sgn frames
    :param recognition_beam_size: beam size for validation (recognition, i.e. CTC).
        If 0 then greedy decoding (default).
    :param translation_beam_size: beam size for validation (translation).
        If 0 then greedy decoding (default).
    :param translation_beam_alpha: beam search alpha for length penalty (translation),
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param do_recognition: flag for predicting glosses
    :param do_translation: flag for predicting text
    :param dataset_version: phoenix_2014 or phoenix_2014_trans
    :param frame_subsampling_ratio: frame subsampling ratio

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    valid_iter = make_data_iter(
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False,
        train=False,
    )
    #print(batch_size)

    # disable dropout
    model.eval()
    # don't track gradients during validation
    for line in range(1):
        with torch.no_grad():

            all_attention_scores = []
            all_txt_hyp = []

            total_translation_loss = 0


            for valid_batch in iter(valid_iter):
                batch = Batch(
                    is_train=False,
                    torch_batch=valid_batch,
                    txt_pad_index=txt_pad_index,
                    sgn_dim=sgn_dim,
                    use_cuda=use_cuda,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )
                sort_reverse_index = batch.sort_by_sgn_lengths()

                length_beam_size = 5

                mask_index = 5
                pad_index = 1

                # encoder out



                encoder_output, encoder_hidden = model.encode(
                    sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
                )

                x = encoder_output

                predicted_lengths_logits = torch.matmul(x[:, 0, :],
                                                        model.encoder.length_embed.weight.transpose(0, 1)).float()
                predicted_lengths_logits[:, 0] += float('-inf')  # Cannot predict the len_token
                predicted_lengths = F.log_softmax(predicted_lengths_logits, dim=-1)
                encoder_output = x[:, 1:, :]



                #
                beam = predict_length_beam(predicted_lengths, length_beam_size)
                max_len = beam.max().item()
                #
                src_tokens = batch.sgn
                bsz = src_tokens.size(0)
                #
                length_mask = torch.triu(src_tokens.new(max_len, max_len).fill_(1).long(), 1)
                length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(bsz)], dim=0)  # 10 5，9116
                tgt_tokens = src_tokens.new(bsz, length_beam_size, max_len).fill_(mask_index)  # 16 5 27
                tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * pad_index
                tgt_tokens = tgt_tokens.view(bsz * length_beam_size, max_len)  # 80

                encoder_output, sgn_mask = duplicate_encoder_out(encoder_output, batch.sgn_mask, bsz,
                                                                 length_beam_size)  #


                encoder_hidden = None


                pad_mask = tgt_tokens.eq(pad_index)  # 如果等于pad为true 否则为false


                txt_mask = (tgt_tokens != pad_index).unsqueeze(1)

                tgt_tokens = tgt_tokens.long()

                unroll_steps = batch.nat_txt_input.size(1)  # 50
                decoder_outputs = model.decode(  # true
                    test=True,
                    encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden,
                    rd=0,
                    curriculum_type=curriculum_type,
                    sgn_mask=sgn_mask,
                    txt_input=tgt_tokens,
                    unroll_steps=unroll_steps,
                    txt_mask=txt_mask,
                )
                probs = F.softmax(decoder_outputs[0][-1], dim=-1)
                token_probs, tgt_tokens = probs.max(dim=-1)

                assign_single_value_byte(tgt_tokens, pad_mask, pad_index)  # 80 27    80 27
                assign_single_value_byte(token_probs, pad_mask, 1.0)

                lprobs = token_probs.log().sum(-1)
                hypotheses = tgt_tokens

                bsz = src_tokens.size(0)

                hypotheses = hypotheses.view(bsz, length_beam_size, max_len)
                lprobs = lprobs.view(bsz, length_beam_size)  # 16 5
                tgt_lengths = (1 - length_mask).sum(-1)
                avg_log_prob = lprobs / tgt_lengths.float()  # 16 5
                best_lengths = avg_log_prob.max(-1)[1]
                hypos = torch.stack([hypotheses[b, l, :] for b, l in enumerate(best_lengths)], dim=0)
                all_txt_hyp.extend(hypos[sort_reverse_index])




            decoded_txt = model.txt_vocab.arrays_to_sentences(arrays=all_txt_hyp)
            join_char = " "
            txt_ref = [join_char.join(t) for t in data.txt]
            txt_hyp = [join_char.join(t) for t in decoded_txt]



            def remove_dup(hyp):
                new_hyp = []
                if hyp == []:
                    return []
                for sentence in hyp:
                    new_sen = []
                    sen = sentence.split()
                    if sen == []:
                        new_hyp.append("")
                        continue
                    anchor = sen[0]
                    new_sen.append(anchor)
                    for word in sen[1:]:
                        if word == anchor:
                            pass
                        else:
                            anchor = word
                            new_sen.append(anchor)
                    new_hyp.append(" ".join(new_sen))

                return new_hyp

            txt_bleu_nat = bleu(references=txt_ref, hypotheses=txt_hyp)

            txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
            txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

            #print(txt_bleu_nat)

            txt_hyp_new = remove_dup(txt_hyp)
            txt_rouge_new = rouge(references=txt_ref, hypotheses=txt_hyp_new)
            txt_bleu_nat2 = bleu(references=txt_ref, hypotheses=txt_hyp_new)
            print(txt_bleu_nat2, txt_rouge_new)


















        valid_scores = {}
        if do_recognition:
            valid_scores["wer"] = 0
            valid_scores["wer_scores"] = 0
        if do_translation:
            valid_scores["bleu"] = txt_bleu_nat['bleu4']
            valid_scores["bleu_scores"] = txt_bleu_nat
            valid_scores["bleu_nat"] = txt_bleu_nat['bleu4']
            valid_scores["bleu_scores_nat"] = txt_bleu_nat
            valid_scores["chrf"] = txt_chrf
            valid_scores["rouge"] = txt_rouge


    results = {
        "valid_scores": valid_scores,
        "all_attention_scores": all_attention_scores,
    }


    if do_translation:
        results["valid_translation_loss"] = total_translation_loss
        results["valid_ppl"] = 0
        results["decoded_txt"] = decoded_txt
        results["txt_ref"] = txt_ref
        results["txt_hyp"] = txt_hyp


    return results

# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    train_data, dev_data, test_data, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"])

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,

        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    rd = cfg["model"]["decoder"]['rd']
    curriculum_type = cfg["model"]["decoder"]['curriculum_type']

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        recognition_beam_sizes = [1]
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]

    if "testing" in cfg.keys():
        max_recognition_beam_size = cfg["testing"].get(
            "max_recognition_beam_size", None
        )
        if max_recognition_beam_size is not None:
            recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    if do_recognition:
        recognition_loss_function = torch.nn.CTCLoss(
            blank=model.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
        )
        if use_cuda:
            recognition_loss_function.cuda()
    if do_translation:
        translation_loss_function = XentLoss(
            pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
        )
        if use_cuda:
            translation_loss_function.cuda()

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    assert model.gls_vocab.stoi[SIL_TOKEN] == 0



    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    data=test_data,
                    rd=0,
                    curriculum_type=curriculum_type,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")

        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    data=dev_data,
                    rd=rd,
                    curriculum_type=curriculum_type,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

