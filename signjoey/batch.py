# coding: utf-8
import math
import random
import torch
import numpy as np
seed = 1
random = np.random.RandomState(0)

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(
        self,
        torch_batch,
        txt_pad_index,
        sgn_dim,
        is_train: bool = False,
        use_cuda: bool = False,
        frame_subsampling_ratio: int = 3,
        random_frame_subsampling: bool = True,
        random_frame_masking_ratio: float = None,
    ):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with sgn (sign),
        gls (gloss), and txt (text) length, masks, number of non-padded tokens in txt.
        Furthermore, it can be sorted by sgn length.

        :param torch_batch:
        :param txt_pad_index:
        :param sgn_dim:
        :param is_train:
        :param use_cuda:
        :param random_frame_subsampling
        """

        # Sequence Information
        self.sequence = torch_batch.sequence
        self.signer = torch_batch.signer
        self.index = torch_batch.index
        # Sign
        self.sgn, self.sgn_lengths = torch_batch.sgn


        self.sgn_dim = sgn_dim
        self.sgn_mask = (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)


        # Text
        self.txt = None
        self.txt_mask = None
        self.txt_input = None
        self.txt_lengths = None

        # Gloss
        self.gls = None
        self.gls_lengths = None


        # Other
        self.num_txt_tokens = None
        self.num_gls_tokens = None
        self.use_cuda = use_cuda
        self.num_seqs = self.sgn.size(0)



        self.txt,self.txt_lengths = torch_batch.txt

        self.nat_txt_input = torch.zeros(self.txt[:,1:].shape)
        self.nat_target = torch.zeros(self.txt[:,1:].shape)

        self.nat_txt_input2 = torch.zeros(self.txt[:, 1:].shape)
        self.nat_target2 = torch.zeros(self.txt[:, 1:].shape)

        self.masked_token = 0


        for i in range(torch_batch.batch_size):
            txt = self.txt[i,1:]  #不要bos
            txt_input = txt.new(txt.tolist())
            lenth = self.txt_lengths[i]-1
            dec_target = txt.new([txt_pad_index] * len(txt))


            if is_train:
                #ori
                sample_size = lenth
                txt_input[:lenth] = 5  #全mask
                dec_target[:lenth] = txt[:lenth]


            else:
                txt_input[:lenth] = 5
                dec_target[:lenth] = txt[:lenth]
                sample_size = lenth



            self.nat_target[i] = dec_target


            self.nat_txt_input[i] = txt_input
            self.masked_token += sample_size


        self.nat_txt_input = self.nat_txt_input.long()
        self.nat_target = self.nat_target.long()

        txt, txt_lengths = torch_batch.txt
        # txt_input is used for teacher forcing, last one is cut off
        self.at_txt_input = txt[:, :-1]
        self.txt_lengths = txt_lengths -1
        # txt is used for loss computation, shifted by one since BOS
        self.at_txt = txt[:, 1:]
        # we exclude the padded areas from the loss computation
        self.at_txt_mask = (self.at_txt_input != txt_pad_index).unsqueeze(1)
        self.nat_txt_mask = (self.nat_txt_input != txt_pad_index).unsqueeze(1)
        self.num_txt_tokens = (self.at_txt != txt_pad_index).data.sum().item()




        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        Move the batch to GPU

        :return:
        """
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()




        self.txt = self.txt.cuda()

        self.at_txt = self.at_txt.cuda()
        self.at_txt_mask = self.at_txt_mask.cuda()
        self.at_txt_input = self.at_txt_input.cuda()

        self.nat_txt_mask = self.nat_txt_mask.cuda()
        self.nat_txt_input = self.nat_txt_input.cuda()
        self.nat_target = self.nat_target.cuda()



        self.txt_lengths = self.txt_lengths.cuda()



    def sort_by_sgn_lengths(self):
        """
        Sort by sgn length (descending) and return index to revert sort

        :return:
        """
        index_new = []
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        self.sgn = self.sgn[perm_index]
        for idx in perm_index:
            index_new.append(self.index[idx])
        self.index = index_new
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        ###

        self.signer = [self.signer[pi] for pi in perm_index]
        self.sequence = [self.sequence[pi] for pi in perm_index]

        self.txt = self.txt[perm_index]

        self.at_txt = self.at_txt[perm_index]
        self.at_txt_mask = self.at_txt_mask[perm_index]
        self.at_txt_input = self.at_txt_input[perm_index]

        self.nat_txt_mask = self.nat_txt_mask[perm_index]
        self.nat_txt_input = self.nat_txt_input[perm_index]
        self.nat_target = self.nat_target[perm_index]

        self.txt_lengths = self.txt_lengths[perm_index]


        if self.use_cuda:
            self._make_cuda()

        return rev_index
