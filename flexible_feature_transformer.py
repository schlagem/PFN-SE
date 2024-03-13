import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext
from tqdm import tqdm

import torch
from torch import nn

import utils
from transformer import TransformerModel
from bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
    get_bucket_limits,
    get_custom_bar_dist,
)
from utils import (
    get_cosine_schedule_with_warmup,
    get_openai_lr,
    StoreDictKeyPair,
    get_weighted_single_eval_pos_sampler,
    get_uniform_single_eval_pos_sampler,
)
import encoders
import positional_encodings
from utils import init_dist
from torch.cuda.amp import autocast, GradScaler
from torch import nn


class FlexibleAttributeSizeTransformer(nn.Module):
    def __init__(
        self,
        encoder_generator,
        style_encoder,
        y_encoder_generator,
        pos_encoder_generator,
        emsize,
        n_out,
        nlayers,
        nhead,
        nhid,
        input_normalization,
        dropout,
        efficient_eval_masking,
        initializer,
        model_extra_args,
    ) -> None:
        super().__init__()

        self.emsize_npt_like = 32  # emsize // 32
        self.random_feature_embedding_size = self.emsize_npt_like
        encoder = encoder_generator(self.emsize_npt_like, emsize)  # IF NEW ARCH
        self.model_encoder_npt_like_datapoints = TransformerModel(
            torch.nn.Linear(1, self.emsize_npt_like),
            n_out,
            self.emsize_npt_like,
            nhead,
            nhid,
            nlayers=4,
            dropout=dropout,
            style_encoder=None,
            y_encoder=None,
            input_normalization=input_normalization,
            num_global_att_tokens=32,
            pos_encoder=None,
            decoder={"standard": (None, self.emsize_npt_like)}
            # (torch.nn.Identity, 0)
            ,
            init_method=initializer,
            efficient_eval_masking=efficient_eval_masking,
            decoder_once=None,
            return_all_outputs=True
            #     , num_global_att_tokens=32 if num_global_att_tokens > 0 else 0
            ,
            **model_extra_args
        )
        self.model_encoder_npt_like_attributes = TransformerModel(
            torch.nn.Identity(),
            n_out,
            self.emsize_npt_like,
            nhead,
            nhid,
            nlayers=8,
            dropout=dropout,
            style_encoder=None,
            y_encoder=None,
            input_normalization=input_normalization,
            return_all_outputs=True,
            num_global_att_tokens=32,
            pos_encoder=None,
            decoder={"standard": (None, self.emsize_npt_like)},
            init_method=initializer,
            efficient_eval_masking=False,
            decoder_once=None  # (torch.nn.Identity, 0)
            #   , num_global_att_tokens=32 if num_global_att_tokens > 0 else 0
            ,
            **model_extra_args
        )

        self.model_transformer = TransformerModel(
            encoder,
            n_out,
            emsize,
            nhead,
            nhid,
            nlayers,
            dropout,
            style_encoder=style_encoder,
            y_encoder=y_encoder_generator(1, emsize),
            input_normalization=input_normalization,
            pos_encoder=(
                pos_encoder_generator or positional_encodings.NoPositionalEncoding
            )(emsize, seq_len * 2),
            decoder=decoders,
            init_method=initializer,
            efficient_eval_masking=efficient_eval_masking,
            decoder_once=decoder_once,
            num_global_att_tokens=num_global_att_tokens,
            **model_extra_args
        )
        self.random_feature_embedding_encoder = torch.nn.Linear(
            self.random_feature_embedding_size, self.emsize_npt_like
        )

    def forward(self, src, src_mask=None, single_eval_pos=None) -> torch.Tensor:
        assert isinstance(
            src, tuple
        ), "inputs (src) have to be given as (x,y) or (style,x,y) tuple"

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src
        N, B, F = x_src.shape
        # x_src = torch.ones_like(x_src)
        # x_src[:, 1:, :] = 0
        # x_src[:, 0, :] = x_src[:, 0, :]  * torch.arange(1, F+1, device=x_src.device).float() / F
        x_src = x_src.reshape((N, B * F, 1))  # N, B, F -> N, B*F, 1
        # print('EMBED datapoints', x_src.shape)
        x_src, _ = self.model_encoder_npt_like_datapoints(
            (None, x_src, None), src_mask=src_mask, single_eval_pos=single_eval_pos
        )  # N, B*F, 1 -> N, B*F, emsize_npt_like
        x_src = x_src["standard"]
        x_src = x_src.reshape(N, B, F, -1).transpose(
            0, 2
        )  # N, B*F, emsize_npt_like -> N, B, F, emsize_npt_like -> F, B, N, emsize_npt_like
        x_src = x_src.reshape(
            F, B * N, -1
        )  # F, B, N, emsize_npt_like -> F, B * N, emsize_npt_like
        random_feature_embedding = torch.randn(
            (F, B, self.random_feature_embedding_size), device=x_src.device
        )
        random_feature_embedding = random_feature_embedding.repeat(
            1, N, 1
        )  # F, B, random_feature_embedding_size -> F, B * N, random_feature_embedding_size
        random_feature_embedding = self.random_feature_embedding_encoder(
            random_feature_embedding
        )
        x_src = (
            random_feature_embedding + x_src
        )  # F, B * N, emsize_npt_like -> F, B * N, emsize_npt_like
        x_src, _ = self.model_encoder_npt_like_attributes(
            (None, x_src, None)
        )  # F, B * N, emsize_npt_like -> F, B * N, emsize_npt_like
        x_src = x_src["standard"]
        x_src = x_src.mean(0)  # F, B * N, emsize_npt_like -> B * N, emsize_npt_like

        x_src = x_src.reshape(
            (B, N, -1)
        )  # B * N, emsize_npt_like -> B, N, emsize_npt_like
        x_src = x_src.transpose(0, 1)  # B, N, emsize_npt_like -> N, B, emsize_npt_like

        print("HID", x_src[0:2, 0:2, 0])
        # print('TRANSFORMER', x_src.shape)
        return self.model_transformer(
            (style_src, x_src, y_src),
            src_mask=src_mask,
            single_eval_pos=single_eval_pos,
        )
