# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    GlobalSelfAttention 

)
from .transformer import TransformerEncoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model('select_plan')
class SelectPlanModel(FairseqEncoderDecoderModel):
    """
    Data to text generation `"Data-to-Text Generation with Content Selection and Planning" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict, src_feat = task.source_dictionary, task.target_dictionary, task.src_feat

        def build_embedding(dictionary, embed_dim, path=None, feat=False):
            
            if feat:
                emb = Embeddings([Embedding(len(vocab), embed_dim, vocab.pad())
                            for _, vocab in dictionary.items()])
            else:
                padding_idx = dictionary.pad()
                num_embeddings = len(dictionary)
                emb = Embedding(num_embeddings, embed_dim, padding_idx)
                # if provided, load from preloaded dictionaries
                if path:
                    embed_dict = utils.parse_embedding(path)
                    utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        encoder_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim, args.encoder_embed_path, feat=True
        )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, None)
        return SelectPlanModel(encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FeedForwardEncoder(args, src_dict, embed_tokens)
        #return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return PointerRNNDecoder(args, tgt_dict, embed_tokens)

class FeedForwardEncoder(FairseqEncoder):
    """
    Feed Forward encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`FeedForwardLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = embed_tokens

        self.layers = nn.ModuleList([])
        self.layers.extend([
            FeedForwardLayer(i, embed_dim, args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_tokens(src_tokens)
        #x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens[:,:,0].eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        #if self.layer_norm:
        #    x = self.layer_norm(x)

        if  encoder_padding_mask is not None:
             encoder_padding_mask =  encoder_padding_mask

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # :B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

class PointerLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        return attn_scores.t()

class PointerRNNDecoder(FairseqIncrementalDecoder):
    def __init__(
        self, args, dictionary, embed_tokens,
    ):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.num_layers = args.decoder_layers
        hidden_size = args.decoder_ffn_embed_dim
        encoder_output_units =  args.encoder_embed_dim
        self.layer_norm = LayerNorm(encoder_output_units)
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=encoder_output_units if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(self.num_layers)
        ])
        self.attention = PointerLayer(hidden_size, encoder_output_units, bias=False)


    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        encoder_padding_mask = encoder_out['encoder_padding_mask'].t()
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()
        srclen = encoder_out.size(0)
        
        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, _ = cached_state
        else:         
            prev_hiddens = [encoder_out.mean(dim=0) for i in range(self.num_layers)]
            prev_cells = [encoder_out.mean(dim=0) for i in range(self.num_layers)]

        # get outputs from encoder
        #encoder_out = self.layer_norm(encoder_out)
        x = torch.stack([encoder_out[:,i,:].index_select(0, prev_output_tokens[i])
                        for i in range(encoder_out.size(1))], dim=1)
        x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #encoder_out = F.dropout(encoder_out, p=self.dropout, training=self.training)

        attn_scores = x.new_zeros(bsz, seqlen, srclen)
        for j in range(seqlen):
            input = x[j, :, :]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            attn_scores[:, j, :] = self.attention(hidden, encoder_out, encoder_padding_mask)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, None),
        )

        return attn_scores, None



    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn
    
    def to_pointer(self, tgt_item):
        return tgt_item

class FeedForwardLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, layer_num, embed_dim, args):
        super().__init__()
        self.layer_num = layer_num
        self.embed_dim = embed_dim
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        #self.activation_dropout = getattr(args, 'activation_dropout', 0)
        #if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
        #    self.activation_dropout = getattr(args, 'relu_dropout', 0)
        #self.normalize_before = args.encoder_normalize_before
        if layer_num == 0:
            input_dim = self.embed_dim
            self.layer_norm = nn.Sequential()
        else:
            input_dim = args.encoder_ffn_embed_dim
            self.layer_norm = LayerNorm(input_dim)
        self.fc = Linear(input_dim, args.encoder_ffn_embed_dim)
        #self.self_attn = MultiheadAttention(
        #    args.encoder_ffn_embed_dim, args.encoder_attention_heads,
        #    dropout=args.attention_dropout,
        #)
        self.self_attn = GlobalSelfAttention(
            args.encoder_ffn_embed_dim, 
        )
        #self.layer_norm2 = LayerNorm(args.encoder_ffn_embed_dim)
        #self.fc2 = Linear(args.encoder_ffn_embed_dim * 2, args.encoder_ffn_embed_dim)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        #residual = x
        #x = self.layer_norm(x)
        x = self.activation_fn(self.fc(x))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #if self.layer_num > 0:
        #    x = residual + x
        #residual = x
        #x = self.layer_norm2(x)
        #x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        #x, _ = self.self_attn(x.transpose(0, 1).contiguous(), x.transpose(0, 1).contiguous(), encoder_padding_mask) 
        #x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class Embeddings(nn.ModuleList):
    def __init__(self, *args):
        super(Embeddings, self).__init__(*args)
        self.embedding_dim = sum([x.embedding_dim for x in self])
        self.padding_idx = self[0].padding_idx       
        for x in self:
            assert x.padding_idx == self.padding_idx, "Pad id differ"
        self.out = "concat"
        if self.out == "proj":
            self.out_proj = nn.Linear(self.embedding_dim, self[0].embedding_dim)
            self.embedding_dim = self[0].embedding_dim
    def forward(self, input): 
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        #assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.out == "concat":
            return torch.cat(outputs, 2)
        if self.out == "sum":
            return outputs.sum(2)
        else:
            return self.out_proj(torch.cat(outputs, 2))


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


# default parameters used in tensor2tensor implementation
@register_model_architecture('select_plan', 'select_plan')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.decoder_ffn_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.decoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 1)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
