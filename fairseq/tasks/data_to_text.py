# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

#from collections import OrderedDict
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    DataToTextDataset,
    FeatDict,
)

from . import FairseqTask, register_task


def load_pair_dataset(
    data_path, split,
    src, src_dicts, src_feat,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
):
    def split_exists(split, src, tgt, feat, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, feat))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')
        if split_exists(split_k, src, tgt, tgt, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_data = FeatDict()#OrderedDict()
        for feat in src_feat:
            src_data[feat] = indexed_dataset.make_dataset(prefix + feat, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=src_dicts[feat])

        src_datasets.append(src_data)
        tgt_datasets.append(indexed_dataset.make_dataset(prefix + tgt, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=tgt_dict))
        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src_feat[0], tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)
 
    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return DataToTextDataset(
        src_dataset, src_dataset[src_feat[0]].sizes, src_dicts, src_feat,
        tgt_dataset, tgt_dataset.sizes, tgt_dict, 
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        remove_eos_from_source=True,
    )


@register_task('data_to_text')
class DataToTextTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--source-features', type=str, metavar='ID1,ID2,..',
                            help='input features ids', nargs='+')
        # fmt: on

    def __init__(self, args, src_dicts, tgt_dict):
        super().__init__(args)
        self.src_dicts = src_dicts
        self.tgt_dict = tgt_dict
        self.src_feat = args.source_features

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_features is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))

        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        src_dicts = FeatDict()#OrderedDict()
        for feat in args.source_features:
            src_dicts[feat] = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(feat)))
            assert src_dicts[feat].pad() == tgt_dict.pad()
            assert src_dicts[feat].unk() == tgt_dict.unk()
            print('| [{}] dictionary: {} types'.format(feat, len(src_dicts[feat])))

        return cls(args, src_dicts, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        self.datasets[split] = load_pair_dataset(
            data_path, split, self.args.source_lang, self.src_dicts, self.src_feat, 
            self.args.target_lang, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return DataToTextDataset(src_tokens, src_lengths, self.source_dictionary, self.src_feat)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dicts

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
