# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

from torch.nn.utils.rnn import pad_sequence
import random
import os
import sys
import argparse
import configparser
import timeit
import torch
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")
from examples.pytorch.starcoder.utils.star_coder import StarCoder
from examples.pytorch.starcoder.utils.vocab import Vocab, GPTTokenizer

# from utils import word_list

def main():
    print(" -AIXcoder- ".center(119, "="))
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num',            type=int,              default=40,
                        help='number of layers')
    parser.add_argument('--output_len',           type=int,              default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num',             type=int,              default=48,
                        help='head number')
    parser.add_argument('--size_per_head',        type=int,              default=128,
                        help='size per head')
    parser.add_argument('--inter_size',           type=int,              default=1,
                        help='inter size')
    parser.add_argument('--vocab_size',           type=int,              default=49152,
                        help='vocab size')
    parser.add_argument('--beam_width',           type=int,              default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k',                type=int,              default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p',                type=float,            default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature',          type=float,            default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size',     type=int,               default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size',   type=int,               default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path',            type=str,               default='/data3/StarCoderBase/CodeGPT.pt',
                        help='path to the checkpoint file.')
    parser.add_argument('--lib_path',             type=str,               default='../../../build/lib/libth_star_coder.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--vocab_file',           type=str,               default='/data3/StarCoderBase/vocab.json',
                        help='vocabulary file.')    
    parser.add_argument('--tokenizer_file',       type=str,               default='/data3/StarCoderBase/aixTokenizer',
                        help='vocabulary file.')
    # parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
    #                     help='merges file.')
    parser.add_argument('--start_id',             type=int,               default=10,
                        help='start token id.')
    parser.add_argument('--end_id',               type=int,               default=10,
                        help='end token id.')
    parser.add_argument('--max_batch_size',       type=int,               default=1,
                        help='max batch size.')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--min_length', type=int, default=0,
                        help='A minimum number of tokens to generate')
    parser.add_argument('--max_seq_len',          type=int,               default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type', '--data_type', 
                                                  type=str, 
                                                  choices=['fp32', 'fp16', 'bf16', 'fp8'], 
                                                  default='fp32')
    parser.add_argument('--time',                 action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file',    type=str,               default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file',   type=str,               default=None,
                        help='path to sample output file.')
    parser.add_argument('--enable_random_seed',   action='store_true',
                        help='is enable the random seed.')
    parser.add_argument('--skip_end_tokens',      dest='skip_end_tokens', action='store_true',
                        help='Whether to remove or not end tokens in outputs.')
    parser.add_argument('--no_detokenize',        dest='detokenize',      action='store_false',
                        help='Skip detokenizing output token ids.')
    parser.add_argument('--sparse',               action='store_true',    dest='sparse',
                        help='Enable sparse matrix multiplication. (Need SM 8.0 or 8.6 and SPARSITY_SUPPORT=ON)')
    # parser.add_argument('--use_jieba_tokenizer', action='store_true',
    #                     help='use JiebaBPETokenizer as tokenizer.')
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help='Data type of FT checkpoint weights',
    )
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')

    parser.add_argument('--banned_words',
        type=str,
        default="",
        help='A comma separated list of tokens that should never be generated. Everything between the commas will'
             ' be tokenized and converted to token ids that will be banned.'
             ' Note that spaces before and after commas are included in tokenization.'
             ' An example highlighting this importance is that "the" and " the" are'
             ' two separate tokens some vocabularies.'
             ' Therefore, do ban a certain phrase, we would need to specify all tokens'
             ' in the vocabulary that include the phrase.'
             ' Example use: --banned_words "the, the,a,boy". This will ban the tokens "the", " the", "a" and "boy".'
             ' We can also use a pipe "|" to ban different tokens for different sentences in a batch.'
             ' Example: --banned_words "the, the|a,boy" will ban the tokens "the" and " the" in output sentence 1 and'
             ' ban the tokens "a" and "boy" in output sentence 2. When using this mode, we must specify a set of tokens to ban'
             ' for each sentence in the batch.',
    )

    args             =                        parser.parse_args()

    ckpt_config      =                configparser.ConfigParser()
    ckpt_config_path = os.path.join(args.ckpt_path, 'config.ini')
    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)
    print(" -Config Parser- ".center(119, "="))
    if 'StarCoder' in ckpt_config.keys():
        for args_key, config_key, func in [
            (        'layer_num',         'layer_num', ckpt_config.getint),
            (      'max_seq_len',       'max_seq_len', ckpt_config.getint),
            ('weights_data_type', 'weights_data_type',    ckpt_config.get),
        ]:
            if config_key in ckpt_config['StarCoder'].keys():
                prev_val                =       args.__dict__[args_key]
                args.__dict__[args_key] = func('StarCoder', config_key)
                print('Loading {:>17} from config.ini,    previous: {:>4},    current: {:>4}'.format(
                    args_key, prev_val, args.__dict__[args_key]))
            else:
                print('Not loading {:>17} from config.ini'.format(args_key))
        for key in ['head_num', 'size_per_head', 'tensor_para_size']:
            if key in args.__dict__:
                prev_val           =             args.__dict__[key]
                args.__dict__[key] = ckpt_config.getint('StarCoder', key)
                print('Loading {:>17} from config.ini,    previous: {:>4},    current: {:>4}'.format(
                    key, prev_val, args.__dict__[key]))
            else:
                print('Not loading {:>17} from config.ini'.format(key))

    layer_num                  =                  args.layer_num
    output_len                 =                 args.output_len
    head_num                   =                   args.head_num
    size_per_head              =              args.size_per_head
    inter_size                 =                 args.inter_size
    vocab_size                 =                 args.vocab_size
    beam_width                 =                 args.beam_width
    top_k                      =                      args.top_k
    top_p                      =                      args.top_p
    temperature                =                args.temperature
    tensor_para_size           =           args.tensor_para_size
    pipeline_para_size         =         args.pipeline_para_size
    start_id                   =                   args.start_id
    end_id                     =                     args.end_id
    max_batch_size             =             args.max_batch_size
    max_seq_len                =                args.max_seq_len
    return_cum_log_probs       =       args.return_cum_log_probs
    return_output_length       =        return_cum_log_probs > 0
    vocab_file                 =                 args.vocab_file
    tokenizer_file             =             args.tokenizer_file
    

    print(" -Arguments- ".center(119, "="))
    for arg in vars(args):
        val = getattr(args, arg)
        if val == None:
            val = 'None'
        print("{:<40}: {:>77}".format(arg, val))
    print("".center(119, "="))

    # Inputs
    start_ids = []
    start_lengths = []
    print(" -Prepare inputs- ".center(119, "=")) 
    tokenizer = GPTTokenizer(
        vocab_dir="/data3/StarCoderBase/"
        )
    
    tokens = tokenizer.encode("""import torch
import os
import re
import json
import numpy as np
import pathlib 
import """)
    print(" -input- ".center(119, "="))
    print(tokenizer.decode(tokens))
    
    start_ids.append(tokens)
    start_ids = torch.IntTensor(start_ids)
    start_lengths.append(len(tokens))
    start_lengths = torch.IntTensor(start_lengths)
    
    batch_size = max_batch_size
    beam_search_diversity_rate = args.beam_search_diversity_rate
    len_penalty = args.len_penalty
    repetition_penalty = args.repetition_penalty
    min_length = args.min_length
    random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
    bad_words_list = [] 
    
    print(" -Prepare inputs DONE- ".center(119, "="))

    # Prepare model.
    star_coder = StarCoder(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
                           max_seq_len, tensor_para_size, pipeline_para_size, lib_path=args.lib_path,
                           inference_data_type=args.inference_data_type,
                           weights_data_type=args.weights_data_type)
    if not star_coder.load(ckpt_path=args.ckpt_path):
        print("[WARNING] Checkpoint file not found. Model loading is skipped.")

    with torch.no_grad():
        # Generate tokens.
        tokens_batch = star_coder(
                    start_ids=start_ids,
                    start_lengths=start_lengths,
                    output_len=output_len,
                    beam_width=beam_width,
                    top_k=top_k * torch.ones(size=[batch_size], dtype=torch.int32),
                    top_p=top_p * torch.ones(size=[batch_size], dtype=torch.float32),
                    beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[batch_size], dtype=torch.float32),
                    temperature=temperature * torch.ones(size=[batch_size], dtype=torch.float32),
                    len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    repetition_penalty=repetition_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
                    min_length=min_length * torch.ones(size=[batch_size], dtype=torch.int32),
                    random_seed=random_seed_tensor,
                    bad_words_list=bad_words_list,
                    return_output_length=return_output_length,
                    return_cum_log_probs=return_cum_log_probs)
        if return_cum_log_probs > 0:
            tokens_batch, _, cum_log_probs = tokens_batch
            print('[INFO] Log probs of sentences:', cum_log_probs)
        
        
        import pdb;pdb.set_trace()
        # only a thread (rank 0) gets the output, while the others are supposed to return None.
        if tokens_batch is not None:
            outputs = []
            tokens_batch = tokens_batch.cpu().numpy()
            for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
                for beam_id in range(beam_width):
                    token = tokens[beam_id][start_lengths[i]:]  # exclude context input from the output
                    if args.skip_end_tokens:
                        print('skip eos', len(tokens[beam_id]), start_lengths[i], len(token), len(token[token != end_id]))
                        token = token[token != end_id]
                    output = enc.decode(token) if args.detokenize else ' '.join(str(t) for t in token.tolist())
                    outputs.append(output)
                    print(f"[INFO] batch {i}, beam {beam_id}: \n[Context]\n{context}\n\n[Output]\n{output}\n")

            if args.sample_output_file:
                with open(args.sample_output_file, "w+") as f:
                    outputs = [o.replace("\n", "\\n") for o in outputs]
                    f.writelines("\n".join(outputs))

        # Measure inference time.
        if args.time:
            iterations = 10
            # warmup
            for i in range(iterations):
                tokens_batch = StarCoder(
                    start_ids=start_ids,
                    start_lengths=start_lengths,
                    output_len=output_len,
                    beam_width=beam_width)

            batch_num = 0
            token_num = 0
            time = timeit.default_timer()
            for i in range(iterations):
                tokens_batch = StarCoder(
                    start_ids=start_ids,
                    start_lengths=start_lengths,
                    output_len=output_len,
                    beam_width=beam_width)
                batch_num += 1
                for j, tokens in enumerate(tokens_batch):
                    token_num += tokens.shape[-1] - start_lengths[j]
            time_elapsed = timeit.default_timer() - time
            throughput = token_num / time_elapsed
            print(f"[INFO] FT-GPT generates {batch_num} batches, taking {time_elapsed:0.3f} secs "
                  f"to generate {token_num} tokens, {throughput:0.3f} tokens/sec.")


if __name__ == '__main__':
    main()