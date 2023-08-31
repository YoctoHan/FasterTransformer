import os
import sys
import torch

import numpy as np
import regex as re

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE
    # tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

class Vocab:
    def __init__(self, path):
        import json
        with open(path, "r") as f:
            self.vocab = json.load(f)
        self.tokens = [""] * len(self.vocab)
        for key, value in self.vocab.items():
            self.tokens[value] = key
    
    def get_idx(self, tokens):
        if type(tokens) == str:
            if len(tokens) > 2 and tokens[0] == "[" and tokens[-1] == "]":
                tokens = tokens[1:-1]
            tokens = tokens.split(", ")
            tokens = map(lambda t: t.strip(), tokens)
        return list(map(lambda t: self.vocab[t], tokens))

    def get_tokens(self, idx):
        if isinstance(idx, (int, np.int64)):
            return self.tokens[idx]
        return list(map(lambda i: self.tokens[i], idx))


@lru_cache()
def bytes_to_unicode():
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
  
class GPTTokenizer(object):

    def __init__(self, vocab_dir, errors='replace', max_len=None, print_info=False):
        
        tokenizer_info = torch.load(os.path.join(vocab_dir, "aixTokenizer"), map_location="cpu")
        self.encoder = tokenizer_info.get("vocab")
        bpe_data = tokenizer_info.get("merges_files")
        special_tokens_dict = tokenizer_info.get("special_tokens_dict")
        if self.encoder is None or bpe_data is None or special_tokens_dict is None:
            raise ValueError("format in aixTokenizer was wrong")
        
        self.max_len = max_len if max_len is not None else int(1e12)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.print_info = print_info

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens_dict["additional_special_tokens"])
        
        self.bos_token = special_tokens_dict["bos_token"]
        self.eos_token = special_tokens_dict["eos_token"]
        self.unk_token = special_tokens_dict["unk_token"]

        self.eos_id = self.encoder.get("<|endoftext|>", 0)
        self.bos_id = self.encoder.get(self.bos_token, 0)
        self.unk_id = self.encoder.get(self.unk_token, 0)
        
        self.pad_tok_id = self.encoder.get("<fim_pad>", 0)
        self.prefix_tok_id = self.encoder["<fim_prefix>"]
        self.middle_tok_id = self.encoder["<fim_middle>"]
        self.suffix_tok_id = self.encoder["<fim_suffix>"]

    def __len__(self):
        return len(self.encoder)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        
        self.special_tokens = dict((tok, self.encoder.get(tok, -1))
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        if self.print_info:
            print("Special tokens {}".format(self.special_tokens))

    def token2id(self, token: str):
        if token not in self.encoder:
            print(f"WARNING: {token} was not in vocabulary, use UNK id {self.encoder.get(self.unk_token, 0)}")
            return self.encoder.get(self.unk_token, 0)
        else:
            return self.encoder.get(token, 0)
    
    def id2token(self, id_: int):
        if id_ >= len(self.encoder):
            print(f"WARNING: {id_} is larger than vocabulary, return None")
            return ""
        else:
            return self.decoder.get(id_, "")
        
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except BaseException:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            print(
                "WARNING: Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(
                    len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text: str, file_path="", bos=False, eos=False):
        
        if len(file_path) > 0:
            ids_ = self.convert_tokens_to_ids(self.tokenize(file_path + "\n" + text))
            ids_ = [self.encoder["<filename>"]] + ids_
        else:
            ids_ = self.convert_tokens_to_ids(self.tokenize(text))
        
        if bos:
            ids_ = [self.bos_id] + ids_
        if eos:
            ids_ = ids_ + [self.eos_id]
        
        return ids_
    
    def encode_span(self, pre_context: str, post_context: str, file_path=""):
        assert isinstance(pre_context, str)
        assert isinstance(post_context, str)
        
        if len(file_path) > 0:
            pre_context = file_path + "\n" + pre_context
            res_ = [self.special_tokens["<fim_prefix>"], self.encoder["<filename>"]]
        else:
            res_ = [self.special_tokens["<fim_prefix>"]]
        
        if len(pre_context) > 0:
            res_.extend(self.encode(pre_context))
        res_.append(self.special_tokens["<fim_suffix>"])
        if len(post_context) > 0:
            res_.extend(self.encode(post_context))
        res_.append(self.special_tokens["<fim_middle>"])
        return res_

    def encode_chat(self, contents, env_prompts=None):
        """
            [
                {
                    "content": "Is it possible to imagine a society without law?",
                    "role": "user",
                },
                {
                    "content": "It is difficult to imagine a society...",
                    "role": "assistant",
                },
                {
                    "content": 'It seems like you consider the absence of law equal...',
                    "role": "user",
                },
                ...
            ]
        """
        assert self.encoder.get(f"<|assistant|>") is not None, f"vocab is not Chat model"
        system_msg = "Below is a dialogue between a human and an AI assistant called AixChat, which trained by aiXcoder. The AixChat tries to be helpful, polite, honest, sophisticated, and humble-but-knowledgeable. The AixChat is happy to help with almost anything about programming tasks, and will do its best to understand exactly what is needed, and will respond in details in Chinese. It also tries to avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful."
        prompt = [self.encoder["<|system|>"]] + self.encode("\n" + system_msg) + [self.encoder["<|end|>"]] + self.encode("\n")
        last_is_user = False
        
        if env_prompts is not None:
            for message in env_prompts:
                if message["role"] == "user":
                    prompt += [self.encoder["<|user|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = True
                else:
                    prompt += [self.encoder["<|assistant|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = False
        
        if contents is not None:
            for message in contents:
                if message["role"] == "user":
                    prompt += [self.encoder["<|user|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = True
                else:
                    prompt += [self.encoder["<|assistant|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = False
        
        if last_is_user:
            prompt += [self.encoder["<|assistant|>"]]
        else:
            prompt += [self.encoder["<|user|>"]] + self.encode("\nPlease continue") + [self.encoder["<|end|>"]] + self.encode("\n") +  [self.encoder["<|assistant|>"]]
        
        return prompt
    
    def decode(self, tokens):
        text = ''.join(self.convert_ids_to_tokens(tokens))
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
