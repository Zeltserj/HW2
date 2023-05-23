from typing import Optional, List
import torch
from torch.utils.data import Dataset


class Vocab:
    def __init__(
        self,
        train_path,
        pretrained_word_path: Optional[str] = None,
        seperator="\t",
        sub_word=False,
        chars=False,
    ):
        self.PAD = "<PAD>"
        self.UNK = "UUUNKKK"
        self.OUTSIDE = "O"
        self.PAD_TAG = "PAD_TAG"
        self.TEST_DATAPOINT = "TEST_TAG"
        self.tags = {self.PAD_TAG, self.TEST_DATAPOINT}
        self.last_pretrained_idx = 0
        self.special_tokens = [self.PAD, self.UNK]
        self.words = set()
        self.chars = set()
        self.W2I = {}
        self.I2W = {}
        self.sub_word = sub_word
        if pretrained_word_path:
            with open(pretrained_word_path) as pretrainedfile:
                for i, line in enumerate(pretrainedfile.readlines()):
                    word = line.strip()
                    if word:
                        self.words.add(word)
                        self.W2I[word] = i
                        self.I2W[i] = word

            self.last_pretrained_idx = len(self.words)

        with open(train_path) as trainsetfile:
            next_idx = self.last_pretrained_idx
            for line in trainsetfile.readlines():
                if line != "\n":
                    word, tag = line.split(seperator)
                    word = word.lower()
                    if word not in self.words:
                        self.words.add(word)
                        self.W2I[word] = next_idx
                        self.I2W[next_idx] = word
                        next_idx += 1
                    self.tags.add(tag.strip())
        self.T2I, self.I2T = self._indexise(self.tags)

        self.W2I.update({self.PAD: len(self.words)})
        self.I2W.update({len(self.words): self.PAD})

        self.words.update({self.PAD})
        if self.UNK not in self.words:
            self.words.add(self.UNK)
            self.W2I[self.UNK] = len(self.words)
            self.I2W[len(self.words)] = self.UNK

        if self.sub_word:
            self.prefixes = {self.PAD}
            self.suffixes = {self.PAD}
            for word in self.words:
                if len(word) > 2:
                    self.prefixes.add(word[:3])
                    self.suffixes.add(word[-3:])
            self.P2I, self.I2P = self._indexise(self.prefixes)
            self.S2I, self.I2S = self._indexise(self.suffixes)
        if chars:
            for word in self.words:
                self.chars.update(set(word))
            self.C2I, self.I2C = self._indexise(self.chars)

    def __len__(self):
        return len(self.words)

    def __repr__(self) -> str:
        return f"Vocab with {len(self.words)} words and {len(self.tags)} tags\n \
            {self.last_pretrained_idx} are from pretrained embeddings"

    def _indexise(self, token_set):
        token_to_index = {token: i for i, token in enumerate(token_set)}
        index_to_token = {i: token for i, token in enumerate(token_set)}
        return token_to_index, index_to_token


class TagData(Dataset):
    def __init__(
        self,
        filepath,
        vocab: Vocab,
        separator="\t",
        sub_word=False,
        test=False,
        chars=False,
    ) -> None:
        self.data = []
        self.tags = []
        self.suffix_data = []
        self.prefix_data = []
        self.separator = separator
        self.test = test
        self.sub_word = sub_word
        self.chars = chars
        with open(filepath) as f:
            tags = []
            words = []
            for line in f.readlines():
                if line != "\n":
                    if test:
                        word, tag = line.strip(), vocab.TEST_DATAPOINT
                    else:
                        word, tag = line.split(self.separator)
                        tag = tag.strip()
                    word = word.lower()
                    if word in vocab.words:
                        words.append(vocab.W2I[word])
                    else:
                        words.append(vocab.W2I[vocab.UNK])
                    tags.append(vocab.T2I[tag])
                else:
                    for i in range(len(words)):
                        window = []
                        tag_window = []
                        for j in range(-2, 3):
                            if i + j < 0:
                                self.append_word(window, vocab, vocab.W2I[vocab.PAD])
                                tag_window.append(vocab.T2I[vocab.PAD_TAG])
                            if i + j >= 0 and i + j < len(words):
                                self.append_word(window, vocab, words[i + j])
                                tag_window.append(tags[i + j])
                            if i + j >= len(words) and i + j <= len(words) + 2:
                                self.append_word(window, vocab, vocab.W2I[vocab.PAD])
                                tag_window.append(vocab.T2I[vocab.PAD_TAG])
                        self.data.append((torch.tensor(window), torch.tensor(tag_window[2])))

                    tags = []
                    words = []

    def create_word_tuple(self, word, word_idx, vocab: Vocab):
        if len(word) < 3 or word in vocab.special_tokens:
            return (vocab.P2I[vocab.PAD], word_idx, vocab.S2I[vocab.PAD])

        prefix = word[:3]
        suffix = word[-3:]

        if prefix not in vocab.prefixes:
            prefix_idx = vocab.W2I[vocab.UNK]
        else:
            prefix_idx = vocab.P2I[prefix]

        if suffix not in vocab.suffixes:
            suffix_idx = vocab.S2I[vocab.UNK]
        else:
            suffix_idx = vocab.S2I[suffix]

        return (prefix_idx, word_idx, suffix_idx)

    def append_word(self, window: List[str], vocab: Vocab, word_idx: int):
        word = vocab.I2W[word_idx]
        if self.sub_word or self.chars:
            word_tuple = self.create_word_tuple(word, word_idx, vocab)
            window.append(word_tuple)
        else:
            window.append(word_idx)

    def append_data(self, window, tag):
        if self.chars or self.sub_word:
            print(f"window[0]: {window[0]}")
            self.prefix_data.append(window[0])
            self.suffix_data.append(window[-1])
            self.data.append(window[1:-1])
        else:
            self.data.append(window)
        self.tags.append(tag)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
