from cmath import exp
from operator import is_
import os
import pandas as pd
import numpy as np
import json
import tqdm
from itertools import zip_longest

from colorUtil import set_color, COLOR

TAGS = ['geo', 'gpe', 'per', 'org', 'art', 'tim', 'eve', 'nat']


class NERDataSet:

    def __init__(self, data_path, split_rate=0.3, saveVocab=True):
        self.train_file = os.path.join(data_path, 'train.csv')
        self.test_file = os.path.join(data_path, 'test.csv')
        self.data_path = data_path
        self.vocab_file = os.path.join(self.data_path, 'vocab2id.json')
        self.save_path = 'results'
        self.split_rate = split_rate

        if not os.path.exists(self.vocab_file):
            self.word2id = self.build_vocab(saveVocab)
        else:
            with open(self.vocab_file, 'r') as fr:
                self.word2id = json.load(fr)

        # TODO: Tags to id
        tags = ['O'] + [pos + '-' + tag for pos in ['B', 'I'] for tag in TAGS]
        self.tag2id = {tag: i for i, tag in enumerate(tags)}
        self.id2tag = {i: tag for i, tag in enumerate(tags)}

        self.word2id['<pad>'] = len(self.word2id)
        self.word2id['<unk>'] = len(self.word2id) + 1

    def build_vocab(self, saveVocab) -> dict:
        print(set_color("[LOG] Build Vocab and save", COLOR.GREEN))

        df_train = pd.read_csv(self.train_file)
        sentences = [row['Sentence'] for _, row in df_train.iterrows()]
        del df_train
        df_test = pd.read_csv(self.test_file)
        sentences += [row['Sentence'] for _, row in df_test.iterrows()]
        del df_test

        vocab = set()
        for sen in sentences:
            vocab.update(set(sen.split()))

        word2idx = {word: idx for idx, word in enumerate(list(vocab))}
        if saveVocab:
            with open(os.path.join(self.data_path, 'vocab2id.json'), 'w') as f:
                json.dump(word2idx, f)

        return word2idx
    
    @property
    def vocab_size(self):
        return len(self.word2id)
    
    @property
    def target_size(self):
        return len(self.tag2id)

    def getTag(self, tagId):
        return self.id2tag(tagId)

    def build_dataSets(self):

        def convert_to_id(sentences, id_map):
            res = []
            for sen in sentences:
                assert isinstance(sen, str)
                res.append([id_map[item] for item in sen.split()])
            return res

        def genMask(length, maxVal, nparray, pad=self.word2id['<pad>']):
            mask = np.zeros(length) + maxVal
            for row, col in np.argwhere(nparray == pad):
                mask[row] = min(mask[row], col)
            return mask

        # Try load saved data
        res = self.loadorSavenp()
        if res is not None:
            return res

        # Load Sentences and Tags
        df_train = pd.read_csv(self.train_file)
        train_sentences = [row['Sentence'] for _, row in df_train.iterrows()]
        train_tags = [row['Tag'] for _, row in df_train.iterrows()]
        train_sentences = convert_to_id(train_sentences, self.word2id)
        train_tags = convert_to_id(train_tags, self.tag2id)
        del df_train
        train_sentences = np.array(
            list(zip_longest(*train_sentences,
                             fillvalue=self.word2id['<pad>']))).T
        train_tags = np.array(
            list(zip_longest(*train_tags, fillvalue=self.tag2id['O']))).T

        # Split train set and valid set
        valid_indexes = np.random.choice(train_sentences.shape[0],
                                         int(train_sentences.shape[0] * self.split_rate),
                                         replace=False)
        valid_sentences = train_sentences[valid_indexes, ...]
        valid_tags = train_tags[valid_indexes, ...]
        mask = np.ones(train_sentences.shape[0], bool)
        mask[valid_indexes] = False
        train_sentences = train_sentences[mask, ...]
        train_tags = train_tags[mask, ...]

        assert train_sentences.shape == train_tags.shape
        assert valid_sentences.shape == valid_tags.shape

        # Load Test set
        df_test = pd.read_csv(self.test_file)
        test_sentences = [row['Sentence'] for _, row in df_test.iterrows()]
        test_sentences = convert_to_id(test_sentences, self.word2id)
        del df_test
        test_sentences = np.array(
            list(zip_longest(*test_sentences,
                             fillvalue=self.word2id['<pad>']))).T
        # ! train seq_len not equal to test seq_len

        # Generate seq_length mask for LSTM
        train_mask = genMask(train_sentences.shape[0],
                             train_sentences.shape[1], train_sentences)
        valid_mask = genMask(valid_sentences.shape[0],
                             valid_sentences.shape[1], valid_sentences)
        test_mask = genMask(test_sentences.shape[0], test_sentences.shape[1],
                            test_sentences)

        print(
            set_color(
                f"Shape Train {train_tags.shape}, valid {valid_tags.shape}, test {test_sentences.shape}",
                COLOR.BLUE))
        res = [
            train_sentences, train_tags, train_mask, 
            valid_sentences, valid_tags, valid_mask, 
            test_sentences, test_mask
        ]
        self.loadorSavenp(nparrays=res, mode='save')
        return res

    def loadorSavenp(self, nparrays=None, mode='load'):
        np_files = [
            'train_s', 'train_tag', 'train_mask', 'valid_s', 'valid_tag',
            'valid_mask', 'test_s', 'test_mask'
        ]
        res = []
        if mode == 'load':
            if not os.path.exists(self.save_path):
                return None
            print(set_color(f"Load Saved data", COLOR.GREEN))
            for file in np_files:
                try:
                    with open(os.path.join(self.save_path, file + '.npy'),
                              'rb') as fr:
                        res.append(np.load(fr))
                except FileNotFoundError:
                    return None

            return res
        else:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            for idx, file in enumerate(np_files):
                with open(os.path.join(self.save_path, file + '.npy'),
                          'wb') as fw:
                    np.save(fw, nparrays[idx])

class NERDSWrapper:
    def __init__(self, ds_x, ds_y, ds_mask, is_test=False):
        self.x = ds_x
        self.y = ds_y
        self.mask = ds_mask
        self.is_test = is_test
        
    def __getitem__(self, idx):
        if self.is_test:
            return self.x[idx], self.mask[idx]

        return self.x[idx], self.y[idx], self.mask[idx]
    
    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    ds = NERDataSet('data/')
    # ds.build_vocab(saveVocab=True)
    ds.build_dataSets()
