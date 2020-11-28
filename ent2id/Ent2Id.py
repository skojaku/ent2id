import numpy as np
from scipy import sparse

import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

from tqdm import tqdm
import re
from collections.abc import Iterable


class Ent2Id:
    """
    Mapping entity name to ids based on character n-gram
    model = Ent2Id()
    model.fit(inst_table["Name"].values, inst_table["ID"].values)
    ids, score = model.predict("University of Queensland", max_ret_num = 20)
    """

    def __init__(self, ns=2, nf=5, languages="all"):
        self.ns = ns
        self.nf = nf

        self.subword_mode = "word"
        self.init_stop_words(languages)

        self.clear()

    def clear(self):
        self.ent_list = []
        self.subword2id = {}
        self.link_list = []
        self.W = None

    def init_stop_words(self, languages):

        nltk.download("stopwords")

        if languages == "all":
            languages = [
                "dutch",
                "german",
                "slovene",
                "hungarian",
                "romanian",
                "kazakh",
                "turkish",
                "russian",
                "README",
                "italian",
                "english",
                "greek",
                "tajik",
                "norwegian",
                "portuguese",
                "finnish",
                "danish",
                "french",
                "swedish",
                "azerbaijani",
                "spanish",
                "indonesian",
                "arabic",
                "nepali",
            ]
        if not isinstance(languages, list):
            languages = [languages]

        self.stopwords = []
        for lang in languages:
            self.stopwords += stopwords.words(lang)
        self.stopwords = set(self.stopwords)

    def add_stop_words(self, additional_words):
        for w in additional_words:
            self.stopwords.add(w)

    def fit(self, ent_name_list, id_list):
        self.clear()
        self.partial_fit(ent_name_list, id_list)
        self.make_ent2id_mapping()

    def partial_fit(self, ent_name_list, id_list):

        # Construct vocabrary
        self.W = None
        self.ent_list += list(id_list)
        for i, ent_name in enumerate(tqdm(ent_name_list, desc="Generating vocabulary")):
            words = self.generate_ngrams_range(ent_name, self.ns, self.nf)
            for word in words:
                self.subword2id[word] = self.subword2id.get(word, len(self.subword2id))
                self.link_list += [(self.subword2id[word], id_list[i])]
        return self

    def make_ent2id_mapping(self):

        uid_list = np.unique(self.ent_list)
        self.id2ent = dict(zip(np.arange(len(self.ent_list)), self.ent_list))
        self.nword = len(self.subword2id)
        self.nent = len(self.ent_list)

        # Make character ngram vs entity matrix
        link_list = np.array(self.link_list)
        self.W = sparse.csr_matrix(
            (np.ones(link_list.shape[0]), (link_list[:, 0], link_list[:, 1])),
            shape=(self.nword, self.nent),
        )
        self.B = self.W.copy()
        self.B.data = np.ones_like(self.B.data)

        self.dr = np.array(self.W.sum(axis=1)).reshape(-1)
        self.dc = np.array(self.W.sum(axis=0)).reshape(-1)
        self.denom = np.sum(self.dc)

        def safe_log(v):
            return np.log(np.maximum(v, 1e-12))

        self.minlog = np.log(1e-12)
        r, c, v = sparse.find(self.W)
        v = safe_log(v) - (
            safe_log(self.dr[r]) + safe_log(self.dc[c]) - safe_log(self.denom)
        )
        v += -self.minlog
        self.W = sparse.csr_matrix((v, (r, c)), shape=(self.nword, self.nent),)

    def predict(self, input_text_list):
        if self.W is None:
            self.make_ent2id_mapping()

        if isinstance(input_text_list, str):
            input_text_list = [input_text_list]

        word_ids_list = []
        for input_text in input_text_list:
            word_ids = np.array(
                [
                    self.subword2id.get(s, np.nan)
                    for s in self.generate_ngrams_range(input_text, self.ns, self.nf)
                ]
            )
            word_ids = word_ids[~np.isnan(word_ids)]
            word_ids_list += [word_ids]

        score = self.calc_score(word_ids_list)
        hits = np.array(score.argmax(axis=1)).reshape(-1)
        return (
            [self.id2ent[h] for h in hits],
            np.array([score[i, h] for i, h in enumerate(hits)]),
        )

    def calc_score(self, word_ids_list):
        query_ids = np.concatenate(
            [np.ones(len(word_ids)) * k for k, word_ids in enumerate(word_ids_list)]
        ).astype(int)
        word_ids = np.concatenate(word_ids_list).astype(int)

        V = sparse.csr_matrix(
            (np.ones_like(word_ids), (query_ids, word_ids)),
            shape=(len(word_ids_list), self.nword),
        )
        V.data = np.ones_like(V.data)

        score = V @ self.W
        return score

    # Function to generate n-grams from sentences.
    def generate_word_ngrams(self, s, num):
        # Convert to lowercases
        s = s.lower()
        s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
        s = re.sub(r" +", " ", s)
        tokens = [
            token
            for token in s.split(" ")
            if (token != "") and (not token in self.stopwords)
        ]
        return [" ".join(token) for token in list(ngrams(tokens, num))]

    def generate_ngrams(self, s, n):
        # Convert to lowercases
        s = s.lower()

        # Replace all none alphanumeric characters with spaces
        s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
        s = re.sub(r" +", " ", s)
        return ["".join(n) for n in ngrams(s, n)]

    def generate_ngrams_range(self, s, ns, nf):
        retval = []
        for n in np.arange(ns, nf + 1):
            if self.subword_mode == "word":
                retval += self.generate_word_ngrams(s, n)
            elif self.subword_mode == "char":
                retval += self.generate_ngrams(s, n)
        return retval
