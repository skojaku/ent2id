import numpy as np
from scipy import sparse
import numba
from numba import types
from numba.typed import Dict
from collections import Counter

#
import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

from tqdm import tqdm
import re
from collections.abc import Iterable

from joblib import Parallel, delayed


class Ent2Id:
    """
    Mapping entity name to ids based on character n-gram
    model = Ent2Id()
    model.fit(inst_table["Name"].values, inst_table["ID"].values)
    ids, score = model.predict("University of Queensland", max_ret_num = 20)
    """

    def __init__(self, ns=2, nf=5, languages="all", subword = "word", aggregate_duplicates = True):
        self.ns = ns
        self.nf = nf

        self.subword_mode = subword
        self.init_stop_words(languages)
        self.n_jobs = 10
        self.aggregate_duplicates = aggregate_duplicates


        self.clear()

    def clear(self):
        self.subword2id = {}
        self.ent2id = {}
        self.id2ent = {}
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

        # Add to dictionary
        W = None
        # Construct vocabrary
        for i, ent_name in enumerate(tqdm(ent_name_list, desc="Generating vocabulary")):
            words = generate_ngrams_range(
                ent_name, self.ns, self.nf, self.subword_mode, self.stopwords
            )
            n = len(self.ent2id)

            if self.aggregate_duplicates is False:
                id_list[i] = id_list[i] + "____%d" % n

            self.ent2id[id_list[i]] = self.ent2id.get(id_list[i], n)
            j = self.ent2id[id_list[i]]
            for word in words:
                self.subword2id[word] = self.subword2id.get(word, len(self.subword2id))
                self.link_list += [(self.subword2id[word], j)]
        return self

    def make_ent2id_mapping(self):

        self.id2ent = {value: key for key, value in self.ent2id.items()}
        self.nword = len(self.subword2id)
        self.nent = len(self.id2ent)

        # Make character ngram vs entity matrix
        link_list = np.array(self.link_list)
        self.W = sparse.csr_matrix(
            (np.ones(link_list.shape[0]), (link_list[:, 0], link_list[:, 1])),
            shape=(self.nword, self.nent),
        )
        self.link_list = []  # clear list

        self.dr = np.array(self.W.sum(axis=1)).reshape(-1)
        self.dc = np.array(self.W.sum(axis=0)).reshape(-1)
        self.denom = np.sum(self.dc)

        def safe_log(v):
            return np.log(np.maximum(v, 1e-34))

        r, c, v = sparse.find(self.W)
        v = safe_log(v) - (
            safe_log(self.dr[r] / self.denom) + safe_log(self.dc[c])
        )
        v += 1e-3  # offset log(v) is actually zero but it will be treated as nothing. Therefore add a small value as an offset
        self.W = sparse.csr_matrix((v, (r, c)), shape=(self.nword, self.nent),)
        self.col_penalty = np.array(self.W.sum(axis=0)).reshape(-1)

    def predict(self, input_text_list):
        if self.W is None:
            self.make_ent2id_mapping()

        if isinstance(input_text_list, str):
            input_text_list = [input_text_list]

        word_ids_list = []
        query_ids = []
        for qid, input_text in enumerate(input_text_list):
            word_ids = np.array(
                [
                    self.subword2id.get(s, np.nan)
                    for s in generate_ngrams_range(
                        input_text, self.ns, self.nf, self.subword_mode, self.stopwords
                    )
                ]
            )
            word_ids = np.unique(word_ids[~np.isnan(word_ids)])
            word_ids_list += [word_ids.astype(int)]
            query_ids += [qid * np.ones(len(word_ids))]

        # Calculate Score
        query_ids = np.concatenate(query_ids).astype(int)
        word_ids = np.concatenate(word_ids_list).astype(int)
        V = sparse.csr_matrix(
            (np.ones_like(word_ids), (query_ids, word_ids)),
            shape=(len(word_ids_list), self.nword),
        )
        score = V @ self.W
        entids, score = row_argmax(
            score.data, score.indices, score.indptr, score.shape[0], score.shape[1], self.col_penalty
        )
        entids = [self.id2ent[i] for i in entids]
        print(entids)
        if self.aggregate_duplicates is False:
            entids = [s.split("___")[0] for s in entids]
        return entids, score

    def predict_test(self, input_text_list):
        if self.W is None:
            self.make_ent2id_mapping()

        if isinstance(input_text_list, str):
            input_text_list = [input_text_list]

        word_ids_list = []
        for input_text in input_text_list:
            word_ids = np.array(
                [
                    self.subword2id.get(s, np.nan)
                    for s in generate_ngrams_range(
                        input_text, self.ns, self.nf, self.subword_mode, self.stopwords
                    )
                ]
            )
            word_ids = word_ids[~np.isnan(word_ids)]
            word_ids_list += [word_ids.astype(int)]

        return self.find_best_match(word_ids_list)


# Function to generate n-grams from sentences.
def generate_word_ngrams(s, num, stopwords):
    # Convert to lowercases
    s = s.lower()
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r" +", " ", s)
    tokens = [
        token for token in s.split(" ") if (token != "") and (not token in stopwords)
    ]
    return [" ".join(token) for token in list(ngrams(tokens, num))]


def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r" +", " ", s)
    return ["".join(n) for n in ngrams(s, n)]


def generate_ngrams_range(s, ns, nf, subword_mode, stopwords):
    retval = []
    for n in np.arange(ns, nf + 1):
        if subword_mode == "word":
            retval += generate_word_ngrams(s, n, stopwords)
        elif subword_mode == "char":
            retval += generate_ngrams(s, n)
    return retval


@numba.jit(nopython=True, fastmath=True)
def row_argmax(S_data, S_indices, S_indptr, Nr, Nc, penalty):
    hits = np.zeros(Nr)
    s = np.zeros(Nr)
    for i in range(Nr):
        w = S_data[S_indptr[i] : S_indptr[i + 1]]
        nei = S_indices[S_indptr[i] : S_indptr[i + 1]]
        if len(w) == 0:
            continue
        ind = np.argmax(2 * w - penalty[nei])
        hits[i] = S_indices[S_indptr[i] + ind]
        s[i] = w[ind]
    return hits, s
