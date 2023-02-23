# -*- coding: utf-8 -*-

from collections import Counter

import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import json
from gensim.models import Word2Vec

nltk.download('wordnet')

ENCODING = 'utf-8'

# Play with these for different experiments:
EXPERIMENT_PARAMS = {
    'sif_alpha': 0.1,  # any float
    'min_word_count': 1,  # any integer
    'word_embedding': 'ada',  # word2vec / glove / ada
    'target': 'simple_topics',  # difficulty / simple_topics / story
    'input_text': 'title'  # body / title
}

STORY_WORDS = [
    'exam', 'red', 'her', 'turn', 'act', 'win', 'order', 'ice', 'sit', 'eat', 'ant', 'ring', 'cat', 'sent', 'rent',
    'time', 'star', 'rule', 'dice', 'sea', 'cut', 'ask', 'age', 'art', 'point', 'direction', 'bar', 'king', 'man',
    'play', 'class', 'gene', 'ping', 'reach', 'tie', 'rob', 'person', 'son', 'lie', 'west', 'lock', 'board', 'work',
    'car', 'she', 'rest', 'lane', 'city', 'game', 'table', 'lose', 'free', 'sell', 'buy', 'die', 'customer', 'cost',
    'color', 'front', 'student', 'seat', 'eat', 'room', 'language', 'clock', 'trip', 'white', 'ball', 'box', 'road',
    'angle', 'money', 'unit', 'circular', 'press', 'train', 'visit', 'site', 'water', 'date', 'north', 'walk', 'south',
    'employee', 'price', 'pay', 'power', 'day', 'player', 'media', 'Alice', 'Bob', 'children', 'quest', 'friend',
    'face', 'minute', 'hour', 'higher', 'cycle', 'pile', 'store', 'vision', 'travel', 'wall', 'cities', 'outside',
    'collect', 'express', 'candies', 'blue', 'house', 'rounded', 'broken', 'chess', 'shape', 'image', 'cab', 'robot',
    'stock', 'member', 'colored', 'speed', 'night', 'building', 'obstacle', 'company', 'course', 'shop', 'candy',
    'sold', 'banana', 'street', 'paint', 'salary', 'boring', 'drive', 'puzzle', 'book', 'doll', 'fee', 'garden',
    'dollar', 'jump', 'pancake'
]


def topic(q):
    title = q['title']
    body = q['body']
    if 'tree' in title or 'tree' in body or 'root' in title or 'root' in body or 'repeat' in title or 'repeat' in body:
        return 0
    if 'bit' in title or 'bit' in body or 'xor' in title or 'xor' in body or 'binary' in title or 'binary' in body:  # keep after tree
        return 1
    if 'list' in title or 'list' in body:
        return 2
    if 'array' in title or 'array' in body:
        return 3
    if 'minimum' in title or 'minimum' in body or 'maximum' in title or 'maximum' in body:
        return 4
    if 'string' in title or 'string' in body:
        return 5
    return 6


def is_in_sentence(words, sentence):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    sentence_tokens = nltk.word_tokenize(sentence)
    sentence_lemmas = [lemmatizer.lemmatize(t.lower()) for t in sentence_tokens]
    word_tokens = nltk.word_tokenize(' '.join(words))
    word_lemmas = [lemmatizer.lemmatize(t.lower()) for t in word_tokens]
    return any([t in sentence_lemmas for t in word_lemmas])


class Experiment:
    def __init__(self, data_path):
        for k, v in EXPERIMENT_PARAMS.items():
            print(f'{k}: {v}')
        self.data_path = data_path
        self._load_questions()
        self._build_word_embeddings()
        self._load_target()
        self._print_statistics()

    def _load_questions(self):
        with open(self.data_path + 'leetcode_questions_parsed.json', 'r') as f:
            self.leetcode_json = json.load(f)
        self.input_text = EXPERIMENT_PARAMS['input_text']
        self.questions = [v[self.input_text] for k, v in self.leetcode_json.items()]
        self.questions = [q.lower() for q in self.questions]

    def _build_word_embeddings(self):
        self.word_embedding = EXPERIMENT_PARAMS['word_embedding']
        self.sif_alpha = EXPERIMENT_PARAMS['sif_alpha']
        self.min_word_count = EXPERIMENT_PARAMS['min_word_count']
        tokenized_questions = [nltk.word_tokenize(q) for q in self.questions]
        flattened_questions = [word for q in tokenized_questions for word in q]
        word_counts = Counter(flattened_questions)
        word_counts = {k: word_counts[k] for k in word_counts if word_counts[k] >= self.min_word_count}
        total_count = sum([v for v in word_counts.values()])
        word_props = {k: word_counts[k] / total_count for k in word_counts}
        word_weights = {k: self.sif_alpha / (self.sif_alpha + word_props[k]) for k in word_props}
        self.vocabulary_size = len(word_weights)
        if self.word_embedding == 'ada':
            if self.input_text == 'body':
                file_name = 'leetcode_ada_embeddings4.json'
            elif self.input_text == 'title':
                file_name = 'leetcode_titles_ada_embeddings4.json'
            else:
                raise Exception(f'ada embedding not supported for {self.input_text}')
            with open(self.data_path + file_name, 'r') as f:
                embeddings_json = json.load(f)
            self.X = np.array([v for k, v in embeddings_json.items()])
            return
        elif self.word_embedding == 'word2vec':
            word_embeddings = Word2Vec(
                sentences=tokenized_questions, size=100, window=5, min_count=self.min_word_count, workers=4)
        elif self.word_embedding == 'glove':
            word_embeddings = {}
            with open(self.data_path + "glove.42B.300d.txt", 'r', encoding=ENCODING) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    word_embeddings[word] = vector
        else:
            raise Exception(f'Bad word embedding: {self.word_embedding}')
        question_vectors = np.array([sum([word_weights[w] * word_embeddings[w]
                                          for w in q if w in word_weights and w in word_embeddings])
                                     for q in tokenized_questions])
        pca_model = PCA(n_components=1)
        pca_model.fit(question_vectors)
        pca_component = pca_model.components_
        vectors_minus_component = question_vectors - question_vectors.dot(pca_component.transpose()) * pca_component
        scaler = MinMaxScaler()
        self.X = scaler.fit_transform(vectors_minus_component)

    def _load_target(self):
        self.target = EXPERIMENT_PARAMS['target']
        if self.target == 'difficulty':
            labels = [v['label'] for k, v in self.leetcode_json.items()]
            label_to_int_dict = {'Easy': 0, 'Medium': 1, 'Hard': 2}
            self.y = np.array([label_to_int_dict[t] for t in labels])
        elif self.target == 'simple_topics':
            self.y = np.array([topic(v) for k, v in self.leetcode_json.items()])
        elif self.target == 'story':
            titles_and_bodies = [f"{v['title']} {v['body']}" for k, v in self.leetcode_json.items()]
            self.y = np.array([int(is_in_sentence(STORY_WORDS, q)) for q in titles_and_bodies])
        else:
            raise Exception(f'Bad target: {self.target}')

    def _print_statistics(self):
        print('Number of texts:', len(self.questions))
        print('Average number of tokens per text: ', sum([len(q) for q in self.questions]) / len(self.questions))
        print('Vocabulary size: ', self.vocabulary_size)
        titles = np.array([v['title'] for k, v in self.leetcode_json.items()])
        for c in np.unique(self.y):
            print(f'Texts in class {c}: ', sum(self.y == c))
            examples = np.random.choice(titles[self.y == c], size=5, replace=False)
            print(f'Examples for class {c}:')
            for i, e in enumerate(examples):
                print(e)
            print()


def load_leetcode(data_path='data/leetcode/'):
    experiment = Experiment(data_path)
    return experiment.X, experiment.y


def load_stackoverflow(data_path='data/stackoverflow/'):

    # load SO embedding
    with open(data_path + 'vocab_withIdx.dic', 'r', encoding=ENCODING) as inp_indx, \
            open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r', encoding=ENCODING) as inp_dic, \
            open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        emb_index = inp_dic.readlines()
        emb_vec = inp_vec.readlines()
        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec):
            word = index_word[index.replace('\n', '')]
            word_vectors[word] = np.array(list((map(float, vec.split()))))

        del emb_index
        del emb_vec

    with open(data_path + 'title_StackOverflow.txt', 'r', encoding=ENCODING) as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    pca = PCA(n_components=1)
    pca.fit(all_vector_representation)
    pca = pca.components_

    XX1 = all_vector_representation - all_vector_representation.dot(pca.transpose()) * pca

    XX = XX1

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    with open(data_path + 'label_StackOverflow.txt') as label_file:
        y = np.array(list((map(int, label_file.readlines()))))
        print(y.dtype)

    return XX, y


def load_search_snippet2(data_path='data/SearchSnippets/'):
    mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'SearchSnippets_vocab2idx.dic', 'r') as inp_indx:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'SearchSnippets.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        all_lines = [line for line in all_lines]
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(12340, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_

    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_biomedical(data_path='data/Biomedical/'):
    mat = scipy.io.loadmat(data_path + 'Biomedical-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    rand_seed = 0

    # load SO embedding
    with open(data_path + 'Biomedical_vocab2idx.dic', 'r') as inp_indx:
        # open(data_path + 'vocab_emb_Word2vec_48_index.dic', 'r') as inp_dic, \
        # open(data_path + 'vocab_emb_Word2vec_48.vec') as inp_vec:
        pair_dic = inp_indx.readlines()
        word_index = {}
        for pair in pair_dic:
            word, index = pair.replace('\n', '').split('\t')
            word_index[word] = index

        index_word = {v: k for k, v in word_index.items()}

        del pair_dic

        word_vectors = {}
        for index, vec in zip(emb_index, emb_vec.T):
            word = index_word[str(index)]
            word_vectors[word] = vec

        del emb_index
        del emb_vec

    with open(data_path + 'Biomedical.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:-1]
        # print(sum([len(line.split()) for line in all_lines])/20000) #avg length
        text_file = " ".join([" ".join(nltk.word_tokenize(c)) for c in all_lines])
        word_count = Counter(text_file.split())
        total_count = sum(word_count.values())
        unigram = {}
        for item in word_count.items():
            unigram[item[0]] = item[1] / total_count

        all_vector_representation = np.zeros(shape=(20000, 48))
        for i, line in enumerate(all_lines):
            word_sentence = nltk.word_tokenize(line)

            sent_rep = np.zeros(shape=[48, ])
            j = 0
            for word in word_sentence:
                try:
                    wv = word_vectors[word]
                    j = j + 1
                except KeyError:
                    continue

                weight = 0.1 / (0.1 + unigram[word])
                sent_rep += wv * weight
            if j != 0:
                all_vector_representation[i] = sent_rep / j
            else:
                all_vector_representation[i] = sent_rep

    svd = TruncatedSVD(n_components=1, random_state=rand_seed, n_iter=20)
    svd.fit(all_vector_representation)
    svd = svd.components_
    XX = all_vector_representation - all_vector_representation.dot(svd.transpose()) * svd

    scaler = MinMaxScaler()
    XX = scaler.fit_transform(XX)

    return XX, y


def load_data(dataset_name):
    print('load data')
    if dataset_name == 'stackoverflow':
        return load_stackoverflow()
    elif dataset_name == 'biomedical':
        return load_biomedical()
    elif dataset_name == 'search_snippets':
        return load_search_snippet2()
    elif dataset_name == 'leetcode':
        return load_leetcode()
    else:
        raise Exception('dataset not found...')
