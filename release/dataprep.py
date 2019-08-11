import json
import re
import numpy as np
import pickle
import os
import zipfile as zf
import fnmatch


class DataPreparation:

    def __init__(self,
                 test_input_file_path,
                 input_ngram=1,
                 output_ngram=1,
                 is_input_data_prepared=False,
                 is_input_data_in_json=False,
                 add_identification_data=False,
                 test_id_file_path=None,
                 is_out_sub_seq=False,
                 is_in_sub_seq=False,
                 max_batch_size=512,
                 load_from_file=None):
        self.test_input_file_path = test_input_file_path
        self.input_ngram = input_ngram
        self.output_ngram = output_ngram
        self.is_input_data_prepared = is_input_data_prepared
        self.is_input_data_in_json = is_input_data_in_json
        self.add_identification_data = add_identification_data
        self.test_id_file_path = test_id_file_path
        self.is_out_sub_seq = is_out_sub_seq
        self.is_in_sub_seq = is_in_sub_seq
        self.max_batch_size = max_batch_size
        self.load_from_file = load_from_file

        self.train_x = []
        self.train_y = []
        self.valid_x = []
        self.valid_y = []
        self.test_x = []
        self.test_y = []
        self.train_sub_seq_max_len = 0

        if self.load_from_file is not None:
            self._load_from_file()

        self._read_files()

        self._split_files()

        if self.is_input_data_in_json:
            self._load_input_from_json()

        if not self.is_input_data_prepared:
            self._prepare_input()
        else:
            self._define_input_vectors()

        self._add_in_vec_to_x()

        if self.add_identification_data:
            self._add_id_data()

        if self.is_in_sub_seq:
            self._flatten_sub_seq()

        if self.is_out_sub_seq:
            self._encode_sub_seq()

        self._split_to_batches()

    def _load_from_file(self):
        with zf.ZipFile(self.load_from_file, 'r') as zip:
            train_x_dir = fnmatch.filter(zip.namelist(), os.path.join('train_x', '*.pkl'))
            train_y_dir = fnmatch.filter(zip.namelist(), os.path.join('train_y', '*.pkl'))
            valid_x_dir = fnmatch.filter(zip.namelist(), os.path.join('valid_x', '*.pkl'))
            valid_y_dir = fnmatch.filter(zip.namelist(), os.path.join('valid_y', '*.pkl'))
            test_x_dir = fnmatch.filter(zip.namelist(), os.path.join('test_x', '*.pkl'))
            input_dict_dir = fnmatch.filter(zip.namelist(), os.path.join('input_dict', '*.pkl'))
            output_dict_dir = fnmatch.filter(zip.namelist(), os.path.join('output_dict', '*.pkl'))
            test_input_sequences_dir = fnmatch.filter(zip.namelist(), os.path.join('test_input_sequences', '*.pkl'))
            train_sub_seq_max_len_dir = fnmatch.filter(zip.namelist(), os.path.join('train_sub_seq_max_len', '*.pkl'))

            self.train_x = pickle.loads(zip.read(train_x_dir[0]))
            self.train_y = pickle.loads(zip.read(train_y_dir[0]))
            self.valid_x = pickle.loads(zip.read(valid_x_dir[0]))
            self.valid_y = pickle.loads(zip.read(valid_y_dir[0]))
            self.test_x = pickle.loads(zip.read(test_x_dir[0]))
            self.input_dict = pickle.loads(zip.read(input_dict_dir[0]))
            self.output_dict = pickle.loads(zip.read(output_dict_dir[0]))
            self.output_idxs_to_grams_correspond = DataPreparation.get_idxs_to_grams_correspond(self.output_dict)
            self.test_input_sequences = pickle.loads(zip.read(test_input_sequences_dir[0]))
            self.train_sub_seq_max_len = pickle.loads(zip.read(train_sub_seq_max_len_dir[0]))

    def _read_files(self):
        self.test_input_dataset = DataPreparation.read_file(self.test_input_file_path)

    def _split_files(self):
        self.test_input_headers, self.test_input_sequences = DataPreparation.split_to_sequences(
            self.test_input_dataset
        )

    def _load_input_from_json(self):
        self.test_input_sequences = [json.loads(sequence) for sequence in self.test_input_sequences]

    def _prepare_input(self):
        self.input_grams_to_idxs_correspond = self.get_grams_to_idxs_correspond(self.input_dict)
        self.test_input_vectorized_sequences = DataPreparation.sequences_one_hot_vectorization(
            self.test_input_sequences,
            self.input_grams_to_idxs_correspond,
            n=self.input_ngram,
            is_sub_seq=self.is_in_sub_seq
        )

    def _define_input_vectors(self):
        self.test_input_vectorized_sequences = self.test_input_sequences

    def _add_in_vec_to_x(self):
        self.test_x = [self.test_input_vectorized_sequences]

    def _add_id_data(self):
        if self.test_id_file_path is not None:
            self.test_id_dataset = DataPreparation.read_file(self.test_id_file_path)
            self.test_id_headers, self.test_id_sequences = DataPreparation.split_to_sequences(
                self.test_id_dataset
            )
            self.test_input_identification_vectors = [json.loads(sequence) for sequence in self.test_id_sequences]
        else:
            self.test_input_identification_vectors = DataPreparation.get_identification_vectors(
                self.test_input_headers
            )

    def _flatten_sub_seq(self):
        self.test_input_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.test_input_vectorized_sequences
        )

        self.test_x = [self.test_input_flatten_vectorized_sequences]

        if self.add_identification_data:
            self.test_sub_seq_lens = [len(el) for el in self.test_input_vectorized_sequences]

            self.test_input_reshaped_vectorized_sequences, \
            self.test_input_reshaped_vectorized_codes_sequences, \
            self.test_input_reshaped_vectorized_ids_sequences = DataPreparation.reshape_sequences(
                self.test_input_vectorized_sequences,
                self.test_sub_seq_lens,
                None,
                self.test_input_identification_vectors
            )

            self.test_input_flatten_reshaped_vectorized_ids_sequences = DataPreparation.flatten_sub_sequences(
                self.test_input_reshaped_vectorized_ids_sequences
            )

            self.test_x.append(self.test_input_flatten_reshaped_vectorized_ids_sequences)

    def _encode_sub_seq(self):
        self.test_sub_seq_lens = [self.train_sub_seq_max_len for _ in self.test_input_vectorized_sequences]
        self.test_sub_seq_codes = [[1 if i == j else 0 for j in range(self.train_sub_seq_max_len)] for i in range(self.train_sub_seq_max_len)]

        self.test_input_reshaped_vectorized_sequences, \
        self.test_input_reshaped_vectorized_codes_sequences, \
        self.test_input_reshaped_vectorized_ids_sequences = DataPreparation.reshape_sequences(
            self.test_input_vectorized_sequences,
            self.test_sub_seq_lens,
            self.test_sub_seq_codes,
            self.test_input_identification_vectors if self.add_identification_data else None
        )

        self.test_x = [
            self.test_input_reshaped_vectorized_sequences,
            self.test_input_reshaped_vectorized_codes_sequences
        ]

        if self.add_identification_data:
            self.test_x.append(self.test_input_reshaped_vectorized_ids_sequences)

    def _split_to_batches(self):
        self.test_x_batches = DataPreparation.test_to_batches(self.test_x)
        self.test_x = DataPreparation.test_to_nparray(self.test_x_batches)

    @staticmethod
    def read_file(file_path):
        res = ""
        file_descriptor = open(file_path, 'r', encoding='utf-8')
        while True:
            line = file_descriptor.read(100000000)
            if not line:
                break
            else:
                res += line
        file_descriptor.close()
        return res

    @staticmethod
    def split_to_sequences(dataset):
        headers = re.findall(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', dataset)
        bodies = [el.replace('\n', '') for el in re.split(r'Вопрос\s+\d+\.(?:\d+\.)*\s*', dataset)[1:]]
        return headers, bodies

    @staticmethod
    def get_dict(ngrams):
        res = sorted(list(set(ngrams)))
        return res

    @staticmethod
    def get_grams_to_idxs_correspond(dictionary):
        res = dict((g, i) for i, g in enumerate(dictionary))
        return res

    @staticmethod
    def get_idxs_to_grams_correspond(dictionary):
        res = dict((i, g) for i, g in enumerate(dictionary))
        return res

    @staticmethod
    def lead_sequences_to_same_length(sequences, sequence_length, mask_value):
        res_sequences = []
        for sequence in sequences:
            res_sequence = sequence[:]
            if mask_value is None:
                sequence_element = [''.join(['_' for _ in sequence[0]])]
            else:
                sequence_element = mask_value
            for _ in range(len(sequence), sequence_length, 1):
                res_sequence += sequence_element
            res_sequences.append(res_sequence)
        return res_sequences

    @staticmethod
    def sequences_increment(sequences):
        res_sequences = []
        for sequence in sequences:
            res_sequence = sequence[:] + [''.join(['_' for _ in sequence[0]])]
            res_sequences.append(res_sequence)
        return res_sequences

    @staticmethod
    def sequences_one_hot_vectorization(sequences, grams_to_idxs_correspond, n=1, is_sub_seq=False):
        res = []
        for sequence in sequences:
            if is_sub_seq:
                res_sub_seq = []
                for sub_seq in sequence:
                    res_sub_seq += DataPreparation.sequence_one_hot_vectorization(sub_seq, grams_to_idxs_correspond, n)
                res.append(res_sub_seq)
            else:
                res += DataPreparation.sequence_one_hot_vectorization(sequence, grams_to_idxs_correspond, n)
        return res

    @staticmethod
    def sequence_one_hot_vectorization(sequence, grams_to_idxs_correspond, n):
        res = []
        sequence_vectors = []
        if n > 0:
            for i in range(0, len(sequence) - n + 1, n):
                ngram = ''
                for j in range(n):
                    ngram += sequence[i + j]
                vector = [0 for _ in range(len(grams_to_idxs_correspond))]
                if ngram in grams_to_idxs_correspond:
                    vector[grams_to_idxs_correspond[ngram]] = 1
                sequence_vectors.append(vector)
        else:
            ngrams = sequence
            for ngram in ngrams:
                vector = [0 for _ in range(len(grams_to_idxs_correspond))]
                if ngram in grams_to_idxs_correspond:
                    vector[grams_to_idxs_correspond[ngram]] = 1
                sequence_vectors.append(vector)
        res.append(sequence_vectors)
        return res

    @staticmethod
    def get_identification_vectors(headers):
        res = []
        for header in headers:
            if re.match(r'Вопрос\s+1\.\s*', header):
                res.append([1, 0, 0, 0, 0, 0, 0])
            elif re.match(r'Вопрос\s+2\.\s*', header):
                res.append([0, 1, 0, 0, 0, 0, 0])
            elif re.match(r'Вопрос\s+3\.\s*', header):
                res.append([0, 0, 1, 0, 0, 0, 0])
            elif re.match(r'Вопрос\s+4\.\s*', header):
                res.append([0, 0, 0, 1, 0, 0, 0])
            elif re.match(r'Вопрос\s+5\.\s*', header):
                res.append([0, 0, 0, 0, 1, 0, 0])
            elif re.match(r'Вопрос\s+6\.\s*', header):
                res.append([0, 0, 0, 0, 0, 1, 0])
            elif re.match(r'Вопрос\s+7\.\s*', header):
                res.append([0, 0, 0, 0, 0, 0, 1])
            else:
                print("Undefined question type '" + header + "'!")
        return res

    @staticmethod
    def reshape_sequences(sequences, sub_seq_lens, codes=None, ids=None):
        res_seqs = []
        res_codes = []
        res_ids = []
        for i in range(len(sub_seq_lens)):
            for j in range(sub_seq_lens[i]):
                res_seqs.append(sequences[i][:])
                if codes is not None:
                    res_codes.append([codes[j][:] for _ in range(len(sequences[i]))])
                if ids is not None:
                    res_ids.append([ids[i][:] for _ in range(len(sequences[i]))])
        return res_seqs, res_codes, res_ids

    @staticmethod
    def flatten_sub_sequences(sequences):
        res = []
        for sub_seq in sequences:
            res += sub_seq
        return res

    @staticmethod
    def split_to_batches(x, y, max_batch_size):
        res_x = []
        res_y = []
        for _ in x:
            res_x.append([])
        for _ in y:
            res_y.append([])
        for i in range(len(x[0])):
            if len(res_x[0]) == 0:
                for k in range(len(x)):
                    res_x[k].append([x[k][i]])
                for k in range(len(y)):
                    res_y[k].append([y[k][i]])
            else:
                for j in range(len(res_x[0])):
                    if len(res_x[0][j][0]) == len(x[0][i]):
                        if len(res_x[0][j]) < max_batch_size:
                            for k in range(len(x)):
                                res_x[k][j].append(x[k][i])
                            for k in range(len(y)):
                                res_y[k][j].append(y[k][i])
                            break
                    if j == len(res_x[0]) - 1:
                        for k in range(len(x)):
                            res_x[k].append([x[k][i]])
                        for k in range(len(y)):
                            res_y[k].append([y[k][i]])
        return res_x, res_y

    @staticmethod
    def batches_to_nparray(x, y):
        x_res = []
        y_res = []
        for i in range(len(x)):
            x_batches = []
            for j in range(len(x[i])):
                x_batches.append(np.array(x[i][j]))
            x_res.append(x_batches)
        for i in range(len(y)):
            y_batches = []
            for j in range(len(y[i])):
                y_batches.append(np.array(y[i][j]))
            y_res.append(y_batches)
        return x_res, y_res

    @staticmethod
    def batches_to_sequences(batches):
        res_sequences = []
        for sequences in batches:
            for sequence in sequences:
                res_sequences.append(sequence)
        return res_sequences

    @staticmethod
    def decode_sequences(sequences, dictionary):
        idx_to_gram = dict((i, g) for i, g in enumerate(dictionary))
        res_sequences = []
        for sequence in sequences:
            res_sequence = []
            for vector in sequence:
                index = np.argmax(vector)
                gram = idx_to_gram[index]
                res_sequence.append(gram)
            res_sequences.append(res_sequence)
        return res_sequences

    @staticmethod
    def lead_to_same_len(sequences, length):
        res_sequences = []
        for sequence in sequences:
            res_sequence = sequence[:] + ['_' for _ in range(len(sequence), length, 1)]
            res_sequences.append(res_sequence)
        return res_sequences

    @staticmethod
    def encode_sequences(sequences, dictionary):
        gram_to_idx = dict((g, i) for i, g in enumerate(dictionary))
        vector = [0 for _ in dictionary]
        res_sequences = []
        for sequence in sequences:
            res_sequence = []
            for gram in sequence:
                index = gram_to_idx[gram]
                res_vector = vector[:]
                res_vector[index] = 1
                res_sequence.append(res_vector)
            res_sequences.append(res_sequence)
        return res_sequences

    @staticmethod
    def concatenate_data(lhs, rhs):
        res = []
        if len(lhs) == 0:
            res = rhs[:]
        else:
            for i in range(len(lhs)):
                res.append(lhs[i] + rhs[i])
        return res

    @staticmethod
    def test_to_batches(x):
        res_x = []
        for i in range(len(x)):
            res_batches = []
            for j in range(len(x[i])):
                res_batches.append([x[i][j]])
            res_x.append(res_batches)
        return res_x

    @staticmethod
    def test_to_nparray(x):
        res_x = []
        for i in range(len(x)):
            res_batches = []
            for j in range(len(x[i])):
                res_batches.append(np.array(x[i][j]))
            res_x.append(res_batches)
        return res_x