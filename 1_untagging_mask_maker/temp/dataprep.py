import itertools
import json
import re
import math
import numpy as np
import pickle
import os
import zipfile as zf
import datetime
import fnmatch

from functools import reduce


class DataPreparation:

    def __init__(self,
                 train_input_file_path,
                 train_output_file_path,
                 valid_input_file_path,
                 valid_output_file_path,
                 test_input_file_path,
                 test_output_file_path,
                 input_ngram=1,
                 output_ngram=1,
                 lead_input_sequences_to_same_size=False,
                 lead_output_sequences_to_same_size=False,
                 in_mask_value=None,
                 out_mask_value=None,
                 lead_both_seq_to_same_size=False,
                 is_input_data_prepared=False,
                 is_output_data_prepared=False,
                 is_input_data_in_json=False,
                 is_output_data_in_json=False,
                 add_sequence_element_number_data=False,
                 add_sequence_elements_number_data=False,
                 add_identification_data=False,
                 encode_output=False,
                 train_id_file_path=None,
                 valid_id_file_path=None,
                 test_id_file_path=None,
                 is_out_sub_seq=False,
                 is_in_sub_seq=False,
                 reencoding=False,
                 lead_in_sub_seq_to_same_size=False,
                 lead_out_sub_seq_to_same_size=False,
                 max_batch_size=512,
                 load_from_file=None,
                 add_object_after=False):
        self.train_input_file_path = train_input_file_path
        self.train_output_file_path = train_output_file_path
        self.valid_input_file_path = valid_input_file_path
        self.valid_output_file_path = valid_output_file_path
        self.test_input_file_path = test_input_file_path
        self.test_output_file_path = test_output_file_path
        self.input_ngram = input_ngram
        self.output_ngram = output_ngram
        self.lead_input_sequences_to_same_size = lead_input_sequences_to_same_size
        self.lead_output_sequences_to_same_size = lead_output_sequences_to_same_size
        self.lead_both_seq_to_same_size = lead_both_seq_to_same_size
        self.is_input_data_prepared = is_input_data_prepared
        self.is_output_data_prepared = is_output_data_prepared
        self.is_input_data_in_json = is_input_data_in_json
        self.is_output_data_in_json = is_output_data_in_json
        self.add_sequence_element_number_data = add_sequence_element_number_data
        self.add_sequence_elements_number_data = add_sequence_elements_number_data
        self.add_identification_data = add_identification_data
        self.encode_output = encode_output
        self.train_id_file_path = train_id_file_path
        self.valid_id_file_path = valid_id_file_path
        self.test_id_file_path = test_id_file_path
        self.is_out_sub_seq = is_out_sub_seq
        self.is_in_sub_seq = is_in_sub_seq
        self.in_mask_value = in_mask_value
        self.out_mask_value = out_mask_value
        self.reencoding = reencoding
        self.lead_out_sub_seq_to_same_size = lead_out_sub_seq_to_same_size
        self.lead_in_sub_seq_to_same_size = lead_in_sub_seq_to_same_size
        self.max_batch_size = max_batch_size
        self.train_x = []
        self.train_y = []
        self.valid_x = []
        self.valid_y = []
        self.test_x = []
        self.test_y = []
        self.input_dict = []
        self.output_dict = []
        self.input_seq_max_len = 0
        self.out_sub_seq_code_len = 0
        self.add_object_after = add_object_after
        self.load_from_file = load_from_file

        if self.load_from_file is not None:
            self._load_from_file()
            return

        self._read_files()

        self._split_files()

        if self.is_input_data_in_json:
            self._load_input_from_json()

        if self.is_output_data_in_json:
            self._load_output_from_json()

        if self.lead_input_sequences_to_same_size:
            self._lead_in_seq_to_same_size()

        if self.add_object_after:
            self._add_obj_after()

        if self.lead_output_sequences_to_same_size:
            self._lead_out_seq_to_same_size()

        if self.lead_in_sub_seq_to_same_size:
            self._lead_in_seq_to_same_size()

        if self.lead_out_sub_seq_to_same_size:
            self._lead_out_sub_seq_to_same_size()

        if self.lead_both_seq_to_same_size:
            self._lead_both_seq_to_same_size()

        if not self.is_input_data_prepared:
            self._prepare_input()
        else:
            self._define_input_vectors()

        self._add_in_vec_to_x()

        if not self.is_output_data_prepared:
            self._prepare_output()
        else:
            self._define_output_vectors()

        self._add_out_vec_to_y()

        if self.is_in_sub_seq:
            self._flatten_sub_seq()

        if self.is_out_sub_seq:
            self._encode_sub_seq()

        if self.add_sequence_element_number_data:
            self._add_seq_elem_num_data()

        if self.add_sequence_elements_number_data:
            self._add_seq_elems_num_data()

        if self.add_identification_data:
            self._add_id_data()

        if self.encode_output:
            self._encode_output()

        if self.reencoding:
            self._reencoding()

        self._split_to_batches()

    def _load_from_file(self):
        with zf.ZipFile(self.load_from_file, 'r') as zip:
            train_x_dir = fnmatch.filter(zip.namelist(), os.path.join('train_x', '*.pkl'))
            train_y_dir = fnmatch.filter(zip.namelist(), os.path.join('train_y', '*.pkl'))
            valid_x_dir = fnmatch.filter(zip.namelist(), os.path.join('valid_x', '*.pkl'))
            valid_y_dir = fnmatch.filter(zip.namelist(), os.path.join('valid_y', '*.pkl'))
            test_x_dir = fnmatch.filter(zip.namelist(), os.path.join('test_x', '*.pkl'))
            test_y_dir = fnmatch.filter(zip.namelist(), os.path.join('test_y', '*.pkl'))
            input_dict_dir = fnmatch.filter(zip.namelist(), os.path.join('input_dict', '*.pkl'))
            output_dict_dir = fnmatch.filter(zip.namelist(), os.path.join('output_dict', '*.pkl'))
            test_input_sequences_dir = fnmatch.filter(zip.namelist(), os.path.join('test_input_sequences', '*.pkl'))
            test_output_sequences_dir = fnmatch.filter(zip.namelist(), os.path.join('test_output_sequences', '*.pkl'))
            input_seq_max_len_dir = fnmatch.filter(zip.namelist(), os.path.join('input_seq_max_len', '*.pkl'))
            out_sub_seq_code_len_dir = fnmatch.filter(zip.namelist(), os.path.join('out_sub_seq_code_len', '*.pkl'))

            if len(train_x_dir) > 1:
                self.train_x = []
                self.train_y = []
                self.valid_x = []
                self.valid_y = []
                self.test_x = []
                self.test_y = []
                self.input_dict = []
                self.output_dict = []
                self.test_input_sequences = []
                self.test_output_sequences = []
                self.input_seq_max_len = 0
                self.out_sub_seq_code_len = 0

                self.input_dict = []
                for input_dict_file in input_dict_dir:
                    self.input_dict += pickle.loads(zip.read(input_dict_file))
                self.input_dict = DataPreparation.get_dict(self.input_dict)

                self.output_dict = []
                for output_dict_file in output_dict_dir:
                    self.output_dict += pickle.loads(zip.read(output_dict_file))
                self.output_dict = DataPreparation.get_dict(self.output_dict)

                self.output_idxs_to_grams_correspond = DataPreparation.get_idxs_to_grams_correspond(self.output_dict)

                for input_seq_max_len_file in input_seq_max_len_dir:
                    input_seq_max_len = pickle.loads(zip.read(input_seq_max_len_file))
                    if self.input_seq_max_len < input_seq_max_len:
                        self.input_seq_max_len = input_seq_max_len

                for i in range(len(train_x_dir)):
                    train_x = pickle.loads(zip.read(train_x_dir[i]))
                    train_y = pickle.loads(zip.read(train_y_dir[i]))
                    valid_x = pickle.loads(zip.read(valid_x_dir[i]))
                    valid_y = pickle.loads(zip.read(valid_y_dir[i]))
                    test_x = pickle.loads(zip.read(test_x_dir[i]))
                    test_y = pickle.loads(zip.read(test_y_dir[i]))
                    input_dict = pickle.loads(zip.read(input_dict_dir[i]))
                    output_dict = pickle.loads(zip.read(output_dict_dir[i]))
                    test_input_sequences = pickle.loads(zip.read(test_input_sequences_dir[i]))
                    test_output_sequences = pickle.loads(zip.read(test_output_sequences_dir[i]))
                    out_sub_seq_code_len = pickle.loads(zip.read(out_sub_seq_code_len_dir[i]))

                    self.max_batch_size = len(train_x[0][0])

                    train_x_sequences = []
                    for element in train_x:
                        train_x_sequences.append(DataPreparation.batches_to_sequences(element))
                    train_x_sequences[0] = DataPreparation.decode_sequences(train_x_sequences[0], input_dict)
                    train_x_sequences[0] = DataPreparation.lead_to_same_len(train_x_sequences[0],
                                                                            self.input_seq_max_len)
                    train_x_sequences[0] = DataPreparation.encode_sequences(train_x_sequences[0], self.input_dict)
                    self.train_x = DataPreparation.concatenate_data(self.train_x, train_x_sequences)

                    train_y_sequences = []
                    for element in train_y:
                        train_y_sequences.append(DataPreparation.batches_to_sequences(element))
                    train_y_sequences[0] = DataPreparation.decode_sequences(train_y_sequences[0], output_dict)
                    train_y_sequences[0] = DataPreparation.lead_to_same_len(train_y_sequences[0],
                                                                            self.input_seq_max_len)
                    train_y_sequences[0] = DataPreparation.encode_sequences(train_y_sequences[0], self.output_dict)
                    self.train_y = DataPreparation.concatenate_data(self.train_y, train_y_sequences)

                    valid_x_sequences = []
                    for element in valid_x:
                        valid_x_sequences.append(DataPreparation.batches_to_sequences(element))
                    valid_x_sequences[0] = DataPreparation.decode_sequences(valid_x_sequences[0], input_dict)
                    valid_x_sequences[0] = DataPreparation.lead_to_same_len(valid_x_sequences[0],
                                                                            self.input_seq_max_len)
                    valid_x_sequences[0] = DataPreparation.encode_sequences(valid_x_sequences[0], self.input_dict)
                    self.valid_x = DataPreparation.concatenate_data(self.valid_x, valid_x_sequences)

                    valid_y_sequences = []
                    for element in valid_y:
                        valid_y_sequences.append(DataPreparation.batches_to_sequences(element))
                    valid_y_sequences[0] = DataPreparation.decode_sequences(valid_y_sequences[0], output_dict)
                    valid_y_sequences[0] = DataPreparation.lead_to_same_len(valid_y_sequences[0],
                                                                            self.input_seq_max_len)
                    valid_y_sequences[0] = DataPreparation.encode_sequences(valid_y_sequences[0], self.output_dict)
                    self.valid_y = DataPreparation.concatenate_data(self.valid_y, valid_y_sequences)

                    test_x_sequences = []
                    for element in test_x:
                        test_x_sequences.append(DataPreparation.batches_to_sequences(element))
                    test_x_sequences[0] = DataPreparation.decode_sequences(test_x_sequences[0], input_dict)
                    test_x_sequences[0] = DataPreparation.lead_to_same_len(test_x_sequences[0], self.input_seq_max_len)
                    test_x_sequences[0] = DataPreparation.encode_sequences(test_x_sequences[0], self.input_dict)
                    self.test_x = DataPreparation.concatenate_data(self.test_x, test_x_sequences)

                    test_y_sequences = []
                    for element in test_y:
                        test_y_sequences.append(DataPreparation.batches_to_sequences(element))
                    test_y_sequences[0] = DataPreparation.decode_sequences(test_y_sequences[0], output_dict)
                    test_y_sequences[0] = DataPreparation.lead_to_same_len(test_y_sequences[0], self.input_seq_max_len)
                    test_y_sequences[0] = DataPreparation.encode_sequences(test_y_sequences[0], self.output_dict)
                    self.test_y = DataPreparation.concatenate_data(self.test_y, test_y_sequences)

                    self.test_input_sequences += test_input_sequences
                    self.test_output_sequences += test_output_sequences
                    if self.out_sub_seq_code_len < out_sub_seq_code_len:
                        self.out_sub_seq_code_len = out_sub_seq_code_len

                self.train_x_batches, self.train_y_batches = DataPreparation.split_to_batches(self.train_x,
                                                                                              self.train_y,
                                                                                              self.max_batch_size)
                self.valid_x_batches, self.valid_y_batches = DataPreparation.split_to_batches(self.valid_x,
                                                                                              self.valid_y,
                                                                                              self.max_batch_size)
                self.test_x_batches, self.test_y_batches = DataPreparation.split_to_batches(self.test_x, self.test_y, 1)
                self.train_x, self.train_y = DataPreparation.batches_to_nparray(self.train_x_batches,
                                                                                self.train_y_batches)
                self.valid_x, self.valid_y = DataPreparation.batches_to_nparray(self.valid_x_batches,
                                                                                self.valid_y_batches)
                self.test_x, self.test_y = DataPreparation.batches_to_nparray(self.test_x_batches, self.test_y_batches)
            else:
                self.train_x = pickle.loads(zip.read(train_x_dir[0]))
                self.train_y = pickle.loads(zip.read(train_y_dir[0]))
                self.valid_x = pickle.loads(zip.read(valid_x_dir[0]))
                self.valid_y = pickle.loads(zip.read(valid_y_dir[0]))
                self.test_x = pickle.loads(zip.read(test_x_dir[0]))
                self.test_y = pickle.loads(zip.read(test_y_dir[0]))
                self.input_dict = pickle.loads(zip.read(input_dict_dir[0]))
                self.output_dict = pickle.loads(zip.read(output_dict_dir[0]))
                self.output_idxs_to_grams_correspond = DataPreparation.get_idxs_to_grams_correspond(self.output_dict)
                self.test_input_sequences = pickle.loads(zip.read(test_input_sequences_dir[0]))
                self.test_output_sequences = pickle.loads(zip.read(test_output_sequences_dir[0]))
                self.input_seq_max_len = pickle.loads(zip.read(input_seq_max_len_dir[0]))
                self.out_sub_seq_code_len = pickle.loads(zip.read(out_sub_seq_code_len_dir[0]))

    def _read_files(self):
        self.train_input_dataset = DataPreparation.read_file(self.train_input_file_path)
        self.train_output_dataset = DataPreparation.read_file(self.train_output_file_path)
        self.valid_input_dataset = DataPreparation.read_file(self.valid_input_file_path)
        self.valid_output_dataset = DataPreparation.read_file(self.valid_output_file_path)
        self.test_input_dataset = DataPreparation.read_file(self.test_input_file_path)
        self.test_output_dataset = DataPreparation.read_file(self.test_output_file_path)

    def _split_files(self):
        self.train_input_headers, self.train_input_sequences = DataPreparation.split_to_sequences(
            self.train_input_dataset
        )
        self.train_output_headers, self.train_output_sequences = DataPreparation.split_to_sequences(
            self.train_output_dataset
        )
        self.valid_input_headers, self.valid_input_sequences = DataPreparation.split_to_sequences(
            self.valid_input_dataset
        )
        self.valid_output_headers, self.valid_output_sequences = DataPreparation.split_to_sequences(
            self.valid_output_dataset
        )
        self.test_input_headers, self.test_input_sequences = DataPreparation.split_to_sequences(
            self.test_input_dataset
        )
        self.test_output_headers, self.test_output_sequences = DataPreparation.split_to_sequences(
            self.test_output_dataset
        )

    def _load_input_from_json(self):
        self.train_input_sequences = [json.loads(sequence) for sequence in self.train_input_sequences]
        self.valid_input_sequences = [json.loads(sequence) for sequence in self.valid_input_sequences]
        self.test_input_sequences = [json.loads(sequence) for sequence in self.test_input_sequences]

    def _load_output_from_json(self):
        self.train_output_sequences = [json.loads(sequence) for sequence in self.train_output_sequences]
        self.valid_output_sequences = [json.loads(sequence) for sequence in self.valid_output_sequences]
        self.test_output_sequences = [json.loads(sequence) for sequence in self.test_output_sequences]

    def _lead_in_seq_to_same_size(self):
        self.train_input_seq_max_len = DataPreparation.get_sequence_max_length(self.train_input_sequences)
        self.valid_input_seq_max_len = DataPreparation.get_sequence_max_length(self.valid_input_sequences)
        self.test_input_seq_max_len = DataPreparation.get_sequence_max_length(self.test_input_sequences)
        self.input_seq_max_len = max(
            self.train_input_seq_max_len,
            self.valid_input_seq_max_len,
            self.test_input_seq_max_len
        )
        self.train_input_sequences = DataPreparation.lead_sequences_to_same_length(
            self.train_input_sequences,
            self.input_seq_max_len,
            self.in_mask_value
        )
        self.valid_input_sequences = DataPreparation.lead_sequences_to_same_length(
            self.valid_input_sequences,
            self.input_seq_max_len,
            self.in_mask_value
        )
        self.test_input_sequences = DataPreparation.lead_sequences_to_same_length(
            self.test_input_sequences,
            self.input_seq_max_len,
            self.in_mask_value
        )

    def _add_obj_after(self):
        self.train_output_seq_max_len = DataPreparation.get_sequence_max_length(self.train_output_sequences)
        self.valid_output_seq_max_len = DataPreparation.get_sequence_max_length(self.valid_output_sequences)
        self.test_output_seq_max_len = DataPreparation.get_sequence_max_length(self.test_output_sequences)
        self.output_seq_max_len = max(
            self.train_output_seq_max_len,
            self.valid_output_seq_max_len,
            self.test_output_seq_max_len
        ) + 1
        self.train_output_sequences = DataPreparation.sequences_increment(self.train_output_sequences)
        self.valid_output_sequences = DataPreparation.sequences_increment(self.valid_output_sequences)
        self.test_output_sequences = DataPreparation.sequences_increment(self.test_output_sequences)

    def _lead_out_seq_to_same_size(self):
        self.train_output_sequences = DataPreparation.lead_sequences_to_same_length(
            self.train_output_sequences,
            self.output_seq_max_len,
            self.out_mask_value
        )
        self.valid_output_sequences = DataPreparation.lead_sequences_to_same_length(
            self.valid_output_sequences,
            self.output_seq_max_len,
            self.out_mask_value
        )
        self.test_output_sequences = DataPreparation.lead_sequences_to_same_length(
            self.test_output_sequences,
            self.output_seq_max_len,
            self.out_mask_value
        )

    def _lead_in_sub_seq_to_same_size(self):
        self.train_in_sub_seq_max_len = DataPreparation.get_sub_seq_max_length(self.train_input_sequences)
        self.valid_in_sub_seq_max_len = DataPreparation.get_sub_seq_max_length(self.valid_input_sequences)
        self.test_in_sub_seq_max_len = DataPreparation.get_sub_seq_max_length(self.test_input_sequences)
        self.in_sub_seq_max_len = max(
            self.train_in_sub_seq_max_len,
            self.valid_in_sub_seq_max_len,
            self.test_in_sub_seq_max_len
        )
        self.train_input_sequences = DataPreparation.lead_sub_seq_to_same_length(
            self.train_input_sequences,
            self.in_sub_seq_max_len,
            "_"
        )
        self.valid_input_sequences = DataPreparation.lead_sub_seq_to_same_length(
            self.valid_input_sequences,
            self.in_sub_seq_max_len,
            "_"
        )
        self.test_input_sequences = DataPreparation.lead_sub_seq_to_same_length(
            self.test_input_sequences,
            self.in_sub_seq_max_len,
            "_"
        )

    def _lead_out_sub_seq_to_same_size(self):
        self.train_out_sub_seq_max_len = DataPreparation.get_sub_seq_max_length(self.train_output_sequences)
        self.valid_out_sub_seq_max_len = DataPreparation.get_sub_seq_max_length(self.valid_output_sequences)
        self.test_out_sub_seq_max_len = DataPreparation.get_sub_seq_max_length(self.test_output_sequences)
        self.out_sub_seq_max_len = max(
            self.train_out_sub_seq_max_len,
            self.valid_out_sub_seq_max_len,
            self.test_out_sub_seq_max_len
        )
        self.train_output_sequences = DataPreparation.lead_sub_seq_to_same_length(
            self.train_output_sequences,
            self.out_sub_seq_max_len,
            "_"
        )
        self.valid_output_sequences = DataPreparation.lead_sub_seq_to_same_length(
            self.valid_output_sequences,
            self.out_sub_seq_max_len,
            "_"
        )
        self.test_output_sequences = DataPreparation.lead_sub_seq_to_same_length(
            self.test_output_sequences,
            self.out_sub_seq_max_len,
            "_"
        )

    def _lead_both_seq_to_same_size(self):
        self.seq_max_len = max(
            self.input_seq_max_len,
            self.output_seq_max_len
        )
        self.train_input_sequences = DataPreparation.lead_sequences_to_same_length(
            self.train_input_sequences,
            self.seq_max_len,
            self.in_mask_value
        )
        self.valid_input_sequences = DataPreparation.lead_sequences_to_same_length(
            self.valid_input_sequences,
            self.seq_max_len,
            self.in_mask_value
        )
        self.test_input_sequences = DataPreparation.lead_sequences_to_same_length(
            self.test_input_sequences,
            self.seq_max_len,
            self.in_mask_value
        )
        self.train_output_sequences = DataPreparation.lead_sequences_to_same_length(
            self.train_output_sequences,
            self.seq_max_len,
            self.out_mask_value
        )
        self.valid_output_sequences = DataPreparation.lead_sequences_to_same_length(
            self.valid_output_sequences,
            self.seq_max_len,
            self.out_mask_value
        )
        self.test_output_sequences = DataPreparation.lead_sequences_to_same_length(
            self.test_output_sequences,
            self.seq_max_len,
            self.out_mask_value
        )

    def _prepare_input(self):
        self.train_input_ngrams = self.get_ngrams(
            self.train_input_sequences,
            n=self.input_ngram,
            is_sub_seq=self.is_in_sub_seq
        )
        self.valid_input_ngrams = self.get_ngrams(
            self.valid_input_sequences,
            n=self.input_ngram,
            is_sub_seq=self.is_in_sub_seq
        )
        self.test_input_ngrams = self.get_ngrams(
            self.test_input_sequences,
            n=self.input_ngram,
            is_sub_seq=self.is_in_sub_seq
        )
        self.input_ngrams = self.train_input_ngrams + self.valid_input_ngrams
        self.input_dict = DataPreparation.get_dict(self.input_ngrams)
        self.input_grams_to_idxs_correspond = self.get_grams_to_idxs_correspond(self.input_dict)
        self.input_idxs_to_grams_correspond = self.get_idxs_to_grams_correspond(self.input_dict)
        self.train_input_vectorized_sequences = DataPreparation.sequences_one_hot_vectorization(
            self.train_input_sequences,
            self.input_grams_to_idxs_correspond,
            n=self.input_ngram,
            is_sub_seq=self.is_in_sub_seq
        )
        self.valid_input_vectorized_sequences = DataPreparation.sequences_one_hot_vectorization(
            self.valid_input_sequences,
            self.input_grams_to_idxs_correspond,
            n=self.input_ngram,
            is_sub_seq=self.is_in_sub_seq
        )
        self.test_input_vectorized_sequences = DataPreparation.sequences_one_hot_vectorization(
            self.test_input_sequences,
            self.input_grams_to_idxs_correspond,
            n=self.input_ngram,
            is_sub_seq=self.is_in_sub_seq
        )

    def _define_input_vectors(self):
        self.train_input_vectorized_sequences = self.train_input_sequences
        self.valid_input_vectorized_sequences = self.valid_input_sequences
        self.test_input_vectorized_sequences = self.test_input_sequences

    def _add_in_vec_to_x(self):
        self.train_x.append(self.train_input_vectorized_sequences)
        self.valid_x.append(self.valid_input_vectorized_sequences)
        self.test_x.append(self.test_input_vectorized_sequences)

    def _prepare_output(self):
        self.train_output_ngrams = self.get_ngrams(
            self.train_output_sequences,
            n=self.output_ngram,
            is_sub_seq=self.is_out_sub_seq
        )
        self.valid_output_ngrams = self.get_ngrams(
            self.valid_output_sequences,
            n=self.output_ngram,
            is_sub_seq=self.is_out_sub_seq
        )
        self.test_output_ngrams = self.get_ngrams(
            self.test_output_sequences,
            n=self.output_ngram,
            is_sub_seq=self.is_out_sub_seq
        )
        self.output_ngrams = self.train_output_ngrams + self.valid_output_ngrams
        self.output_dict = DataPreparation.get_dict(self.output_ngrams)
        self.output_grams_to_idxs_correspond = self.get_grams_to_idxs_correspond(self.output_dict)
        self.output_idxs_to_grams_correspond = self.get_idxs_to_grams_correspond(self.output_dict)
        self.train_output_vectorized_sequences = DataPreparation.sequences_one_hot_vectorization(
            self.train_output_sequences,
            self.output_grams_to_idxs_correspond,
            n=self.output_ngram,
            is_sub_seq=self.is_out_sub_seq
        )
        self.valid_output_vectorized_sequences = DataPreparation.sequences_one_hot_vectorization(
            self.valid_output_sequences,
            self.output_grams_to_idxs_correspond,
            n=self.output_ngram,
            is_sub_seq=self.is_out_sub_seq
        )
        self.test_output_vectorized_sequences = DataPreparation.sequences_one_hot_vectorization(
            self.test_output_sequences,
            self.output_grams_to_idxs_correspond,
            n=self.output_ngram,
            is_sub_seq=self.is_out_sub_seq
        )

    def _define_output_vectors(self):
        self.train_output_vectorized_sequences = self.train_output_sequences
        self.valid_output_vectorized_sequences = self.valid_output_sequences
        self.test_output_vectorized_sequences = self.test_output_sequences

    def _add_out_vec_to_y(self):
        self.train_y.append(self.train_output_vectorized_sequences)
        self.valid_y.append(self.valid_output_vectorized_sequences)
        self.test_y.append(self.test_output_vectorized_sequences)

    def _flatten_sub_seq(self):
        self.train_input_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.train_input_vectorized_sequences
        )
        self.valid_input_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.valid_input_vectorized_sequences
        )
        self.test_input_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.test_input_vectorized_sequences
        )
        self.train_output_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.train_output_vectorized_sequences
        )
        self.valid_output_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.valid_output_vectorized_sequences
        )
        self.test_output_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.test_output_vectorized_sequences
        )
        self.train_x = [self.train_input_flatten_vectorized_sequences]
        self.valid_x = [self.valid_input_flatten_vectorized_sequences]
        self.test_x = [self.test_input_flatten_vectorized_sequences]
        self.train_y = [self.train_output_flatten_vectorized_sequences]
        self.valid_y = [self.valid_output_flatten_vectorized_sequences]
        self.test_y = [self.test_output_flatten_vectorized_sequences]

    def _encode_sub_seq(self):
        self.out_sub_seq_code_len = self.output_seq_max_len
        self.out_sub_seq_codes = [[1 if i == j else 0 for j in range(self.out_sub_seq_code_len)] for i in
                                  range(self.out_sub_seq_code_len)]
        self.train_input_reshaped_sequences = DataPreparation.reshape_sequences(
            self.train_input_sequences,
            self.train_output_vectorized_sequences
        )
        self.valid_input_reshaped_sequences = DataPreparation.reshape_sequences(
            self.valid_input_sequences,
            self.valid_output_vectorized_sequences
        )
        self.test_input_reshaped_sequences = DataPreparation.reshape_sequences(
            self.test_input_sequences,
            self.test_output_vectorized_sequences
        )
        self.train_input_flatten_reshaped_sequences = DataPreparation.flatten_sub_sequences(
            self.train_input_reshaped_sequences
        )
        self.valid_input_flatten_reshaped_sequences = DataPreparation.flatten_sub_sequences(
            self.valid_input_reshaped_sequences
        )
        self.test_input_flatten_reshaped_sequences = DataPreparation.flatten_sub_sequences(
            self.test_input_reshaped_sequences
        )
        self.train_input_reshaped_vectorized_sequences, self.train_input_codes_sequences = DataPreparation.reshape_sequences(
            self.train_input_vectorized_sequences,
            self.train_output_vectorized_sequences,
            self.out_sub_seq_codes
        )
        self.valid_input_reshaped_vectorized_sequences, self.valid_input_codes_sequences = DataPreparation.reshape_sequences(
            self.valid_input_vectorized_sequences,
            self.valid_output_vectorized_sequences,
            self.out_sub_seq_codes
        )
        self.test_input_reshaped_vectorized_sequences, self.test_input_codes_sequences = DataPreparation.reshape_sequences(
            self.test_input_vectorized_sequences,
            self.test_output_vectorized_sequences,
            self.out_sub_seq_codes
        )
        self.train_input_flatten_reshaped_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.train_input_reshaped_vectorized_sequences
        )
        self.valid_input_flatten_reshaped_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.valid_input_reshaped_vectorized_sequences
        )
        self.test_input_flatten_reshaped_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.test_input_reshaped_vectorized_sequences
        )
        self.train_output_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.train_output_vectorized_sequences
        )
        self.valid_output_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.valid_output_vectorized_sequences
        )
        self.test_output_flatten_vectorized_sequences = DataPreparation.flatten_sub_sequences(
            self.test_output_vectorized_sequences
        )
        self.train_input_flatten_codes_sequences = DataPreparation.flatten_sub_sequences(
            self.train_input_codes_sequences
        )
        self.valid_input_flatten_codes_sequences = DataPreparation.flatten_sub_sequences(
            self.valid_input_codes_sequences
        )
        self.test_input_flatten_codes_sequences = DataPreparation.flatten_sub_sequences(
            self.test_input_codes_sequences
        )
        self.train_x = [self.train_input_flatten_reshaped_vectorized_sequences,
                        self.train_input_flatten_codes_sequences]
        self.valid_x = [self.valid_input_flatten_reshaped_vectorized_sequences,
                        self.valid_input_flatten_codes_sequences]
        self.test_x = [self.test_input_flatten_reshaped_vectorized_sequences, self.test_input_flatten_codes_sequences]
        self.train_y = [self.train_output_flatten_vectorized_sequences]
        self.valid_y = [self.valid_output_flatten_vectorized_sequences]
        self.test_y = [self.test_output_flatten_vectorized_sequences]

    def _add_seq_elem_num_data(self):
        self.train_input_sequences_elements_number = DataPreparation.count_elements(
            self.input_dict,
            self.train_input_sequences,
            n=self.input_ngram
        )
        self.valid_input_sequences_elements_number = DataPreparation.count_elements(
            self.input_dict,
            self.valid_input_sequences,
            n=self.input_ngram
        )
        self.test_input_sequences_elements_number = DataPreparation.count_elements(
            self.input_dict,
            self.test_input_sequences,
            n=self.input_ngram
        )
        self.max_number = 2 * DataPreparation.get_max_number(
            self.train_input_sequences_elements_number +
            self.valid_input_sequences_elements_number
        )
        self.train_input_vectorized_sequences_elements_number_sequences = \
            DataPreparation.number_sequences_vectorization(
                self.train_input_sequences,
                self.input_grams_to_idxs_correspond,
                self.train_input_sequences_elements_number,
                self.max_number,
                n=self.input_ngram
            )
        self.valid_input_vectorized_sequences_elements_number_sequences = \
            DataPreparation.number_sequences_vectorization(
                self.valid_input_sequences,
                self.input_grams_to_idxs_correspond,
                self.valid_input_sequences_elements_number,
                self.max_number,
                n=self.input_ngram
            )
        self.test_input_vectorized_sequences_elements_number_sequences = \
            DataPreparation.number_sequences_vectorization(
                self.test_input_sequences,
                self.input_grams_to_idxs_correspond,
                self.test_input_sequences_elements_number,
                self.max_number,
                n=self.input_ngram
            )
        self.train_input_vectorized_sequences = DataPreparation.concatenate(
            self.train_input_vectorized_sequences,
            self.train_input_vectorized_sequences_elements_number_sequences
        )
        self.valid_input_vectorized_sequences = DataPreparation.concatenate(
            self.valid_input_vectorized_sequences,
            self.valid_input_vectorized_sequences_elements_number_sequences
        )
        self.test_input_vectorized_sequences = DataPreparation.concatenate(
            self.test_input_vectorized_sequences,
            self.test_input_vectorized_sequences_elements_number_sequences
        )

    def _add_seq_elems_num_data(self):
        self.train_input_sequences_elements_number = DataPreparation.count_elements(
            self.input_dict,
            self.train_input_sequences,
            n=self.input_ngram
        )
        self.valid_input_sequences_elements_number = DataPreparation.count_elements(
            self.input_dict,
            self.valid_input_sequences,
            n=self.input_ngram
        )
        self.test_input_sequences_elements_number = DataPreparation.count_elements(
            self.input_dict,
            self.test_input_sequences,
            n=self.input_ngram
        )
        self.max_number = 2 * DataPreparation.get_max_number(
            self.train_input_sequences_elements_number +
            self.valid_input_sequences_elements_number
        )
        self.normalize_train_input_sequences_elements_number = DataPreparation.normalize(
            self.train_input_sequences_elements_number,
            self.max_number
        )
        self.normalize_valid_input_sequences_elements_number = DataPreparation.normalize(
            self.valid_input_sequences_elements_number,
            self.max_number
        )
        self.normalize_test_input_sequences_elements_number = DataPreparation.normalize(
            self.test_input_sequences_elements_number,
            self.max_number
        )
        self.train_input_vectorized_sequences_elements_number_sequences = \
            DataPreparation.create_sequences_from_vectors(
                self.normalize_train_input_sequences_elements_number,
                self.train_input_vectorized_sequences
            )
        self.valid_input_vectorized_sequences_elements_number_sequences = \
            DataPreparation.create_sequences_from_vectors(
                self.normalize_valid_input_sequences_elements_number,
                self.valid_input_vectorized_sequences
            )
        self.test_input_vectorized_sequences_elements_number_sequences = \
            DataPreparation.create_sequences_from_vectors(
                self.normalize_test_input_sequences_elements_number,
                self.test_input_vectorized_sequences
            )
        self.train_input_vectorized_sequences = DataPreparation.concatenate(
            self.train_input_vectorized_sequences,
            self.train_input_vectorized_sequences_elements_number_sequences
        )
        self.valid_input_vectorized_sequences = DataPreparation.concatenate(
            self.valid_input_vectorized_sequences,
            self.valid_input_vectorized_sequences_elements_number_sequences
        )
        self.test_input_vectorized_sequences = DataPreparation.concatenate(
            self.test_input_vectorized_sequences,
            self.test_input_vectorized_sequences_elements_number_sequences
        )

    def _add_id_data(self):
        if self.train_id_file_path is not None:
            self.train_id_dataset = DataPreparation.read_file(self.train_id_file_path)
            self.train_id_headers, self.train_id_sequences = DataPreparation.split_to_sequences(
                self.train_id_dataset
            )
            self.train_input_identification_vectors = [json.loads(sequence) for sequence in self.train_id_sequences]
        else:
            self.train_input_identification_vectors = DataPreparation.get_identification_vectors(
                self.train_input_headers
            )
        if self.valid_id_file_path is not None:
            self.valid_id_dataset = DataPreparation.read_file(self.valid_id_file_path)
            self.valid_id_headers, self.valid_id_sequences = DataPreparation.split_to_sequences(
                self.valid_id_dataset
            )
            self.valid_input_identification_vectors = [json.loads(sequence) for sequence in self.valid_id_sequences]
        else:
            self.valid_input_identification_vectors = DataPreparation.get_identification_vectors(
                self.valid_input_headers
            )
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
        self.train_input_reshaped_identification_vectors = DataPreparation.reshape_sequences(
            self.train_input_identification_vectors,
            self.train_output_vectorized_sequences if self.is_out_sub_seq else self.train_input_vectorized_sequences
        )
        self.valid_input_reshaped_identification_vectors = DataPreparation.reshape_sequences(
            self.valid_input_identification_vectors,
            self.valid_output_vectorized_sequences if self.is_out_sub_seq else self.valid_input_vectorized_sequences
        )
        self.test_input_reshaped_identification_vectors = DataPreparation.reshape_sequences(
            self.test_input_identification_vectors,
            self.test_output_vectorized_sequences if self.is_out_sub_seq else self.test_input_vectorized_sequences
        )
        self.train_input_flatten_reshaped_identification_vectors = DataPreparation.flatten_sub_sequences(
            self.train_input_reshaped_identification_vectors
        )
        self.valid_input_flatten_reshaped_identification_vectors = DataPreparation.flatten_sub_sequences(
            self.valid_input_reshaped_identification_vectors
        )
        self.test_input_flatten_reshaped_identification_vectors = DataPreparation.flatten_sub_sequences(
            self.test_input_reshaped_identification_vectors
        )
        self.train_x.append(self.train_input_flatten_reshaped_identification_vectors)
        self.valid_x.append(self.valid_input_flatten_reshaped_identification_vectors)
        self.test_x.append(self.test_input_flatten_reshaped_identification_vectors)

    def _encode_output(self):
        self.max_group_number = DataPreparation.get_max_group_number(
            self.train_output_sequences +
            self.valid_output_sequences +
            self.test_output_sequences
        )
        self.group_vector_length = math.ceil(math.log2(self.max_group_number))
        self.max_groups_number = DataPreparation.get_max_groups_number(
            self.train_output_sequences +
            self.valid_output_sequences +
            self.test_output_sequences
        )
        self.groups_vector_length = self.group_vector_length * self.max_groups_number + 3
        self.groups_codes = [[int(element) for element in tuple] for tuple in
                             itertools.product('01', repeat=self.group_vector_length)]
        self.train_output_identification_vectors = DataPreparation.get_identification_vectors(
            self.train_output_headers
        )
        self.valid_output_identification_vectors = DataPreparation.get_identification_vectors(
            self.valid_output_headers
        )
        self.test_output_identification_vectors = DataPreparation.get_identification_vectors(
            self.test_output_headers
        )
        self.train_output_vectorized_sequences = self.encoding(
            self.train_output_sequences,
            self.train_output_identification_vectors
        )
        self.valid_output_vectorized_sequences = self.encoding(
            self.valid_output_sequences,
            self.valid_output_identification_vectors
        )
        self.test_output_vectorized_sequences = self.encoding(
            self.test_output_sequences,
            self.test_output_identification_vectors
        )
        self.train_output_vectorized_sequences = DataPreparation.masking(
            self.train_output_vectorized_sequences,
            self.groups_vector_length
        )
        self.valid_output_vectorized_sequences = DataPreparation.masking(
            self.valid_output_vectorized_sequences,
            self.groups_vector_length
        )
        self.test_output_vectorized_sequences = DataPreparation.masking(
            self.test_output_vectorized_sequences,
            self.groups_vector_length
        )

    def _reencoding(self):
        self.reencoding_dict = DataPreparation.get_reencoding_dict(
            self.train_input_vectorized_sequences +
            self.valid_input_vectorized_sequences +
            self.test_input_vectorized_sequences
        )
        self.reencoding_codes = DataPreparation.get_reencoding_codes(self.reencoding_dict)
        self.reencoding_correspond = DataPreparation.get_reencoding_correspond(
            self.reencoding_dict,
            self.reencoding_codes
        )
        self.train_input_vectorized_sequences = DataPreparation.reencoding(
            self.train_input_vectorized_sequences,
            self.reencoding_correspond
        )
        self.valid_input_vectorized_sequences = DataPreparation.reencoding(
            self.valid_input_vectorized_sequences,
            self.reencoding_correspond
        )
        self.test_input_vectorized_sequences = DataPreparation.reencoding(
            self.test_input_vectorized_sequences,
            self.reencoding_correspond
        )

    def _split_to_batches(self):
        self.train_x_batches, self.train_y_batches = DataPreparation.split_to_batches(
            self.train_x,
            self.train_y,
            self.max_batch_size
        )
        self.valid_x_batches, self.valid_y_batches = DataPreparation.split_to_batches(
            self.valid_x,
            self.valid_y,
            self.max_batch_size
        )
        self.test_x_batches, self.test_y_batches = DataPreparation.split_to_batches(self.test_x, self.test_y, 1)
        self.train_x, self.train_y = DataPreparation.batches_to_nparray(self.train_x_batches, self.train_y_batches)
        self.valid_x, self.valid_y = DataPreparation.batches_to_nparray(self.valid_x_batches, self.valid_y_batches)
        self.test_x, self.test_y = DataPreparation.batches_to_nparray(self.test_x_batches, self.test_y_batches)

    def save(self, file_path='data'):
        with zf.ZipFile(file_path + '.zip', 'w') as zip:
            id = re.sub(r'[\s:.-]', '', str(datetime.datetime.now())) + '.pkl'
            zip.writestr(os.path.join('train_x', id), pickle.dumps(self.train_x, protocol=0))
            zip.writestr(os.path.join('train_y', id), pickle.dumps(self.train_y, protocol=0))
            zip.writestr(os.path.join('valid_x', id), pickle.dumps(self.valid_x, protocol=0))
            zip.writestr(os.path.join('valid_y', id), pickle.dumps(self.valid_y, protocol=0))
            zip.writestr(os.path.join('test_x', id), pickle.dumps(self.test_x, protocol=0))
            zip.writestr(os.path.join('test_y', id), pickle.dumps(self.test_y, protocol=0))
            zip.writestr(os.path.join('input_dict', id), pickle.dumps(self.input_dict, protocol=0))
            zip.writestr(os.path.join('output_dict', id), pickle.dumps(self.output_dict, protocol=0))
            zip.writestr(os.path.join('test_input_sequences', id), pickle.dumps(self.test_input_sequences, protocol=0))
            zip.writestr(os.path.join('test_output_sequences', id), pickle.dumps(self.test_output_sequences, protocol=0))
            zip.writestr(os.path.join('input_seq_max_len', id), pickle.dumps(self.input_seq_max_len, protocol=0))
            zip.writestr(os.path.join('out_sub_seq_code_len', id), pickle.dumps(self.out_sub_seq_code_len, protocol=0))

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
    def get_ngrams(sequences, n=1, is_sub_seq=False):
        res = []
        for sequence in sequences:
            if is_sub_seq:
                for sub_sequence in sequence:
                    res += DataPreparation.get_ngrams_from_seq(sub_sequence, n)
            else:
                res += DataPreparation.get_ngrams_from_seq(sequence, n)
        return res

    @staticmethod
    def get_ngrams_from_seq(sequence, n):
        res = []
        if n > 0:
            for i in range(0, len(sequence) - n + 1, n):
                ngram = ''
                for j in range(n):
                    ngram += sequence[i + j]
                res.append(ngram)
        else:
            res += sequence
        return res

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
    def get_sequence_max_length(sequences):
        res = 0
        for sequence in sequences:
            if len(sequence) > res:
                res = len(sequence)
        return res

    @staticmethod
    def get_sub_seq_max_length(sequences):
        res = 0
        for sequence in sequences:
            for sub_sequence in sequence:
                if len(sub_sequence) > res:
                    res = len(sub_sequence)
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
    def lead_sub_seq_to_same_length(sequences, sequence_length, mask_value):
        res_sequences = []
        for sequence in sequences:
            res_sub_sequences = []
            for sub_sequences in sequence:
                res_sub_sequence = sub_sequences[:]
                if mask_value is None:
                    sequence_element = [''.join(['_' for _ in sub_sequences[0]])]
                else:
                    sequence_element = mask_value
                for _ in range(len(sub_sequences), sequence_length, 1):
                    res_sub_sequence += sequence_element
                res_sub_sequences.append(res_sub_sequence)
            res_sequences.append(res_sub_sequences)
        return res_sequences

    """
    def sequences_indexation(self, sequences, grams_to_idxs_correspond, n=1):
        res = []
        for sequence in sequences:
            sequence_indexes = []
            if n > 0:
                for i in range(0, len(sequence) - n + 1, n):
                    ngram = ''
                    for j in range(n):
                        ngram += sequence[i + j]
                    if ngram in grams_to_idxs_correspond:
                        index = grams_to_idxs_correspond[ngram]
                        sequence_indexes.append(index)
            else:
                ngrams = sequence.split()
                for ngram in ngrams:
                    if ngram in grams_to_idxs_correspond:
                        index = grams_to_idxs_correspond[ngram]
                        sequence_indexes.append(index)
            res.append(np.array([sequence_indexes]))
        return res
    """

    """
    def sequences_word2vec_vectorization(self, sequences, ngrams):
        model = gensim.models.Word2Vec(ngrams, min_count=1, size=self.embedding_size, window=self.embedding_window)
        self.embedding_dict = dict(zip(model.wv.index2word, model.wv.syn0))
        res = []
        for sequence in sequences:
            sequence_vectors = []
            for i in range(0, len(sequence) - self.n + 1, self.n):
                ngram = ''
                for j in range(self.n):
                    ngram += sequence[i + j]
                vector = self.embedding_dict[ngram]
                sequence_vectors.append(vector)
            res.append(sequence_vectors)
        return res
    """

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

    """
    def sequences_word2vec_devectorization(self, sequences):
        res = []
        for sequence in sequences:
            text = ''
            for vector in sequence:
                norm = np.linalg.norm(vector)
                symbol = ''
                for key, val in self.embedding_sorted_norms_dict.items():
                    if val <= norm:
                        symbol = key
                    else:
                        break
                text += symbol
            res.append(text)
        return res
    """

    """
    @staticmethod
    def sequences_one_hot_devectorization(sequences, idxs_to_grams_correspond):
        res = []
        for sequence in sequences:
            text = ''
            for index in sequence:
                text += idxs_to_grams_correspond[index]
            res.append(text)
        return res
    """

    """
    @staticmethod
    def get_sorted_norms_dict(vectors):
        norms = dict()
        for key, val in vectors.items():
            norms[key] = np.linalg.norm(val)
        reverse_norms = {v: k for k, v in norms.items()}
        sorted_reverse_norms = OrderedDict(sorted(reverse_norms.items()))
        res = {v: k for k, v in sorted_reverse_norms.items()}
        return res
    """

    """
    @staticmethod
    def add_margin(sequences, margin):
        res = []
        for sequence in sequences:
            for _ in range(margin):
                sequence += ' '
            res.append(sequence)
        return res
    """

    """
    @staticmethod
    def get_vector_length_max(sequences):
        res = 0
        for sequence in sequences:
            for element in sequence:
                if len(element) > res:
                    res = len(element)
        return res
    """

    """
    @staticmethod
    def lead_vector_to_same_length(sequences, length):
        res = []
        for sequence in sequences:
            res_sequence = []
            for element in sequence:
                res_element = element[:]
                for _ in range(len(element), length, 1):
                    res_element.append(0.0)
                res_sequence.append(res_element)
            res.append(res_sequence)
        return res
    """

    @staticmethod
    def count_elements(dict, sequences, n=1):
        res = []
        for sequence in sequences:
            res_seq = []
            for i in range(len(dict)):
                number = 0
                if n > 0:
                    for j in range(0, len(sequence) - n + 1, n):
                        ngram = ''
                        for k in range(n):
                            ngram += sequence[j + k]
                        if ngram == dict[i]:
                            number += 1
                else:
                    for j in range(len(sequence)):
                        if sequence[j] == dict[i]:
                            number += 1
                res_seq.append(number)
            res.append(res_seq)
        return res

    @staticmethod
    def get_max_number(sequences):
        res = 0
        for sequence in sequences:
            m = max(sequence)
            res = m if m > res else res
        return res

    @staticmethod
    def number_sequences_vectorization(sequences, grams_to_idxs_correspond, sequences_elements_number, length, n=1):
        res_sequences = []
        for i in range(len(sequences)):
            sequence = sequences[i]
            res_sequence = []
            if n > 0:
                for j in range(0, len(sequence) - n + 1, n):
                    ngram = ''
                    for k in range(n):
                        ngram += sequence[j + k]
                    res_vector = [0 for _ in range(length)]
                    k = grams_to_idxs_correspond[ngram]
                    l = sequences_elements_number[i][k]
                    res_vector[l] = 1
                    res_sequence.append(res_vector)
            else:
                for j in range(len(sequence)):
                    res_vector = [0 for _ in range(length)]
                    k = grams_to_idxs_correspond[sequence[j]]
                    l = sequences_elements_number[i][k]
                    res_vector[l] = 1
                    res_sequence.append(res_vector)
            res_sequences.append(res_sequence)
        return res_sequences

    @staticmethod
    def concatenate(vectorized_sequences, vectorized_sequences_elements_number_sequences):
        res_sequences = []
        for i in range(len(vectorized_sequences)):
            res_sequence = []
            for j in range(len(vectorized_sequences[i])):
                res_vector = vectorized_sequences[i][j][:] + vectorized_sequences_elements_number_sequences[i][j][:]
                res_sequence.append(res_vector)
            res_sequences.append(res_sequence)
        return res_sequences

    @staticmethod
    def normalize(sequences, number):
        for i in range(len(sequences)):
            for j in range(len(sequences[i])):
                sequences[i][j] /= number
        return sequences

    @staticmethod
    def create_sequences_from_vectors(vectors, sequences):
        res_sequences = []
        for i in range(len(sequences)):
            res_sequence = []
            for j in range(len(sequences[i])):
                res_sequence.append(vectors[i][:])
            res_sequences.append(res_sequence)
        return res_sequences

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

    """
    @staticmethod
    def get_identification_vectors(headers):
        questions_codes = [[int(element) for element in tuple] for tuple in itertools.product('01', repeat=3)]
        res = []
        for header in headers:
            if re.match(r'Вопрос\s+1\.\s*', header):
                res.append(questions_codes[0])
            elif re.match(r'Вопрос\s+2\.\s*', header):
                res.append(questions_codes[1])
            elif re.match(r'Вопрос\s+3\.\s*', header):
                res.append(questions_codes[2])
            elif re.match(r'Вопрос\s+4\.\s*', header):
                res.append(questions_codes[3])
            elif re.match(r'Вопрос\s+5\.\s*', header):
                res.append(questions_codes[4])
            elif re.match(r'Вопрос\s+6\.\s*', header):
                res.append(questions_codes[5])
            elif re.match(r'Вопрос\s+7\.\s*', header):
                res.append(questions_codes[6])
            else:
                print("Undefined question type '" + header + "'!")
        return res
    """

    @staticmethod
    def get_max_groups_number(outputs):
        res = reduce(lambda x, y: x if x > y else y,
                     list(map(lambda output:
                              reduce(lambda x, y: x if x > y else y,
                                     list(map(len, output))), outputs)))
        return res

    @staticmethod
    def get_max_group_number(outputs):
        res = reduce(lambda x, y: x if x > y else y,
                     list(map(lambda output:
                              reduce(lambda x, y: x if x > y else y,
                                     list(map(lambda groups:
                                              reduce(lambda x, y: x if x > y else y, groups), output))), outputs)))
        return res

    def encoding(self, outputs, qid_vectors):
        res = list(map(lambda output:
                       list(map(lambda groups:
                                list(map(lambda group: list(self.groups_codes[group]), groups)), output)), outputs))
        res = list(map(lambda output:
                       list(map(lambda groups:
                                reduce(lambda x, y: x + y, groups), output)), res))
        res = list(map(lambda qid_vector, output:
                       list(map(lambda x: qid_vector + x, output)), qid_vectors, res))
        return res

    @staticmethod
    def masking(sequences, length):
        res_secuences = []
        for sequence in sequences:
            res_secuence = []
            for vector in sequence:
                res_vector = vector[:] + [0 for _ in range(len(vector), length, 1)]
                res_secuence.append(res_vector)
            res_secuences.append(res_secuence)
        return res_secuences

    @staticmethod
    def reshape_sequences(sequences, target, codes=None):
        res_sequences = []
        res_codes_sequences = []
        for i in range(len(target)):
            res_sub_sequence = []
            res_codes_sub_sequence = []
            for j in range(len(target[i])):
                if type(sequences[i][0]) is list:
                    res_sequence = []
                    for k in range(len(sequences[i])):
                        res_sequence.append(sequences[i][k][:])
                    res_sub_sequence.append(res_sequence)
                else:
                    res_sub_sequence.append(sequences[i][:])
                if codes is not None:
                    res_codes_sub_sequence.append(codes[j])
            res_sequences.append(res_sub_sequence)
            res_codes_sequences.append(res_codes_sub_sequence)
        if codes is not None:
            return res_sequences, res_codes_sequences
        else:
            return res_sequences

    @staticmethod
    def flatten_sub_sequences(sequences):
        res = []
        for sub_seq in sequences:
            res += sub_seq
        return res

    @staticmethod
    def repeat_headers(headers, numbers):
        res = []
        for i in range(len(headers)):
            for _ in range(numbers[i]):
                res.append(headers[i])
        return res

    @staticmethod
    def get_reencoding_dict(sequences):
        res = []
        for sequence in sequences:
            for vector in sequence:
                res.append(json.dumps(vector))
        res = list(set(res))
        return res

    @staticmethod
    def get_reencoding_codes(dictionary):
        return [json.dumps([1 if i == j else 0 for j in range(len(dictionary))]) for i in range(len(dictionary))]

    @staticmethod
    def get_reencoding_correspond(dictionary, codes):
        return dict((dictionary[i], codes[i]) for i in range(len(dictionary)))

    @staticmethod
    def reencoding(sequences, correspond):
        res_sequences = []
        for sequence in sequences:
            res_sequence = []
            for vector in sequence:
                res_vector = correspond[json.dumps(vector)]
                res_sequence.append(json.loads(res_vector))
            res_sequences.append(res_sequence)
        return res_sequences

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
