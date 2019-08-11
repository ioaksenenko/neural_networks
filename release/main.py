import numpy as np
import json
import re
import tagmaskmaker as tmm
import tagmaker as tm
import cleaner as cl

from keras.models import model_from_json
from keras.optimizers import RMSprop

from dataprep import DataPreparation


def load_model(model_path="json\\model.json", weights_path="h5\\weights.h5", loss_function='categorical_crossentropy'):
    json_file = open(model_path, "r")
    json_model = json_file.read()
    json_file.close()
    model = model_from_json(json_model)
    model.load_weights(weights_path)
    model.compile(optimizer=RMSprop(lr=0.001), loss=loss_function, metrics=[loss_function])
    return model


def cleaner():
    parser = cl.HTMLParser()
    parser.parse_all_files(path_to_input_files='xhtml\\')
    parser.create_text_set()
    file = open("txt\\1_untagging_mask_maker\\input.txt", 'w', encoding="utf-8")
    is_header = False
    for text in parser.text_set:
        m = re.match(r'<p>\s*(?P<Q>Вопрос\s+\d+\.)\s*</p>', text)
        if m:
            file.write(m.group('Q') + '\n')
            is_header = True
        else:
            if is_header:
                if not re.match(r'^\s*$', text):
                    file.write(text)
                else:
                    if is_header:
                        file.write('\n')
                    is_header = False
    file.close()


def untagging_mask_maker():
    data = DataPreparation(
        test_input_file_path='txt\\1_untagging_mask_maker\\input.txt',
        load_from_file='zip\\1_untagging_mask_maker\\data.zip',
        max_batch_size=2,
    )
    headers = [[el] for el in data.test_input_headers]
    model = load_model(
        model_path="json\\1_untagging_mask_maker\\model.json",
        weights_path="h5\\1_untagging_mask_maker\\weights.h5",
    )
    predict_batches = []
    for i in range(len(data.test_x[0])):
        predicts = model.predict(data.test_x[0][i], verbose=0)
        predict_sequences = []
        for j in range(len(predicts)):
            predict_sequence = []
            for k in range(len(predicts[j])):
                arg_max = np.argmax(predicts[j][k])
                word = data.output_idxs_to_grams_correspond[arg_max]
                predict_sequence.append(word)
            predict_sequences.append(json.dumps(predict_sequence, ensure_ascii=False))
        predict_batches.append(predict_sequences)
    f1 = open("txt\\identifier\\input.txt", "w", encoding='utf-8')
    f2 = open("txt\\2_untagging_submasks_maker\\input_1.txt", "w", encoding='utf-8')
    f3 = open("txt\\1_untagging_mask_maker\\output.txt", "w", encoding='utf-8')
    for i in range(len(predict_batches)):
        for j in range(len(predict_batches[i])):
            f1.write(headers[i][j])
            f2.write(headers[i][j])
            f3.write(headers[i][j])
            body = ''.join(json.loads(predict_batches[i][j], encoding='utf-8'))
            f3.write(body + '\n')
            body = body.replace(' ', '')
            body = re.sub(r'_+', '_', body)
            body = body.replace('_', 'T')
            f1.write(body + '\n')
            f2.write(body + '\n')
    f1.close()
    f2.close()
    f3.close()


def identifier():
    data = DataPreparation(
        test_input_file_path='txt\\identifier\\input.txt',
        load_from_file='zip\\identifier\\data.zip',
        max_batch_size=1024,
    )
    headers = [[el] for el in data.test_input_headers]
    model = load_model(
        model_path="json\\identifier\\model.json",
        weights_path="h5\\identifier\\weights.h5"
    )
    predict_batches = []
    for i in range(len(data.test_x[0])):
        predicts = model.predict(data.test_x[0][i], verbose=0)
        predict_sequences = []
        for j in range(len(predicts)):
            predict_sequence = []
            for k in range(len(predicts[j])):
                predict_sequence.append(1 if predicts[j][k] >= 0.5 else 0)
            predict_sequences.append(json.dumps(predict_sequence))
        predict_batches.append(predict_sequences)
    f_1 = open("txt\\2_untagging_submasks_maker\\input_2.txt", "w", encoding='utf-8')
    f_2 = open("txt\\3_tagging_submasks_maker\\input_2.txt", "w", encoding='utf-8')
    for i in range(len(predict_batches)):
        for j in range(len(predict_batches[i])):
            f_1.write(headers[i][j])
            f_2.write(headers[i][j])
            f_1.write(predict_batches[i][j] + '\n')
            f_2.write(predict_batches[i][j] + '\n')
    f_1.close()
    f_2.close()


def untagging_submasks_maker():
    data = DataPreparation(
        test_input_file_path='txt\\2_untagging_submasks_maker\\input_1.txt',
        test_id_file_path='txt\\2_untagging_submasks_maker\\input_2.txt',
        load_from_file='zip\\2_untagging_submasks_maker\\data.zip',
        add_identification_data=True,
        is_out_sub_seq=True,
        max_batch_size=512,
    )
    model = load_model(
        model_path="json\\2_untagging_submasks_maker\\model.json",
        weights_path="h5\\2_untagging_submasks_maker\\weights.h5"
    )
    predict_sequences = []
    predict_sub_sequence = []
    for i in range(len(data.test_x[0])):
        predict_batches = model.predict([data.test_x[0][i], data.test_x[1][i], data.test_x[2][i]], verbose=0)
        predict_sequence = ""
        for k in range(len(predict_batches[0])):
            arg_max = np.argmax(predict_batches[0][k])
            predict_sequence += data.output_idxs_to_grams_correspond[arg_max]
        predict_sub_sequence.append(predict_sequence)
        if (i + 1) % data.train_sub_seq_max_len == 0:
            elems = list(filter(lambda x: re.match(r'^_+$', x), predict_sub_sequence))
            idx = predict_sub_sequence.index(elems[0]) if len(elems) > 0 else len(predict_sub_sequence)
            predict_sequences.append(predict_sub_sequence[:idx])
            predict_sub_sequence = []
    f = open("txt\\3_tagging_submasks_maker\\input_1.txt", "w", encoding='utf-8')
    for i in range(len(predict_sequences)):
        f.write(data.test_input_headers[i])
        f.write(json.dumps(predict_sequences[i], ensure_ascii=False) + '\n')
    f.close()


def tagging_submasks_maker():
    data = DataPreparation(
        test_input_file_path='txt\\3_tagging_submasks_maker\\input_1.txt',
        test_id_file_path='txt\\3_tagging_submasks_maker\\input_2.txt',
        load_from_file='zip\\3_tagging_submasks_maker\\data.zip',
        add_identification_data=True,
        is_input_data_in_json=True,
        is_in_sub_seq=True,
        max_batch_size=4,
    )
    model = load_model(
        model_path="json\\3_tagging_submasks_maker\\model.json",
        weights_path="h5\\3_tagging_submasks_maker\\weights.h5",
    )
    predict_samples = []
    for i in range(len(data.test_input_sequences)):
        predict_samples.append([])
    k = 0
    for i in range(len(data.test_x[0])):
        if len(predict_samples[k]) == len(data.test_input_sequences[k]):
            k += 1
        predicts = model.predict([data.test_x[0][i], data.test_x[1][i]], verbose=0)
        for j in range(len(predicts)):
            arg_max = np.argmax(predicts[j])
            word = data.output_idxs_to_grams_correspond[arg_max]
            predict_samples[k].append(word)
    f = open("txt\\4_tagging_mask_maker\\input.txt", "w", encoding='utf-8')
    for i in range(len(predict_samples)):
        f.write(data.test_input_headers[i])
        f.write(json.dumps(predict_samples[i], ensure_ascii=False) + '\n')
    f.close()


def tagging_mask_maker():
    input_dir = "txt\\4_tagging_mask_maker\\input.txt"
    output_dir = "txt\\5_tagging_maker\\input.txt"
    rules_dir = "json\\4_tagging_mask_maker\\rules.json"

    inputs = tmm.read_input(input_dir)
    rules = tmm.read_rules(rules_dir)
    outputs = tmm.tagging_mask_maker(inputs, rules)
    tmm.write_output(outputs, output_dir)


def tagging_maker():
    source_dir = 'txt\\1_untagging_mask_maker\\input.txt'
    masks_dir = 'txt\\1_untagging_mask_maker\\output.txt'
    tmasks_dir = 'txt\\5_tagging_maker\\input.txt'
    output_dir = 'txt\\5_tagging_maker\\output.txt'

    source = tm.read_input(source_dir)
    masks = tm.read_input(masks_dir)
    tmasks = tm.read_input(tmasks_dir)

    content = tm.get_content(tmasks)
    text = tm.get_text(source, masks, content)
    tmasks = tm.replace(tmasks, content, text)
    tm.write_output(tmasks, output_dir)


if __name__ == "__main__":
    cleaner()
    untagging_mask_maker() # 100.0% - 100.0%
    #identifier() # 100.0% - 100.0%
    #untagging_submasks_maker() # 99.9% - 99.9%
    #tagging_submasks_maker() # 99.99% - 95.04%
    #tagging_mask_maker()
    #tagging_maker()
