import re
import matplotlib.pyplot as plt
import datetime
import sys
import fnmatch

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K

from dataprep import DataPreparation

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save_model(model, model_path="json\\model.json", weights_path="h5\\weights.h5"):
    json_model = model.to_json()
    json_file = open(model_path, "w")
    json_file.write(json_model)
    json_file.close()
    model.save_weights(weights_path)


def load_model(model_path="json\\model.json", weights_path="h5\\weights.h5"):
    json_file = open(model_path, "r")
    json_model = json_file.read()
    json_file.close()
    model = model_from_json(json_model)
    model.load_weights(weights_path)
    model.compile(
        optimizer=RMSprop(lr=0.01, decay=1E-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_model(data):
    input_1 = Input(shape=(None, len(data.input_dict)))
    lstm_1 = LSTM(64, return_sequences=True, dropout=0.0, recurrent_dropout=0.0)(input_1)
    dense_1 = Dense(len(data.output_dict), activation='softmax')(lstm_1)
    model = Model(inputs=input_1, outputs=dense_1)
    model.compile(
        optimizer=RMSprop(lr=0.01, decay=1E-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


class SaveModel(Callback):
    def __init__(self, model_path, weights_path):
        super().__init__()
        self.model_path = model_path
        self.weights_path = weights_path

    def on_epoch_end(self, epoch, logs=None):
        id = re.sub(r'[\s:.-]', '', str(datetime.datetime.now()))
        model_path = os.path.join(self.model_path, 'model_' + id + '.json')
        weights_path = os.path.join(self.weights_path, 'weights_' + id + '.h5')
        save_model(self.model, model_path=model_path, weights_path=weights_path)


def train_model(model, data, model_path, weights_path, epochs=1):
    res = model.fit_generator(
        generator=generator(data.train_x, data.train_y[0]),
        steps_per_epoch=len(data.train_x[0]),
        epochs=epochs,
        validation_data=generator(data.valid_x, data.valid_y[0]),
        validation_steps=len(data.valid_x[0]),
        callbacks=[SaveModel(model_path, weights_path)]
    )
    return res


def generator(x, y):
    while True:
        for i in range(len(x[0])):
            yield (x[0][i], y[i])


def predict(model, data):
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
    return predict_batches


def draw_graphs(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


def write_batches_to_file(filename, batches, is_seq=True):
    f = open(filename, 'w', encoding='utf-8')
    for i in range(len(batches)):
        f.write('=============================================\n')
        f.write('Батч ' + str(i + 1) + ': \n')
        for j in range(len(batches[i])):
            if is_seq:
                f.write('-------------------------------------------\n')
                f.write('Последовательность ' + str(j + 1) + ': \n')
                for k in range(len(batches[i][j])):
                    f.write('Вектор ' + str(k + 1) + ': \n')
                    f.write(str(batches[i][j][k]) + '\n')
                f.write('-------------------------------------------\n')
            else:
                f.write('Вектор ' + str(j + 1) + ': \n')
                f.write(str(batches[i][j]) + '\n')
        f.write('=============================================\n')
    f.close()


def parse_args():
    if '-f' in sys.argv:
        match = fnmatch.filter(sys.argv, 'name=*')
        if len(match) > 0:
            match = re.match(r'name=(?P<id>.+)', match[0])
            id = match.group('id')
        else:
            id = re.sub(r'[\s:.-]', '', str(datetime.datetime.now()))
        model_path = os.path.join('json', id)
        weights_path = os.path.join('h5', id)
        out_file_name = id + '.txt'
        sys.stdout = open(out_file_name, 'w+')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
    else:
        model_path = 'json'
        weights_path = 'h5'
    return model_path, weights_path


if __name__ == "__main__":
    model = None
    model_path, weights_path = parse_args()
    if '-ld' in sys.argv:
        load_from_file = 'data.zip'
    else:
        load_from_file = None
    data = DataPreparation(
        'txt\\train_data\\input.txt',
        'txt\\train_data\\output.txt',
        'txt\\valid_data\\input.txt',
        'txt\\valid_data\\output.txt',
        'txt\\test_data\\input.txt',
        max_batch_size=2,
        load_from_file=load_from_file,
    )
    if '-s' in sys.argv:
        data.save('data')
    if '-c' in sys.argv:
        model = create_model(data)
        save_model(
            model,
            model_path=os.path.join(model_path, 'model.json'),
            weights_path=os.path.join(weights_path, 'weights.h5'),
        )
    if '-l' in sys.argv:
        model = load_model(
            model_path=os.path.join(model_path, 'model.json'),
            weights_path=os.path.join(weights_path, 'weights.h5'),
        )
    if '-t' in sys.argv:
        if '-c' in sys.argv or '-l' in sys.argv:
            history = train_model(
                model,
                data,
                model_path=model_path,
                weights_path=weights_path,
                epochs=1,
            )
        else:
            print("The model can't train. You need create or load model. "
                  "Use flags '-c' to create model or '-l' to load model.")
    if '-p' in sys.argv:
        if '-c' in sys.argv or '-l' in sys.argv:
            predict_batches = predict(model, data)
            f = open("output.txt", "w", encoding='utf-8')
            for i in range(len(predict_batches)):
                for j in range(len(predict_batches[i])):
                    body = ''.join(json.loads(predict_batches[i][j], encoding='utf-8')).replace(' ', '')
                    body = re.sub(r'_+', '_', body)
                    body = body.replace('_', 'T')
                    f.write(body + '\n')
            f.close()
        else:
            print("The model can't predict. You need create or load model. "
                  "Use flags '-c' to create model or '-l' to load model.")
