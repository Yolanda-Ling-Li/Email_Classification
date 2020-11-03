import os
import numpy as np
from keras.models import load_model
from w2v_lstm_model import Config
from sklearn.metrics import f1_score, accuracy_score


def predict_save(data_dir, x_train, y_train, x_predict, y_predict):
    np.save(os.path.join(data_dir, 'deep_train_x.npy'), x_train)
    np.save(os.path.join(data_dir, 'deep_train_y.npy'), y_train)
    np.save(os.path.join(data_dir, 'deep_test_x.npy'), x_predict)
    np.save(os.path.join(data_dir, 'deep_test_y.npy'), y_predict)


def predict_load(data_dir):
    x_train = np.load(os.path.join(data_dir, 'deep_train_x.npy'))
    y_train = np.load(os.path.join(data_dir, 'deep_train_y.npy'))
    x_predict = np.load(os.path.join(data_dir, 'deep_test_x.npy'))
    y_predict = np.load(os.path.join(data_dir, 'deep_test_y.npy'))
    return x_train, y_train, x_predict, y_predict



def predict():
    config = Config()
    model = load_model(os.path.join(config.data_dir, 'Word2Vec_LSTM_Model.hdf5'))
    x_true_train, y_true_train, x_true_test, y_true_test = predict_load(config.data_dir)
    y_true_train = np.argmax(y_true_train, axis=1)
    y_pred_train = model.predict_classes(x_true_train)
    y_true_test = np.argmax(y_true_test, axis=1)
    y_pred_test = model.predict_classes(x_true_test)

    print("Word2Vec_LSTM_Model:the train data accuracy is %f" % (accuracy_score(y_true_train, y_pred_train)))
    print("Word2Vec_LSTM_Model:the train data F1-score is %f" % (f1_score(y_true_train, y_pred_train, average='macro')))
    print("Word2Vec_LSTM_Model:the test data accuracy is %f" % (accuracy_score(y_true_test, y_pred_test)))
    print("Word2Vec_LSTM_Model:the test data F1-score is %f" % (f1_score(y_true_test, y_pred_test, average='macro')))


def main(): # 当.py文件被直接运行时将被运行，当.py文件以模块形式被导入时不被运行。
    predict()


if __name__ == "__main__":
    main()