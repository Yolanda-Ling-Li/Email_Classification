import os
import numpy as np
from keras.models import load_model
from w2v_lstm_model import Config
from sklearn.metrics import f1_score, accuracy_score


def predict_save(data_dir, x_predict, y_predict):

    x_predict = np.toarray(x_predict)
    np.save(os.path.join(data_dir, 'deep_test_x.npy'), x_predict)
    y_predict = np.toarray(y_predict)
    np.save(os.path.join(data_dir, 'deep_test_y.npy'), y_predict)
    return


def predict_load(data_dir):
    x_predict = np.load(os.path.join(data_dir, 'deep_test_x.npy'))
    y_predict = np.load(os.path.join(data_dir, 'deep_test_y.npy'))
    return x_predict,y_predict



def predict():
    config = Config()
    model = load_model(os.path.join(config.data_dir, 'Word2Vec_LSTM_Model.hdf5'))
    x_true, y_true = predict_load(config.data_dir)
    y_pred = model.predict_classes(x_true)


    f1_score(y_true, y_pred, average='macro')
    print("Word2Vec_LSTM_Model:the test data accuracy is %f" % (accuracy_score(y_true, y_pred)))
    print("Word2Vec_LSTM_Model:the test data F1-score is %f" % (f1_score(y_true, y_pred, average='micro')))


def main(): # 当.py文件被直接运行时将被运行，当.py文件以模块形式被导入时不被运行。
    predict()


if __name__ == "__main__":
    main()