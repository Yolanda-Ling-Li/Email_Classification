from keras.callbacks import ModelCheckpoint,TensorBoard
from w2v_lstm_model import *
from w2v_lstm_predict import *

def train_w2v():
    config = Config()
    if not os.path.exists(config.input_data_dir):
        os.makedirs(config.input_data_dir)

    X_train, X_test, y_train, y_test = w2v_load_data(config.input_data_dir)
    combine = X_train + X_test

    print('Training a Word2vec model...')
    x_shallow_data = word2vec_shallow_model(data=combine, config=config)
    x_shallow_train = x_shallow_data[0: len(X_train)]
    x_shallow_test = x_shallow_data[len(X_train):]
    shallow_train_data = list(zip(x_shallow_train, y_train))
    df = pd.DataFrame(data=shallow_train_data, columns=['x_train', 'y_train'])
    df.to_csv(os.path.join(config.data_dir, 'shallow_train_data.csv'))
    shallow_test_data = list(zip(x_shallow_test, y_test))
    df = pd.DataFrame(data=shallow_test_data, columns=['x_test', 'y_test'])
    df.to_csv(os.path.join(config.data_dir, 'shallow_test_data.csv'))

    n_symbols, embedding_weights, x_data = word2vec_model(data=combine, config=config)
    x_train = x_data[0: len(X_train)]
    x_test = x_data[len(X_train):]
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    print('Setting up Arrays for Keras Embedding Layer...')
    model = w2v_cnn_lstm_model(n_symbols=n_symbols, embedding_weights=embedding_weights, config=config)

    print('Compiling the Model Word2vec+LSTM...')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print('Train the Model Word2vec+LSTM...')
    cbs = [ModelCheckpoint(os.path.join(config.data_dir, 'Word2Vec_LSTM_Model.hdf5'),  # 存储模型参数的路径
                           monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
           # 只存储效果最好loss最小时的模型参数
           TensorBoard(log_dir=config.log_dir_file)]  # 存储loss，acc曲线文件的路径，可以用命令行+6006打开
    model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.num_epoch, validation_split=0.2,
              callbacks=cbs)

    predict_save(config.data_dir, x_test, y_test)


def main():
    train_w2v()

    # 当.py文件被直接运行时将被运行，当.py文件以模块形式被导入时不被运行。


if __name__ == "__main__":
    main()