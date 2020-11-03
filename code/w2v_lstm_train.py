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
    predict_save(config.data_dir, x_train, y_train, x_test, y_test)

    print('Setting up Arrays for Keras Embedding Layer...')
    model = w2v_cnn_lstm_model(n_symbols=n_symbols, embedding_weights=embedding_weights, config=config)

    print('Compiling the Model Word2vec+LSTM...')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    print('Train the Model Word2vec+LSTM...')
    cbs = [ModelCheckpoint(os.path.join(config.data_dir, 'Word2Vec_LSTM_Model.hdf5'),
                           monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False),
           TensorBoard(log_dir=config.log_dir_file)]
    model.fit(x_train, y_train, batch_size=config.batch_size, epochs=config.num_epoch, validation_split=config.valid_split,
              callbacks=cbs)


def main():
    train_w2v()



if __name__ == "__main__":
    main()