from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.keras.backend import set_session
from auxiliary_functions import *
import pickle
import os
import json
import sys

def loadDNASet(dna_set_path):
    '''
    Function that loads the file relating the DNA sequence of a given region with the fold change
    of the ATAC-Seq and divides the set on train and test

    Parameters:
        dna_set_path (str): path to the file with the dna sequences per gene and the fold change of its gene

    Returns:
        The train and test sets used in the next functions
    '''

    # Load the input set from the specified path
    data = pd.read_csv(dna_set_path, index_col=0)
    data_ml = data.copy()
    data_ml = data_ml[['dna_seq_cut','response']]

    # Build the k-mers from the the input sequences
    texts = [' '.join(getKmers(i, 3)) for i in data_ml['dna_seq_cut'].values]
    # Build output variable
    Y = data_ml["response"]/data_ml["response"].max()

    # Divide in train and test
    np.random.seed(1)
    X_train,X_test,Y_train,Y_test = train_test_split(texts,Y.values,test_size=0.1)
    return X_train, X_test, Y_train, Y_test


def tuneModel(X_train, Y_train, model_type, model_params, n_splits, output_dir, gpu_to_use):
    '''
    Function that trains the network (LSTM or Convolutional) tuning different model parameters
    and saves the output of each of their combinations in the specified folder

    Parameters:
        X_train: input variables (i.e. the DNA sequences) of the training set
        Y_train: output variable of the training set
        model_type (str): type of model to train, namely "lstm" or "conv"
        model_params ({str:[int]}): dictionary with the params to tune and their possible values
        n_splits (int): number of partitions to make when applying the stratified k-fold cross-validation
        output_dir (str): path to the directory where the outputs of the model will be saved
        gpu_to_use (str): string with the name of the gpu that is going to be used in the training

    Returns:
        histories ({str:[float]}): a dictionary that relates each combination with the results obtained
                                   in each split of the cross-validation
    '''

    histories = {}
    lr = 0.0001
    epochs = 50
    batch_size = 60
    input_x_shape = X_train.shape[1]
    input_length = input_x_shape-2
    with tf.device("/"+gpu_to_use):
        folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=123).split(X_train, Y_train))

        # First, we tokenize each sample for each fold in order not to have to do it later in every iteration
        # The training and validation tests after encoding are saved in lists
        X_train_enc = list()
        X_valid_enc = list()
        input_dim_enc = list()
        for i, (train_idx, val_idx) in enumerate(folds):
            X_train_cv = np.array(X_train)[train_idx]
            X_valid_cv = np.array(X_train)[val_idx]

            tokenizer = Tokenizer(oov_token = True)
            tokenizer.fit_on_texts(X_train_cv)
            encoded_train = tokenizer.texts_to_sequences(X_train_cv)
            encoded_val = tokenizer.texts_to_sequences(X_valid_cv)
            X_train_enc.append(pad_sequences(encoded_train))
            X_valid_enc.append(pad_sequences(encoded_val))
            input_dim_enc.append(len(tokenizer.word_index) + 1)

        # Build combinations of parameters and keep it as a list of dictionaries
        combinations = list(dict_product(model_params))

        for param_values in combinations:
            print("\Combination:", param_values)
            model_name = "-".join([key+"-"+str(value) for key, value in param_values.items()])
            histories[model_name] = list()
            # Execute the model for each fold
            for i, (train_idx, val_idx) in enumerate(folds):
                print("\nFold",i)
                X_train_cv = X_train_enc[i]
                Y_train_cv = Y_train[train_idx]
                X_valid_cv = X_valid_enc[i]
                Y_valid_cv = Y_train[val_idx]
                input_dim = input_dim_enc[i]

                # Load and compile the model
                if model_type == "lstm":
                    if 'nodos2' in model_params:
                        model = create_lstm_regr_embedding(input_dim, param_values["output_dim"], input_length, param_values["nodos1"],
                                                      param_values["dropout"], lr, True, param_values["nodos2"])
                    else:
                        model = create_lstm_regr_embedding(input_dim, param_values["output_dim"], input_length, param_values["nodos1"],
                                                      param_values["dropout"], lr)
                elif model_type == "conv":
                    model = create_conv_regr_embedding(input_dim, param_values["output_dim"], input_length, param_values["filters"],
                                                  param_values["kernel_size"], param_values["dropout"], lr)

                # Run the model
                output = OutputObserver(X_valid_cv, X_train_cv)
                historyIt = model.fit(X_train_cv, Y_train_cv, batch_size = batch_size, epochs = epochs,
                                      validation_data = (X_valid_cv, Y_valid_cv), verbose = 0, callbacks = [output])
                r2_scores_val = [r2_score(Y_valid_cv, np.squeeze(predictions)) for predictions in output.pred_val]
                r2_scores_train = [r2_score(Y_train_cv, np.squeeze(predictions)) for predictions in output.pred_train]

                # Save the results
                histories[model_name].append(historyIt.history)
                save_name = model_name + "-" +str(i+1)

                # Save the model as JSON
                model_json = model.to_json()
                with open(output_dir+"/model"+save_name+".json", "w") as json_file:
                    json_file.write(model_json)

                # Serialize the weights to HDF5
                model.save_weights(output_dir+"/model"+save_name+".h5")
                print("Saved model to disk")
                # Create the plots associated with the current regression results
                create_plots_regression(historyIt, output_dir+"/plots", save_name, r2_scores_val, r2_scores_train)

        # Save the regression results of all combinations
        with open(output_dir+'/histories.pkl', 'wb') as handle:
            pickle.dump(histories, handle)
