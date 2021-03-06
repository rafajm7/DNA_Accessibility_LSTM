from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.preprocessing.sequence import pad_sequences
from auxiliary_functions import *
import scipy.stats
import pickle
import os
import json
import sys

def loadDNASet(dna_set_path, cutting_intervals):
    '''
    Function that loads the file relating the DNA sequence of a given region with the fold change in ATAC-Seq,
    obtains the output class based on a cutoff of the fold change and divides the set on train and test

    Parameters:
        dna_set_path (str): path to the file with the dna sequences per gene and the fold change of its gene
        cutting_intervals ((int,int)): tuple with the partition of the fold change that defines the two classes

    Returns:
        The train and test sets used in the next functions
    '''

    # Load the input set from the specified path
    data = pd.read_csv(dna_seq_path, index_col=0)
    data_ml = data.copy()
    data_ml = data_ml[['dna_seq_cut','response']]
    # Keep only the samples that have a fold change lower than the first element of the partition or higher than the second element
    data_ml = data_ml.loc[(data_ml["response"] < cutting_intervals[0]) | (data_ml["response"] > cutting_intervals[1])]

    # Build the k-mers from the the input sequences
    texts = [' '.join(getKmers(i, 3)) for i in data_ml['dna_seq_cut'].values]
    # Build output variable
    Y = pd.cut(data_ml['response'], [0,1,max(data_ml["response"])], labels = [0,1]).values
    Y = np.array(Y)
    Y = pd.Series(Y, index = data_ml[["response"]].index)
    X_df = pd.DataFrame(np.array(texts), index = Y.index)

    # Divide in train and test
    X_train,X_test,Y_train,Y_test = train_test_split(X_df,Y, stratify = Y, test_size=0.1,random_state=1)
    return X_train, X_test, Y_train, Y_test


def tuneModel(X_train, Y_train, model_type, model_params, n_undersamplings, n_splits, output_dir, gpu_to_use):
    '''
    Function that trains the network (LSTM or Convolutional) tuning different model parameters
    and saves the output of each of their combinations in the specified folder

    Parameters:
        X_train: input variables (i.e. the DNA sequences) of the training set
        Y_train: output variable of the training set
        model_type (str): type of model to train, namely "lstm" or "conv"
        model_params ({str:[int]}): dictionary with the params to tune and their possible values
        n_undersamplings (int): number of times to do the random undersampling to the training test
                                to obtain more reliable results
        n_splits (int): number of partitions to make when applying the stratified k-fold cross-validation
        output_dir (str): path to the directory where the outputs of the model will be saved
        gpu_to_use (str): string with the name of the gpu that is going to be used in the training

    Returns:
        histories ({str:[float]}): a dictionary that relates each combination with the results obtained
                                   in each split of the cross-validation for each of the undersamplings
    '''

    histories = {}
    lr = 0.0001
    input_x_shape = X_train.shape[1]
    input_length = input_x_shape-2
    with tf.device("/"+gpu_to_use):
        for u in range(1,n_undersamplings+1):
            print("U =",u)
            X_train_u, Y_train_u = random_undersample(X_train, Y_train, "embedding")
            print("Number of genes that don't respond", len(Y_train[Y_train == 0]))
            print("Number of genes that respond", len(Y_train[Y_train == 1]))
            print("Number of genes that don't respond after undersampling", len(Y_train_u[Y_train_u == 0]))
            print("Number of genes that respond after undersampling", len(Y_train_u[Y_train_u == 1]))
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(X_train_u, Y_train_u))

            # First, we tokenize each sample for each fold in order not to have to do it later in every iteration
            # The training and validation tests after encoding are saved in lists
            X_train_enc = list()
            X_valid_enc = list()
            input_dim_enc = list()
            for i, (train_idx, val_idx) in enumerate(folds):
                X_train_cv = X_train_u[train_idx]
                X_valid_cv = X_train_u[val_idx]
                # Create the Tokenizer object
                tokenizer = Tokenizer(oov_token = True)
                # Fit tokenizer
                tokenizer.fit_on_texts(X_train_cv)
                # Apply tokenizer
                encoded_train = tokenizer.texts_to_sequences(X_train_cv)
                encoded_val = tokenizer.texts_to_sequences(X_valid_cv)
                X_train_enc.append(pad_sequences(encoded_train))
                X_valid_enc.append(pad_sequences(encoded_val))
                input_dim_enc.append(len(tokenizer.word_index) + 1)

            # Build combinations of parameters and keep it as a list of dictionaries
            combinations = dict_product(model_params)

            for param_values in combinations:
                print("\Combination:", param_values)
                # Check to avoid repeating experiments with only one layer and changing the nodes in the sexond (non existing) layer
                # Basically, we only run it for the first value of nodos2, and then we remove the "nodos2" part from the model name
                if not (param_values["ncapas"] == 1 and param_values["nodos2"] != model_params["nodos2"][0]):
                    model_name = "-".join([key+"-"+str(value) for key, value in param_values.items() if param_values["ncapas"] == 2 or key != "nodos2"])
                    # Create the list in the first undersampling iteration, so as to add the results of all undersamplings
                    if u == 1:
                        histories[model_name] = list()
                    # Execute the model for each fold
                    for i, (train_idx, val_idx) in enumerate(folds):
                        print("\nFold",i)
                        X_train_cv = X_train_enc[i]
                        Y_train_cv = Y_train_u[train_idx]
                        X_valid_cv = X_valid_enc[i]
                        Y_valid_cv = Y_train_u[val_idx]
                        input_dim = input_dim_enc[i]

                        # Load and compile the model
                        if model_type == "lstm":
                            if 'nodos2' in model_params:
                                model = create_lstm_clf_embedding(input_dim, param_values["output_dim"], input_length, param_values["nodos1"],
                                                              param_values["dropout"], lr, True, param_values["nodos2"])
                            else:
                                model = create_lstm_clf_embedding(input_dim, param_values["output_dim"], input_length, param_values["nodos1"],
                                                              param_values["dropout"], lr)
                        elif model_type == "conv":
                            model = create_conv_clf_embedding(input_dim, param_values["output_dim"], input_length, param_values["filters"],
                                                          param_values["kernel_size"], param_values["dropout"], lr)

                        # Run the model
                        historyIt = model.fit(X_train_cv, Y_train_cv, batch_size= param_values["batch_size"], epochs=param_values["epochs"],
                                            validation_data = (X_valid_cv, Y_valid_cv), verbose = 0)

                        # Save the final sensitivity and specificity on the validation data
                        preds = model.predict(X_valid_cv)
                        historyIt.history['final_val_sensitivity'] = sensitivity(Y_valid_cv, np.squeeze(preds)).numpy()
                        historyIt.history['final_val_specificity'] = specificity(Y_valid_cv, np.squeeze(preds)).numpy()
                        histories[model_name].append(historyIt.history)

                        save_name = model_name + "-" +str(i+1)
                        # Save the model as JSON
                        model_json = model.to_json()
                        with open(output_dir+"/model"+save_name+".json", "w") as json_file:
                            json_file.write(model_json)

                        # Serialize the weights to HDF5
                        model.save_weights(output_dir+"/model"+save_name+".h5")
                        print("Saved model to disk")
                        # Create the plots associated with the current classification results
                        create_plots(historyIt, output_dir+"/plots", save_name)

                    # Save the classification results of all combinations
                    with open(output_dir+'/histories.pkl', 'wb') as handle:
                        pickle.dump(histories, handle)
    return histories


def trainBestModel(X_train, X_test, Y_train, Y_test, model_type, histories, gpu_to_use):
    '''
    Function that loads the results of the tuning step, selects the best model using the mean
    confidence interval of the accuracy and trains it using the whole training data

    Parameters:
        X_train: input variables (i.e. the DNA sequences) of the training set
        X_test: input variables (i.e. the DNA sequences) of the test set
        Y_train: output variable of the training set
        Y_test: output variable of the test set
        model_type (str): type of model to train, namely "lstm" or "conv"
        histories ({str:[float]}): a dictionary that relates each combination with the results obtained
                                   in each split of the cross-validation of each of the undersamplings

    Returns:
        The final model trained and the batch size used in training
    '''

    # Apply random undersampling to training data
    X_train_u, Y_train_u = random_undersample(X_train, Y_train, "embedding")

    # Tokenize the undersampled training set
    tokenizer = Tokenizer(oov_token = True)
    tokenizer.fit_on_texts(X_train_u)
    encoded_train = tokenizer.texts_to_sequences(X_train_u)
    X_train_enc = pad_sequences(encoded_train)
    input_dim = len(tokenizer.word_index) + 1

    # Build confidence interval from the results of the tuning step
    accs = []
    conf_interval_accs = {}
    for key in histories:
        for i,fold in enumerate(histories[key]):
            accs.append(fold['val_accuracy'][-1])

        conf_interval_accs[key] = mean_confidence_interval(np.array(accs))
    # Select best model as the one with the highest mean accuracy
    best_model = max(conf_interval_accs.items(), key=lambda elem:elem[1][0])[0]
    # Build dictionary with the parameters and values used in the best model, from the model name
    model_str_as_list = iter(best_model.split("-"))
    best_params = {param:value for (param, value) in list(zip(model_str_as_list,model_str_as_list))}

    # Load and compile the model
    with tf.device("/"+gpu_to_use):
        lr = 0.0001
        input_x_shape = X_train_u.shape[1]
        input_length = input_x_shape-2
        if model_type == "lstm":
            if 'nodos2' in best_params:
                model = create_lstm_clf_embedding(input_dim, best_params["output_dim"], input_length, best_params["nodos1"],
                                              best_params["dropout"], lr, input_x_shape, True, best_params["nodos2"], input_x_shape)
            else:
                model = create_lstm_clf_embedding(input_dim, best_params["output_dim"], input_length, best_params["nodos1"],
                                              best_params["dropout"], lr, input_x_shape)
        elif model_type == "conv":
            model = create_conv_clf_embedding(input_dim, best_params["output_dim"], input_length, best_params["filters"],
                                          best_params["kernel_size"], lr, best_params["dropout"], input_x_shape)
        # Run the model
        historyIt = model.fit(X_train_enc, Y_train_u, batch_size = best_params["batch_size"], epochs=best_params["epochs"])
    return model, best_params["batch_size"], tokenizer


def testFinalModel(model, X_test, Y_test, batch_size, tokenizer):
    '''
    Function that evaluates the model trained in the previous step and prints the confusion matrix
    of the prediction

    Parameters:
        model: the model trained using the best parameters
        X_test: input variables (i.e. the DNA sequences) of the test set
        Y_test: output variable of the test set
        batch_size (int): the batch size used in training
        tokenizer: the Tokenizer object fitted with the training data
    '''

    # Apply random undersampling to test data
    X_test_u, Y_test_u = random_undersample(X_test, Y_test, "embedding")
    # Tokenize test data
    encoded_test = tokenizer.texts_to_sequences(X_test_u)
    X_test_enc = pad_sequences(encoded_test)

    # Evaluate the model on test data
    model.evaluate(X_test_enc, Y_test_u, batch_size=batch_size)
    preds = model.predict(X_test_u)
    # Get sensitivity and specificity of the predictions
    final_sensitivity = sensitivity(Y_test_u, np.squeeze(preds)).numpy()
    final_specificity = specificity(Y_test_u, np.squeeze(preds)).numpy()
    print("The sensitivity of the prediction has been:", final_sensitivity)
    print("The specificity of the prediction has been:", final_specificity)

    # Build the confusion matrix of the prediction
    bool_preds = np.squeeze((preds>=0.5).astype(int))
    CM = confusion_matrix(Y_test_u, bool_preds)
    print("The confusion matrix has been:")
    print(CM)
