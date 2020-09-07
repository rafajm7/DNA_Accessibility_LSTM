import configparser
import sys
import os
import classification.TrainOneHot as model_clf_onehot
import classification.TrainEmbedding as model_clf_emb
import regression.TrainOneHot as model_regr_onehot
import regression.TrainEmbedding as model_regr_emb

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.keras.backend import set_session

if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    general_params = config["general_params"]

    # Load input params
    dna_set_path = general_params.get("dna_set_path")
    cutting_intervals_str = general_params.get("cutting_intervals")
    cutting_intervals = [float(i) for i in cutting_intervals_str.split(",")]
    problem_type = general_params.get("problem_type")
    model_type = general_params.get("model_type")
    encoding_type = general_params.get("encoding_type")
    region_type = general_params.get("region_type")
    chain_type = general_params.get("chain_type")
    n_undersamplings = general_params.getint("n_undersamplings")
    n_splits = general_params.getint("n_splits")
    gpu_to_use = general_params.get("gpu_to_use")
    output_dir = general_params.get("output_dir")
    output_dir = "{}/{}/models_{}/model_{}/{}".format(output_dir, problem_type, encoding_type, region_type, chain_type)

    # Create output folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load model params
    model_section = '-'.join([model_type, encoding_type])
    model_params = {}
    for key, value in config.items(model_section):
        if key == "dropout":
            model_params[key] = [float(i) for i in value.split(",")]
        else:
            model_params[key] = [int(i) for i in value.split(",")]

    # Run the corresponding module
    if problem_type == "classification" and encoding_type == "onehot":
        X_train, X_test, Y_train, Y_test = model_clf_onehot.loadDNASet(dna_set_path, cutting_intervals)
        histories = model_clf_onehot.tuneModel(X_train, Y_train, model_type, model_params, n_undersamplings, n_splits, output_dir, gpu_to_use)
        model, batch_size = model_clf_onehot.trainBestModel(X_train, X_test, Y_train, Y_test, model_type, histories, gpu_to_use)
        model_clf_onehot.testFinalModel(model, X_test, Y_test, batch_size)
    elif problem_type == "classification" and encoding_type == "embedding":
        X_train, X_test, Y_train, Y_test = model_clf_emb.loadDNASet(dna_set_path, cutting_intervals)
        histories = model_clf_emb.tuneModel(X_train, Y_train, model_type, model_params, n_undersamplings, n_splits, output_dir, gpu_to_use)
        model, batch_size, tokenizer = model_clf_emb.trainBestModel(X_train, X_test, Y_train, Y_test, model_type, histories, gpu_to_use)
        model_clf_emb.testFinalModel(model, X_test, Y_test, batch_size, tokenizer)
    elif problem_type == "regression" and encoding_type == "onehot":
        X_train, X_test, Y_train, Y_test = model_regr_onehot.loadDNASet(dna_set_path)
        model_regr_onehot.tuneModel(X_train, Y_train, model_type, model_params, n_splits, output_dir, gpu_to_use)
    else:
        X_train, X_test, Y_train, Y_test = model_regr_emb.loadDNASet(dna_set_path)
        model_regr_emb.tuneModel(X_train, Y_train, model_type, model_params, n_undersamplings, n_splits, output_dir, gpu_to_use)
