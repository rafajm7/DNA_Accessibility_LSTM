# LSTM Application to the prediction of DNA accessibility

DNA transcription is the first process of gene expression, through which the information contained in the DNA sequence is transferred to a copy of RNA for the subsequent generation of proteins. For this process to occur, DNA must be "open", in a state in which it is physically enabled for reading and decoding. However, not all regions of the genome are equally accessible when transcribing. There are sequencing methods, such as ATAC-Seq, that allow the identification of regions with a higher transcription capacity for the generation of proteins. The value returned by the ATAC-Seq method for each position of the genome corresponds to a measure of the opening of the DNA at that point to transcribe to protein.

In this project we have used ATAC-Seq data corresponding to a mouse's genome. More concretely, a experiment was conducted to the mentioned mouse, provoking an insult in his organism through a injection. Given that experiment, two sets of *.bam* files containing ATAC-Seq values were provided: one in which the mouse was in a baseline state, without being altered by any insult, and another corresponding to the state of the mouse one hour after the injection.

The main objective of the project is to study the prediction in the change of the ATAC-Seq of the mouse protein coding genes, using as input portions of DNA chains of the genes themselves. We will divide those portions into three: promoters, 5 'UTRs and 3' UTRs, which are known to be regions that control and regulate the transcription of genes to RNA.

While doing the study, two Machine Learning approaches were considered: classification and regression. In the first case two classes were defined based on a partition of the fold change mentioned in the previous paragraph: if the fold change was lower than a certain value (by default 0.8), we consider that the gene does not respond to the stimulus of the injection and it belongs to the positive class, and if the fold change was higher than another given threshold (by default 1.5), we consider that the gene responds to the stimulus and belongs to the negative class. In the case of regression, we use the fold change directly as the output variable. However, the training and validation regression results were way worse than those of classification and we primarily focused on the first case, discarding the first case. That's the reason why we have only included training functions for regression in the code.

In order to process the input sequences, we have focused on the application of Deep Learning algorithms. More concretely, we have mainly focused on LSTM networks, and we also have developed functions to implement convolutional neural networks.

Next we explain the two main submodules of the project, named __generateDNAATAC__ and __TrainTestLSTM__.


## First step: generate the (DNA,ATAC-Seq) set to use for training the LSTM

This first step of the project has been implemented in the module __generateDNAATAC__, and coded in __R__. The reason of using this language has been the number of libraries it has to manipulate genetic files such as the *.bams* that include the ATAC-Seq, a *.gtf* file with the specifications of the genome to use, and the extraction of the DNA sequences from the region's positions obtained from the *.gtf* file.

Given the *.bam* files with the ATAC-Seq coverage values for the given mouse, the objective of this submodule is to generate the set that is going to be introduced to the network. The steps followed to implement it are:

```r
1. Load the .bam files with the information about the ATAC-Seq for the whole mouse before and after the injection. It's of high importance to notice that the ATAC-Seq values are given per base pair.
2. Apply the mean across all their base pairs to get the ATAC-Seq coverage value for each protein coding gene. This way we end up having a two-row matrix M = {ATAC_ij: 1 <= i <= 2, 1 <= j <= n} containing the ATAC-Śeq values before the injection and the values one hour after the injection,being n the number of protein coding genes.
3. Using the last matrix, we compute the fold change for each gene as FC_j = M_2j/M_1j, or in other words, as the fraction between the ATAC-Seq one hour after the injection and the ATAC-Seq in the baseline state. That gives us the output variable for our prediction problem.
4. For each type of region mentioned in the introduction above (promoters, 5' UTRs and 3' UTRs) do:

  4.1. Use the latest .gtf file with information about the mouse's genome to extract the positions of the current type of region.
  4.2. Extract the DNA sequences of the given regions using the BSGenome library. Only keep those sequences with a size higher than an specified threshold.
  4.3. Link each region with its corresponding gene and build a dataframe that has as input variable the DNA sequence of the region and as output variable the fold change.
  4.4. Write the final dataframe to disk as a .csv file.

```

To execute this code we need the specified *.bam* files and the *.gtf* file with the mouse genome. The Linux command to run this module is:

```
$ Rscript -e 'source("DNA_Opening_LSTM/GenerateDNAATAC/R/build_dna_atac_set.R");buildRegionSets("path_to_gtf_file.gtf", "path_to_bam_data.bam", "path_to_coverage_matrix.csv", "path_to_output_dir/")'
```

## Second step: Tune, train and evaluate the LSTM network

The second module of the package has been named __TrainTestLSTM__, and is coded in __Python__, taking advantage of the Deep Learning package __keras__.

This module takes as input one of the .csv files generated in the previous module, and trains a Deep Learning model whose objective is to give the best possible prediction of the fold change. As stated earlier, we have focused in LSTM network models, and we have also implemented functions to train Convolutional Neural Networks, in order to compare the performance of both methods.

One important step when training with these kind of sequences is to select an appropiate encoding to make it possible to the network to learn from them. The easiest and most typical encoding is the __One Hot Encoding__, while another more complex type of encoding is called __Embedding__. Thanks to __keras__, the embedding is just a layer of the neural network that accepts as input the DNA sequence labelled as integers and generates a vector in other dimension that helps explain better the output variable. In other words, the __keras__ model not only improves the prediction by changing the weights associated to the network, but also changes the weights associated to the embedding to build an input vector representation that helps give a better prediction.

The benefits from using embeddings arise when we have sequences with very different values (i.e. a big vocabulary). If we applied one hot encoding to those sequences, for each element i in the sequence, a vector with size the size of the vocabulary, being the i-th element 1 and the rest of the vector 0. This would result in a memory problem when training the network. Using embeddings can give us the possibility of generating output vectors with lower dimensions while improving the predictive power of one-hot encodings.

Given that in our problem we only have 4 different values in the sequences (A, C, T or G), one would think that we may not need to use embeddings, as our vocabulary is very small. However, results have shown that considering groups of nucleotides instead of each one of the characters as a single element of the sequences give better performance when using Deep Learning models, like in this [project](https://towardsdatascience.com/lstm-to-detect-neanderthal-dna-843df7e85743). For that reason, we have considered groups of 3-mers as input to our models, resulting in a higher vocabulary, and, in consequence, in a good opportunity to apply embeddings combined with LSTM networks to improve the prediction.

To keep the possibility of choosing the type of encoding, we have added a configuration parameter that lets us pick between one-hot encoding and embeddings.

The steps followed in this module are:

```r
0. Load the initial parameters of the model: type of algorithm to run, type of encoding, type of region and list of possible values for the model parameters.
1. Load the corresponding regions .csv file generated from the previous module.
2. Build the output variable defining a partition of the fold change.
3. Separate data into 80% for training (Tr), 20% for evaluation (Ev).
4. Using the Tr set, undersample the majority class five times to do five 50/50 learning problems.
5. For each undersampling done to the training set:
  5.1. Apply 5-fold cross-validation.
  5.2. For all the possible combinations of the model parameters defined in step 0:
    5.2.1. Train the corresponding Deep Learning model (LSTM or convolutional) using the current iteration parameters.
    5.2.2. Save the results.
6. Pick the best model by choosing the one with the highest mean accuracy.
7. Train the best model with the whole training set.
8. Evaluate the final trained model in the Ev set, and return the final prediction with the confusion matrix.

```

To run this module we need to run the file __TrainTestLSTM/runModel.py__. The parameters needed for the execution are included in a configuration file (there is an example available in __TrainTestLSTM/runModel.cfg__). Those parameters are specify all the options mentioned here, together with some additional options like which GPU to use.

The module is then executed using the Linux command:

```
$ python runModel.py runModel.cfg
```
