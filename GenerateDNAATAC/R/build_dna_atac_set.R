#' Main function that builds the training set for each type of region (i.e. promoters, 3' UTR and 5' UTR),
#' calling the rest of the functions of the present script
#'
#' @param path_to_gtf_file String with the path to the most recent mouse GTF file
#' @param path_to_coverage_matrix String with the path where the coverage matrix will be saved
#' @param output_dir String with the output folder where the final sets will be stored
#'
#' @return
#' @export
#'
#' @examples
buildRegionSets <- function(path_to_gtf_file, path_to_bam_data, path_to_coverage_matrix, output_dir){

  # Load matrix with mean coverage per gene
  df_mean_coverage <- computeCoverageMatrix(path_to_gtf_file, path_to_bam_data, path_to_coverage_matrix)

  # Obtain output variable (solving zero-denominator problem)
  fold_changes <- (0.001+df_mean_coverage[2,])/(0.001+df_mean_coverage[1,])

  # Build cutoff list
  length_cutoff_list = list("promoters" = Inf, "three_prime" = 600, "five_prime" = 200)

  # We construct two dataframes per region type: one with reversed sequences and one with non-reversed sequences
  for (region_type in c("promoters", "three_prime", "five_prime")){

    # Load region data depending on type
    region_set = loadRegionData(path_to_gtf_file, region_type)

    # Build DNA sequences from region data
    dna_seq_list = buildDNASequences(region_set, region_type, length_cutoff_list[[region_type]])

    # Build the final dataframe for each type of sequence (reversed and normal)
    for (sequence_type in names(dna_seq_list)[-1]){
      output_path = paste0(output_dir, "/data_", sequence_type, "_", region_type)
      createFinalDataFrame(region_set = dna_seq_list$region_set, dna_sequences = dna_seq_list[[sequence_type]],
                           fold_changes, df_mean_coverage, output_path)
    }

  }
}


#' Function that, given two BAM files from the mouse genome (before and one hour after the injection),
#' builds the mean coverage matrix for each protein coding gene
#'
#' @param path_to_gtf_file String with the path to the most recent mouse GTF file
#' @param path_to_bam_data String with the folder that contains the BAM files
#' @param path_to_coverage_matrix String with the path where the mean coverage matrix will be saved
#'
#' @return The mean coverage matrix obtained from the .BAM data
#' @export
#'
#' @examples
computeCoverageMatrix <- function(path_to_gtf_file, path_to_bam_data, path_to_coverage_matrix){
  # Load GTF File
  library(refGenome)
  ens <- ensemblGenome()
  setwd(dirname(path_to_gtf_file))
  read.gtf(ens, basename(path_to_gtf_file))

  library(dplyr)
  library(tidyverse)
  # Filter protein coding genes
  all_genes = getGenePositions(ens) %>% as.data.frame()
  protein_coding_genes = all_genes %>% dplyr::select(gene_id, gene_name, seqid, start, end, strand, gene_biotype) %>%
    filter(gene_biotype == "protein_coding") %>%
    group_by(gene_id)

  # Bam lengths are obtained with the command samtools view -c -F 260 fichero.bam
  # They are used for the calculation of the normalized coverage
  # The unit of measure is millions of reads
  bamLengths = c(114.309119,136.959614)
  matrixMean = matrix(nr=2,nc=nrow(protein_coding_genes))
  rownames(matrixMean) = c('Veh0', 'KA1h')
  colnames(matrixMean) = protein_coding_genes$gene_id
  setwd(path_to_bam_data)
  bams <- list.files(pattern = "\\.bam$")

  # Iterate over each bam and obtain mean coverage of each protein coding gene
  for(i in 1:length(bams)){
    print(paste0("Processing bam ", bams[i]))
    print(paste0("Bam length is: ", bamLengths[i]))
    bamFile <- BamFile(bams[i])

    # Calculate mean coverage
    proteinCodingCov <- protein_coding_genes %>% mutate(meanCov = getCoverage(bams[i],seqid,start,end,"mean",bamLengths[i]))

    # Add mean coverage as a row of the dataframe
    matrixMean[i,] <- proteinCodingCov$meanCov
  }

  # Transform matrix to dataframe and save
  dfMean <- as.data.frame(matrixMean)
  saveRDS(dfMean, path_to_coverage_matrix)
  return(dfMean)
}


#' Function that, given the characteristics of a gene (chromosome, start position and end position),
#' calculates the coverage of each base pair of the gene and applies a normalized function to that coverage.
#'
#' @param bamPath String with the path to the BAM file to use to extract the coverage
#' @param seqid ID of the chromosome of the present gene
#' @param start Start position of the gene in the mouse genome
#' @param end End position of the gene in the mouse genome
#' @param FUN Function to be applied to the coverage
#' @param bamLength Length of the BAM file used to normalize
#'
#' @return The mean normalized coverage for the specified gene
#' @export
#'
#' @examples
getCoverage <- function(bamPath, seqid, start, end, FUN, bamLength){
  library(bamsignals)
  # Change name of Mitochondrial chromosome
  if (seqid == "MT") seqid = "M"
  gr <- GRanges(seqnames = paste0("chr",seqid),
                ranges = IRanges(start = c(start), end = c(end)))
  # Calculate coverage for each gene, given its ranges
  coverage <- bamCoverage(bamPath, gr, verbose=FALSE)
  # Normalize coverage by bam length
  cov_normalized = coverage[1]/bamLength
  return(sapply(list(cov_normalized),FUN))
}


#' Function that loads the region's data to use in each iteration (i.e. promoters, 3' UTR or 5' UTR)
#'
#' @param path_to_gtf_file String with the path to the most recent mouse GTF file
#' @param region_type String with the type of region from which to extract the data
#'
#' @return A dataframe with the main data of the specified region
#' @export
#'
#' @examples
loadRegionData <- function(path_to_gtf_file, region_type){

  # Specify type of UTR that needs to be loaded
  # In the case of promoters, we need to obtain 5' UTRs to extract its 100 previous positions
  if(region_type == "three_prime"){
    utr_type = "three_prime_utr"
  }else{
    utr_type = "five_prime_utr"
  }

  # Load 3 or 5 prime data from GTF File
  # GTF file can be dowloaded from Ensembl: https://www.ensembl.org/info/data/ftp/index.html (This page links to the latest version of GTF ensembl).
  # We use rtracklayer import function to read the gtf file into a proper human readable format
  library(dplyr)
  gtf <- rtracklayer::import(pathToGtfFile)
  gtf.df=as.data.frame(gtf,stringsAsFactor=F)

  # Select only protein-coding genes and transcripts
  gtf.df.pc = gtf.df %>% dplyr::filter(gene_biotype %in% "protein_coding", transcript_biotype %in% "protein_coding")

  # Select TSL level 1
  # TSL coulumn has scoring for the reliability of the transcripts based on multiple evidence. I select only the highly confident transcripts
  gtf.df.pc$transcript_support_level = gsub("\\s*\\([^\\)]+\\)","",as.numeric(gtf.df.pc$transcript_support_level))
  gtf.df.pc.tsl1 = gtf.df.pc %>% dplyr::filter(transcript_support_level %in% 1)

  # extract UTRs
  utr_data = gtf.df.pc.tsl1 %>% dplyr::filter(type %in% utr_type)

  # Collapsing the 5'UTRs among the transcripts for each gene
  # The data in the file is at transcript level rather than gene. So I merge multiple transcripts of a gene into one.
  # This makes a list of Granges objects, with each element is Granges for a gene and contains data about it's transcripts
  utr_grList = makeGRangesListFromDataFrame(utr_data, split.field = "gene_id", names.field = "transcript_id")

  # We merge the transcript coordinates using the reduce function.
  utr_collapse = IRanges::reduce(utr_grList, with.revmap=TRUE) %>% as.data.frame() %>%
    mutate(elements_collapsed = lengths(revmap), prime_utr_id = paste(group_name,seqnames, start, end, strand, elements_collapsed, sep=":"))

  # Only keep UTRs with positive strand
  utr_pos = utr_collapse %>% filter(strand == "+")

  if (region_type == "promoters"){
    promoters = data.frame(utr_pos) # create a copy of the df
    promoters$start = utr_pos$start - 100
    promoters$end = utr_pos$start - 1
    promoters$width = 100
    promoters$prime_utr_id = NULL
    return(promoters)
  }
  return(utr_pos)
}


#' Function that extracts the DNA Sequences from the region's data obtained in the previous step,
#' using the BSGenome library.
#'
#' @param region_set Set that contains the region information from the specified region type
#' @param region_type String with the type of the region from which to extract the DNA sequences
#' @param length_cutoff Integer with the minimum length required for a sequence to be selected for training
#'
#' @return A set with the DNA sequence of the specified region type
#' @export
#'
#' @examples
buildDNASequences <- function(region_set, region_type, length_cutoff = Inf){
  # We have identified one wrong promoter
  if (region_type == "promoters"){
    region_set = region_set[-13455,]
  }
  region_set$seqnames = paste0("chr",region_set$seqnames)

  # Extract sequences using the package BSgenome
  library(BSgenome)
  library(BSgenome.Mmusculus.UCSC.mm10)
  options(stringsAsFactors = FALSE)
  region_ranges = makeGRangesFromDataFrame(region_set)
  dna_sequence = BSgenome::getSeq(Mmusculus, region_ranges, as.character = TRUE)

  # Three and five primes have a cutoff in the minimum length of the sequences, while promoters don't
  # We return two types of sequences: reversed and not reversed
  library(SetupSequences)
  if(region_type != "promoters"){
    region_set = region_set[region_set$width >= length_cutoff,]
    dna_seq_cut = dna_sequence[nchar(dna_sequence) >= length_cutoff]
    dna_seq_reverse <- strReverse(dna_seq_cut)
    dna_seq_reverse <- padding_sequence(dna_seq_reverse, len=length_cutoff)
    dna_seq_normal <- c()
    for (i in 1:length(dna_seq_cut)){
      dna_seq_normal[i] = substr(dna_seq_cut[i],0,length_cutoff)
    }
  }else{
    dna_seq_reverse <- strReverse(dna_sequence)
    dna_seq_normal = dna_sequence
  }

  return(list("final_region_set" = region_set, "reverse" = dna_seq_reverse, "normal" = dna_seq_normal))
}


#' Final function the builds the training dataframe using the information obtained in the previous functions
#'
#' @param region_set Set that contains the region information from the region type of the current iteration
#' @param dna_sequences DNA sequences from the region type of the current iteration
#' @param fold_changes Vector with the fold changes in the ATAC-Seq, which will be the output variable of our model
#' @param df_mean_coverage Dataframe with the mean coverage matrix built in the first step of the script
#' @param output_path String with the output path in which to save the dataframe
#'
#' @return
#' @export
#'
#' @examples
createFinalDataFrame <- function(region_set, dna_sequences, fold_changes, df_mean_coverage, output_path){

  # Initialise dataframe
  data = data.frame(matrix(nr=nrow(region_set), nc=6))
  colnames(data) = c("gene_id", "napps", "dna_seq_cut", "ATAC_basal", "ATAC_1h", "response")

  # Iterate each row in the region set
  for (i in 1:nrow(region_set)){
    # Add ID of the gene relating to that region
    data[i,"gene_id"] = region_set[i, "group_name"]
    # Add number of appearance of that gene in the set
    data[i,"napps"] = nrow(data[data$gene_id == region_set[i, "group_name"] & !is.na(data$gene_id),])
    # Add fold change of the gene, and its ATAC-Seq value initially and after 1 hour of the injection
    data[i, "response"] = fold_changes[,region_set[i, "group_name"]]
    data[i, "ATAC_basal"] = df_mean_coverage[1,region_set[i, "group_name"]]
    data[i, "ATAC_1h"] = df_mean_coverage[2,region_set[i, "group_name"]]
  }

  # Add DNA sequences of each region
  data$dna_seq_cut = dna_sequences
  write.csv(data, output_path)
}
