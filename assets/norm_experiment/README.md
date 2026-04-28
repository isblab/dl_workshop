## Norm Experiment

#### Setup Environment

```
conda create -f environment.yaml
```

#### Activate Environment
```
conda activate norm_experiment_env
```

#### Run the script
```
python get_norm_data uniqVH.fasta uniqVL.fasta
```
The above script runs "ESM-2(650M)[^esm2]" and "AbLang2[^ablang2]" to calculate embeddings for both VH and VL sequences. \
The IMGT[^imgt] numbering for these sequences is calculated using ANARCI[^anarci].

All the data is stored in ".pt.gz" format which can be easily loaded using gzip and pytorch. 

This particular datafile is necessary for the "Norm Experiment (Exercise4)" in the colab notebook.


[^esm2]: Zeming Lin et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science379, 1123-1130(2023)
[^ablang2]: Tobias H Olsen et al. Addressing the antibody germline bias and its effect on language models for improved antibody design, Bioinformatics, Volume 40, Issue 11, November 2024
[^imgt]: Marie-Paule Lefranc et al. IMGT unique numbering for immunoglobulin and T cell receptor variable domains and Ig superfamily V-like domains, Developmental & Comparative Immunology, Volume 27, Issue 1, 2003,
[^anarci]: James Dunbar et al. ANARCI: antigen receptor numbering and receptor classification, Bioinformatics, Volume 32, Issue 2, January 2016

