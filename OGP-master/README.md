# *O*-GlcNAcylation sites prediction #
Related to the article "*O*-GlcNAcylation Prediction: An Unattained Objective"

*Author: Théo MAURI et al* - 2021


## Datasets ##
The full list of proteins with Uniprot ID, *O*-GlcNAcylated sites index and sequence are listed in *all\_sites.csv* and *all\_sites\_with\_colnames.csv* files.

The full list of sites with features can be found in two separate files, depending if they are positive or negative:
- file\_for\_ML\_pos.csv for the positive sites (=*O*-GlcNAcylated)
- file\_for\_ML\_neg.csv for the negative sites

where
- class is if a site is really *O*-GlcNAcylated (1) or not (2)
- nb_ST is the number of serines/threonines in a window of +/- 10 around the site
- cpt_ali is the amount of aliphatic residues from positions -3 to -1: 0, 1, 2 or 3
- cpt_pos is the amount of polar positively charged residues from positions -7 to -5: 0, 1, 2 or 3
- Pro_1 is the presence of a proline after the site: 0 or 1
- min1 is the side Chain length class: 0, 1, 2, 3, 4, 5, 6 or 7 where 0 is No Residue, 1 is Glycine, 2 Very Small, 3 Small, 4 Normal, 5 Long, 6 Cycle and 7 Proline at position -1
- plus1 is the side Chain length class: 0, 1, 2, 3, 4, 5, 6 or 7 where 0 is No Residue, 1 is Glycine, 2 Very Small, 3 Small, 4 Normal, 5 Long, 6 Cycle and 7 Proline at position +1
- plus2 is the side Chain length class: 0, 1, 2, 3, 4, 5, 6 or 7 where 0 is No Residue, 1 is Glycine, 2 Very Small, 3 Small, 4 Normal, 5 Long, 6 Cycle and 7 Proline at position +2
- plus3 is the side Chain length class: 0, 1, 2, 3, 4, 5, 6 or 7 where 0 is No Residue, 1 is Glycine, 2 Very Small, 3 Small, 4 Normal, 5 Long, 6 Cycle and 7 Proline at position +3
- plus4 is the side Chain length class: 0, 1, 2, 3, 4, 5, 6 or 7 where 0 is No Residue, 1 is Glycine, 2 Very Small, 3 Small, 4 Normal, 5 Long, 6 Cycle and 7 Proline at position +4
- plus5 is the side Chain length class: 0, 1, 2, 3, 4, 5, 6 or 7 where 0 is No Residue, 1 is Glycine, 2 Very Small, 3 Small, 4 Normal, 5 Long, 6 Cycle and 7 Proline at position +5
- flexibility is a continuous value from 0 to 1 where 0 is flexible and 1 rigid
- naturesite is the nature of site: 0 or 1 where 0 is serine and 1 threonine
- ss is the secondary structure 0, 1 or 2 where 0 is not structured, 1 is alpha helix and 2 is beta strand
- phi_psi is the secondary structure according to phi and psi angles: 0, 1 or 2

## Machine Learning ##

### Retrieve features ###
The *script\_to\_retrieve\_all\_the\_features\_for\_ML.py* Python (V3.6) script is available to retrieve the features from the dataset. It takes as input the two all\_sites.csv files (already in the script) and creates *file\_for\_ML.csv*. 

It also needs to be run where the two folders spider3\_results and dynamine\_results are located.


To run the Python script you need to install xlrd and xlwt by typing this command:

python3 -m pip install xlrd

python3 -m pip install xlwt


To run it please type:

python3 script\_to\_retrieve\_all\_the\_features\_for\_ML.py

### Training and testing ###

The *all\_methods\_x\_runs.R* R script creates training and testing datasets, trains the different models, tests them and retrieves the sensitivity and PPV. (R version is 3.6.3)

This file takes as input *file\_for\_ML\_pos.csv* and *file\_for\_ML\_neg.csv* which were obtained spliting *file\_for\_ML.csv* according to the 2 classes.


GBT algorithm is CPU consuming. To test the script, we recommend to keep the *GBT == FALSE* at the beginning of the script. Otherwise, You can chage it to *GBT == TRUE*.
The script runs 10 runs in parallel (the number of parallel runs can be modified at the beginning of the script).

**To be able to run the script, several libraries must be installed:**
- parallel
- dplyr
- readr
- randomForest
- smotefamily
- ROSE
- caret
- e1071
- xgboost

To run the script please type:

Rscript --vanilla *all\_methods\_x\_runs.R*

The output will be models generated, and the table with the model performances accordind to testing dataset. 

The number of runs in parallel will change the name of these files.

## Extra data ##
1 Directory contains the flexibility predictions (DynaMine):

backbone.pred files are result files where the 11 first line correspond to the header with references then a line per residue with the flexibility score.

1 Directory contains the secondary structure predictions (SPIDER3):

.spd33 files are result files where each line corresponds to an amino acid with the phi\/psi angles and secondary structures; results.txt contains all the protein sequences and associated predictions.

1 Directory contains examples of Machine Learning models (which are R objects)


## Evaluation of available prediction tools ##
To test the currently available prediction tools of *O*-GlcNAcylated sites, sequences of the dataset have been copied and pasted to the following websites:
- https://services.healthtech.dtu.dk/service.php?YinOYang-1.2
- http://csb.cse.yzu.edu.tw/OGTSite/
- http://server.malab.cn/OGlcPred/

