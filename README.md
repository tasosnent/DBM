# Deep Beyond MeSH (DBM)
## A Weakly-Supervised Deep-Learning-based method for fine-grained semantic indexing of biomedical literature

Semantic indexing of biomedical literature is usually done at the level of MeSH descriptors, representing topics of interest for the biomedical community. 
Several related but distinct biomedical concepts are often grouped together in a single coarse-grained descriptor and are treated as a single topic for semantic indexing. 
This study [1] proposes a new method  *Deep Beyond MeSH (DBM)* for the automated refinement of subject annotations at the level of concepts, investigating deep learning approaches. 
Lacking labelled data for this task, *DBM* relies on weak supervision based on concept occurrence in the abstract of an article. 
The *DBM* method is evaluated on an extended large-scale retrospective scenario, based on the method *Retrospective Beyond MeSH (RetroBM)*, introduced in the same study [1].
In particular, *RetroBM* takes advantage of concepts that eventually become MeSH descriptors, for which annotations become available in MEDLINE/PubMed. 
An extended version of this study, including additional details, is also available [2]. 
The results suggest that concept occurrence is a strong heuristic for automated subject annotation refinement and can be further enhanced when combined with dictionary-based heuristics. 
In addition, such heuristics can be useful as weak supervision for developing deep learning models that can achieve further improvement in some cases.

![Alt text](https://github.com/tasosnent/DBM/blob/main/Graphical_Abstract.png?raw=true "Overview of the Weakly-Supervised Deep-Learning-based method Deep Beyond MeSH (DBM) for fine-grained semantic indexing of biomedical literature, and the Retrospective Beyond MeSH (RetroBM) method for the development of large-scale retrospective  datasets for this task.")

The implementation of the **Retrospective Beyond MeSH (RetroBM)** method for the development of large-scale retrospective datasets for fine-grained semantic indexing is available [here](https://github.com/ThomasChatzopoulos/MeSH_retrospective_dataset).

This repository includes the implementation of the **Deep Beyond MeSH (DBM)** method for fine-grained semantic indexing of biomedical literature based on weakly-supervised deep learning. Datasets developed with the *RetroBM* methods are required for the large-scale evaluation of the DBM method. 

In particular, this repository includes:
1. *Data preparation*: The [**data**](https://github.com/tasosnent/DBM/tree/main/data) project for processing and transforming the datasets, in a form adequate to be used for model development.
2. *Model development*: The [**modeling**](https://github.com/tasosnent/DBM/tree/main/modeling) project for fine-tuning Deep-Leanring models, producing predictions, and evaluating the performance of these models.
3. *Reporting of results*: The [**reporting**](https://github.com/tasosnent/DBM/tree/main/reporting) project for aggregating predictions from different models and different datasets. 
4. *Logistic regression model development*: The [**lr_modeling**](https://github.com/tasosnent/DBM/tree/main/lr_modeling) project for developing logistic regression models, producing predictions, and evaluating the performance of these models. 

## Requirements
These scripts are written in Python 3.9.7.

Libraries and versions required are listed in requirements.txt.

This project has been developed in PyCharm 2021.1.2 (Community Edition).

## References
[1] Nentidis, A., Chatzopoulos, T., Krithara, A., Tsoumakas, G., & Paliouras, G. (2023). Large-scale investigation of weakly-supervised deep learning for the fine-grained semantic indexing of biomedical literature. Journal of Biomedical Informatics, Volume 146, 2023, 104499, ISSN 1532-0464, https://doi.org/10.1016/j.jbi.2023.104499.

[2] Nentidis, A., Chatzopoulos, T., Krithara, A., Tsoumakas, G., & Paliouras, G. (2023). Large-scale fine-grained semantic indexing of biomedical literature based on weakly-supervised deep learning. arXiv preprint (An extended version of [1]). https://arxiv.org/pdf/2301.09350v1.pdf