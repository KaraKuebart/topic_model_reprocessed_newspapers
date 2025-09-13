
# Article based topic modeling on re-processed historical newspapers
## DHNB 2025
### Kara Kuebart & Christian Schultze & Felix Selgert

This repo contains the code corresponding to the aforementioned paper.

The read_data file contains code which can import .csv styled data 
compressed in numpy files from our newspaper pipeline. 

To set up the environements used for topic modeling, please note:
- We used conda (miniforge) for our envs. For optimal reproduction of our results, do the same. In other cases: make sure to adapt the slurmfiles if you are not using conda.
- BERTopic is incompatible with gensim due to version incompatibilities.
- We recommend setting up an environement for bert (we name it "for_bert", if you choose a different name, please adapt slurm files) with rapids.
- Go to https://docs.rapids.ai/install/ , select python 3.10 and the cuda version installed on your graphics cards, then follow the instructions.
- You can install all further requirements using pip and the provided requirements files "requirements_for_bert.txt".
- For gensim, a normal conda env with python=3.10 can be created and all further requirements installed from the "requirements_for_gensim.txt" with pip. We name this environement "for_gensim".
- For BERTopic, use for_bert
- For gensim and Leet-Topic, use for_gensim (since Leet-Topic depends on gensim)
- For tomotopy, use any environement. Its sole dependency is numpy.
