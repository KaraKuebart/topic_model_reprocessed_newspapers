
# Article based topic modeling on re-processed historical newspapers
# Code repository
## DHNB 2025
### Kara Kuebart & Christian Schultze & Felix Selgert

This repo contains the code corresponding to the aforementioned paper.
Input data is available here: ...
And results can be found here: ...

The read_data file contains code which can import .csv styled data 
compressed in numpy files from our newspaper pipeline. 
The *reintegrate_text.py* file can be used to transfer raw or preprocessed text from the input data into result files, 
while *export_topic_subdatasets.py* is meant for the creation of topic-specific corpora from a tomotopy results CSV file.
Here, a threshold for the desired minimum relevance of a topic for inclusion in the new .csv file(s) needs to be set. 

To set up the environements to use for topic modeling, please note:
- We use conda (miniforge) for our envs. If you wish to reproduce our results, you may want to do the same. 
In other cases: make sure to adapt the slurmfiles if you are not using conda.
- BERTopic is incompatible with gensim due to version incompatibilities.
- We recommend setting up an environement for bert (we name it "for_bert", if you choose a different name, please adapt slurm files) with rapids.
- Go to https://docs.rapids.ai/install/ , select python 3.10 and the cuda version installed on your graphics cards, then follow the instructions.
- You can install all further requirements using pip and the provided requirements files "requirements_for_bert.txt".
- For gensim, a normal conda env with python=3.10 can be created and all further requirements installed from the "requirements_for_gensim.txt" with pip. 
We name this environement "for_gensim".
- For BERTopic, use for_bert
- For gensim and Leet-Topic, use for_gensim (since Leet-Topic depends on gensim)
- For tomotopy, use any environement. Its sole dependency is numpy.


When using slurmfiles provided here, please remember to change the email contact therein to your own.
