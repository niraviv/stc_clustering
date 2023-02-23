# Short text clustering

This repository is based on https://github.com/hadifar/stc_clustering
and is intended for use in a university course project.

To run an experiment, tweak the parameters in data_loader.py and run
```commandline
python STC.py --dataset leetcode --save_dir data/leetcode/results/
```

To use GloVe, you also need `glove.42B.300d.txt` to be present in the `data/leetcode`
folder. This file is available from https://nlp.stanford.edu/data/glove.42B.300d.zip

Experiment results:

| Text  | Target clustering | Embedding | Cluster init | SIF alpha |  NMI  |  ACC  |
|-------|-------------------|-----------|--------------|-----------|:-----:|:-----:|
| Body  | Easy/Med/Hard     | Word2vec  | Kmeans       | 0.1       | 0.2%  | 44.3% |
| Body  | Easy/Med/Hard     | GloVe     | Kmeans       | 0.1       | 0.4%  | 44.6% |
| Body  | Easy/Med/Hard     | ada       | Kmeans       | --        | 0.6%  | 40.2% |
| Title | Easy/Med/Hard     | Word2vec  | Kmeans       | 0.1       | 0.4%  | 36.0% |
| Title | Easy/Med/Hard     | GloVe     | Kmeans       | 0.1       | 0.6%  | 37.5% |
| Title | Easy/Med/Hard     | ada       | Kmeans       | --        | 0.2%  | 36.3% |
| Title | Simple topics     | Word2vec  | Kmeans       | 0.1       | 10.5% | 35.2% |
| Title | Simple topics     | GloVe     | Kmeans       | 0.1       | 23.5% | 47.2% |
| Title | Simple topics     | ada       | Kmeans       | --        | 32.5% | 57.9% |
| Title | Story             | Word2vec  | Kmeans       | 0.1       | 52.2% | 0.1%  |
| Title | Story             | GloVe     | Kmeans       | 0.1       | 64.3% | 9.8%  |
| Title | Story             | ada       | Kmeans       | --        | 70.3% | 13.0% |
| Body  | Story             | Word2vec  | Kmeans       | 0.1       | 53.3% | 0.1%  |
| Body  | Story             | GloVe     | Kmeans       | 0.1       | 50.2% | 1.1%  |
| Body  | Story             | ada       | Kmeans       | --        | 50.1% | 0.8%  |

Dataset statistics:

| Dataset | Number of texts | Average tokens per text | Vocabulary size |
|---------|-----------------|-------------------------|-----------------|
| Body    | 1688            | 479.5                   | 6143            |
| Title   | 1688            | 28.7                    | 1649            |

For the **simple topics** target clustering, the number of texts in each class:

| Class | Number of texts |
|-------|-----------------|
| 0     | 107             |
| 1     | 226             |
| 2     | 154             |
| 3     | 176             |
| 4     | 157             |
| 5     | 868             |

For the **Easy/Med/Hard** target clustering, the number of texts in each class:

| Class | Number of texts |
|-------|-----------------|
| 0     | 427             |
| 1     | 882             |
| 2     | 379             |

For the **story** target clustering, the number of texts in each class:

| Class | Number of texts |
|-------|-----------------|
| 0     | 1032            |
| 1     | 656             |
