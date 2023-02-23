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

| Text  | Target clustering | Embedding | Cluster init       | Num of cluster inits | NMI  |  ACC  |
|-------|-------------------|-----------|--------------------|----------------------|:----:|:-----:|
| Body  | Easy/Med/Hard     | Word2vec  | Kmeans             | 100                  | 0.2% | 44.3% |
| Body  | Easy/Med/Hard     | GloVe     | Kmeans             | 100                  | 0.4% | 44.6% |
| Body  | Easy/Med/Hard     | ada       | Kmeans             | 100                  | 0.6% | 40.2% |
| Title | Easy/Med/Hard     | Word2vec  | Kmeans             | 100                  | 0.4% | 36.0% |
| Title | Easy/Med/Hard     | GloVe     | Kmeans             | 100                  | 0.6% | 37.5% |
| Title | Easy/Med/Hard     | ada       | Kmeans             | 100                  | 0.2% | 36.3% |
| Body  | Simple topics     | Word2vec  | Kmeans             | 100                  |      |       |
| Body  | Simple topics     | GloVe     | Kmeans             | 100                  |      |       |
| Body  | Simple topics     | ada       | Kmeans             | 100                  |      |       |
| Title | Simple topics     | Word2vec  | Kmeans             | 100                  |      |       |
| Title | Simple topics     | GloVe     | Kmeans             | 100                  |      |       |
| Title | Simple topics     | ada       | Kmeans             | 100                  |      |       |
| Body  | Story             | Word2vec  | Kmeans             | 100                  |      |       |
| Body  | Story             | GloVe     | Kmeans             | 100                  |      |       |
| Body  | Story             | ada       | Kmeans             | 100                  |      |       |
| Title | Story             | Word2vec  | Kmeans             | 100                  |      |       |
| Title | Story             | GloVe     | Kmeans             | 100                  |      |       |
| Title | Story             | ada       | Kmeans             | 100                  |      |       |
| Title | Simple topics     | Word2vec  | Slight supervision | 3                    |      |       |
| Title | Simple topics     | GloVe     | Slight supervision | 3                    |      |       |
| Title | Simple topics     | ada       | Slight supervision | 3                    |      |       |
| Title | Story             | Word2vec  | Slight supervision | 3                    |      |       |
| Title | Story             | GloVe     | Slight supervision | 3                    |      |       |
| Title | Story             | ada       | Slight supervision | 3                    |      |       |

Dataset statistics:

| Dataset | Number of texts | Average tokens per text | Vocabulary size |
|---------|-----------------|-------------------------|-----------------|
| Body    | 1688            | 479.5                   | 6143            |
| Title   | 1688            | 28.7                    | 1649            |

For the **simple topics** target clustering, the number of texts in each class:

| Class | Number of texts |
|-------|-----------------|
| 0     | 201             |
| 1     | 145             |
| 2     | 154             |
| 3     | 650             |
| 4     | 116             |
| 5     | 193             |
| 6     | 229             |

For the **Easy/Med/Hard** target clustering, the number of texts in each class:

| Class | Number of texts |
|-------|-----------------|
| 0     | 427             |
| 1     | 882             |
| 2     | 379             |

For the **story** target clustering, the number of texts in each class:

| Class | Number of texts |
|-------|-----------------|
| 0     | 1688            |
| 1     | 705             |
