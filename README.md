# Short text clustering

This repository is based on https://github.com/hadifar/stc_clustering
and is intended for use in a university course project.

To run an experiment, tweak the parameters in data_loader.py and STC.py and run
```commandline
python STC.py --dataset leetcode --save_dir data/leetcode/results/
```

To use GloVe, you also need `glove.42B.300d.txt` to be present in the `data/leetcode`
folder. This file is available from https://nlp.stanford.edu/data/glove.42B.300d.zip

Experiment results:

| Text  | Target clustering | Embedding | Cluster init                       | Num of cluster inits |  NMI  |  ACC  |
|-------|-------------------|-----------|------------------------------------|----------------------|:-----:|:-----:|
| Body  | Easy/Med/Hard     | Word2vec  | Kmeans                             | 100                  | 0.1%  | 45.1% |
| Body  | Easy/Med/Hard     | GloVe     | Kmeans                             | 100                  | 0.9%  | 40.4% |
| Body  | Easy/Med/Hard     | ada       | Kmeans                             | 100                  | 0.6%  | 39.6% |
| Title | Easy/Med/Hard     | Word2vec  | Kmeans                             | 100                  | 0.4%  | 37.3% |
| Title | Easy/Med/Hard     | GloVe     | Kmeans                             | 100                  | 0.3%  | 38.0% |
| Title | Easy/Med/Hard     | ada       | Kmeans                             | 100                  | 0.4%  | 37.2% |
| Body  | Simple topics     | Word2vec  | Kmeans                             | 100                  | 12.6% | 32.3% |
| Body  | Simple topics     | GloVe     | Kmeans                             | 100                  | 8.8%  | 25.9% |
| Body  | Simple topics     | ada       | Kmeans                             | 100                  | 25.4% | 44.9% |
| Title | Simple topics     | Word2vec  | Kmeans                             | 100                  | 7.2%  | 25.8% |
| Title | Simple topics     | GloVe     | Kmeans                             | 100                  | 8.1%  | 27.1% |
| Title | Simple topics     | ada       | Kmeans                             | 100                  | 12.4% | 33.2% |
| Body  | Story             | Word2vec  | Kmeans                             | 100                  | 0.0%  | 54.4% |
| Body  | Story             | GloVe     | Kmeans                             | 100                  | 0.0%  | 58.1% |
| Body  | Story             | ada       | Kmeans                             | 100                  | 0.4%  | 56.8% |
| Title | Story             | Word2vec  | Kmeans                             | 100                  | 0.0%  | 50.1% |
| Title | Story             | GloVe     | Kmeans                             | 100                  | 7.6%  | 66.4% |
| Title | Story             | ada       | Kmeans                             | 100                  | 8.0%  | 66.0% |
| Body  | Difficulty        | ada       | Slight supervision (5 per cluster) | 1                    | 0.6%  | 39.9% |
| Body  | Simple topics     | ada       | Slight supervision (5 per cluster) | 1                    | 27.5% | 44.7% |
| Body  | Story             | ada       | Slight supervision (5 per cluster) | 1                    | 3.7%  | 62.3% |
| Title | Difficulty        | ada       | Slight supervision (5 per cluster) | 1                    | 0.5%  | 38.5% |
| Title | Simple topics     | ada       | Slight supervision (5 per cluster) | 1                    | 14.6% | 36.2% |
| Title | Story             | ada       | Slight supervision (5 per cluster) | 1                    | 8.2%  | 66.1% |

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
| 0     | 705             |
| 1     | 983             |
