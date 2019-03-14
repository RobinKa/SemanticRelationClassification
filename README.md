# Onto.KOM
Python 3 package for relation extraction and classification. Created as a thesis project for my bachelor studies at TU Darmstadt.

# Links
[Slides](https://drive.google.com/open?id=1eVW_zcqQVsGDJP_MgZyvowbQdNGERjxo)
[Thesis](https://drive.google.com/open?id=1ToX-X4TGBYg9ZxXRDymAOWsRZ7EvIHaw)

# Installation
`pip install .` to install the package

# Requirements
Python 3 packages
- numpy
- pandas
- nltk
- textblob
- sklearn
- keras
- tqdm

For embedding creation one of the following are required
- fasttext (https://github.com/facebookresearch/fastText or https://github.com/salestock/fastText.py on Windows)
- GloVe binaries (https://github.com/stanfordnlp/GloVe)

# Examples
Usage examples can be found in the examples folder

# API Overview

## preprocessing.py
```
download_preprocessing_prerequisites()
text_blob_from_file(file_path)
remove_stop_words(text_blob)
link_noun_phrases(text_blob)
```

## embeddings.py
```
class WordEmbeddings
    load()
    embedding_for(word)
    words
    as_dict()
    as_key_values()

class GloVeEmbeddings(path) # For creating GloVe vectors
    train(self, corpus_path, epochs=30, embedding_size=300,
          context_size=15, min_occurrences=5, glove_path=".",
          threads=8, keep_csv=False)

class FastTextEmbeddings(path) # For FastText models
    train(self, input_file, **kwargs)

class DataFrameEmbeddings(path) # For loading csv files

def create_relation_dataset(embeddings, out_relations_path, out_labels_path,
                            relation_paths, out_false_relations_path=None, max_per_class=None,
                            unknown_word=None)
```

## relationextraction.py
```
extract_concept_net(file_path, out_path, chunk_size=1000)
extract_yago(file_path, out_path, chunk_size=1000)
extract_relations_hearst(corpus_path, stop_words_path)
```

## classification.py
```
class RelationClassifier
    new(self, input_dim, relation_count, one_hot=False, filters=32, max_filters=128,
        subtract_embeddings=False, dropout=False, learn_rate=0.001, optimizer="rmsprop",
        kernel_size=3, lr_decay=0)
    save(self, path)
    load(self, path)
    train(self, relations, labels, batch_size=256, validation_split=0.1, epochs=10,
          val_data=None, verbose=1)
    predict(self, relations)
```

## visualization.py
```
show_embeddings_tsne(embeddings, word_count=1000, size=(100, 100), save_path=None,
                     clusters=None, **tsne_args)
```

## clustering.py
```
class EmbeddingClusterer
    cluster(self, embeddings, min_clusters=5, max_clusters=100)
```
