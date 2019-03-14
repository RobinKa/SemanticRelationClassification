import os
from os import remove, system
from os.path import join, exists, abspath
from collections import OrderedDict
from itertools import islice
import pandas as pd
import numpy as np
from tqdm import tqdm
from .util import load_corpus, write_hdf_from_dict, write_hdf_chunked_from_dict


class WordEmbeddings:
    def __init__(self, path):
        self.path = path

    def embedding_for(self, word):
        """Returns the embedding vector for `word`"""
        pass

    @property
    def words(self):
        """Returns all the words available"""
        return []

    def as_dict(self):
        """Returns the word embeddings as a dictionary word -> embedding vector"""
        return {word: self.embedding_for(word) for word in self.words}

    def as_key_values(self):
        embeddings_dict = self.as_dict()
        return embeddings_dict.keys(), embeddings_dict.values()

    def load(self):
        pass


class FastTextEmbeddings(WordEmbeddings):
    def __init__(self, path):
        super().__init__(path)
        self.model = None
        import fasttext as ft
        self.ft = ft

    def load(self):
        """Loads a FastText model"""
        self.model = self.ft.load_model(self.path)

    def train(self, input_file, **kwargs):
        """Trains a FastText model using a text file at `input_file` and parameters
        in `kwargs` and saves  the model to `model_path`"""
        self.model = self.ft.skipgram(input_file, self.path, **kwargs)

    def embedding_for(self, word):
        """Returns the embedding vector for `word`. `word` is allowed to be
        out of vocabulary."""
        return self.model[word]

    @property
    def words(self):
        """Returns all the words available"""
        return self.model.words


class GloVeEmbeddings(WordEmbeddings):
    def __init__(self, path):
        super().__init__(path)

    def train(self, corpus_path, epochs=30, embedding_size=300,
              context_size=15, min_occurrences=5, glove_path=".",
              threads=8, keep_csv=False):
        """Trains a new glove word embedding given `corpus_path`"""

        if not exists(corpus_path):
            raise ValueError("No corpus found at %s" % corpus_path)

        path_vocab_count = abspath(join(glove_path, "vocab_count"))
        path_cooccur = abspath(join(glove_path, "cooccur"))
        path_shuffle = abspath(join(glove_path, "shuffle"))
        path_glove = abspath(join(glove_path, "glove"))

        # Append .exe to the paths on windows
        if os.name == "nt":
            windows_ext = ".exe"
            path_vocab_count += windows_ext
            path_cooccur += windows_ext
            path_shuffle += windows_ext
            path_glove += windows_ext

        if (not exists(path_vocab_count) or not exists(path_cooccur) or not exists(path_shuffle) or
                not exists(path_glove)):
            raise ValueError(
                "One or more GloVe files are missing from directory %s" % glove_path)

        aux_vocab = "vocab.txt"
        aux_cooccurence = "cooccurence.bin"
        aux_shuffled = "shuffled.bin"

        print("Corpus path:", corpus_path)

        try:
            # Run the GloVe binaries
            # TODO: Probably better to use subprocess.call instead of os.system, but it caused
            #       some issues using different platforms

            def _run_cmd(cmd):
                print("Running", cmd)
                system(cmd)

            cmd_vocab_count = '%s -min-count %d -verbose 2 < "%s" > "%s"' % (
                path_vocab_count, min_occurrences, corpus_path, aux_vocab
            )

            cmd_cooccur = '%s -memory 8.0 -vocab-file "%s" -verbose 2 -window-size %d < "%s" > "%s"' % (
                path_cooccur, aux_vocab, context_size, corpus_path, aux_cooccurence
            )

            cmd_shuffle = '%s -memory 8.0 -verbose 2 < "%s" > "%s"' % (
                path_shuffle, aux_cooccurence, aux_shuffled
            )

            cmd_glove = '%s -save-file "%s" -threads %d -input-file "%s" -x-max 100 -iter %d -vector-size %d -binary 0 -vocab-file "%s" -verbose 2' % (
                path_glove, self.path, threads, aux_shuffled, epochs, embedding_size, aux_vocab
            )

            _run_cmd(cmd_vocab_count)
            _run_cmd(cmd_cooccur)
            _run_cmd(cmd_shuffle)
            _run_cmd(cmd_glove)

            result_path = self.path + ".txt"

            print("Done creating embeddings at", result_path)

            print("Converting csv to hdf5")
            dtypes = {}
            dtypes[0] = str
            for i in range(embedding_size):
                dtypes[i + 1] = np.float32

            data_frame = pd.read_csv(result_path, sep=" ", skiprows=0, header=None,
                                     usecols=range(embedding_size + 1), index_col=0, dtype=dtypes)
            data_frame.to_hdf(self.path + ".h5", "embeddings", mode="w")

            if not keep_csv:
                if exists(result_path):
                    print("Removing csv from", result_path,
                          "(specify keep_csv=1 if you want to keep it)")
                    remove(result_path)
        except:
            print("Failed to create embeddings")

        print("Removing auxiliary files")
        if exists(aux_vocab):
            remove(aux_vocab)
        if exists(aux_cooccurence):
            remove(aux_cooccurence)
        if exists(aux_shuffled):
            remove(aux_shuffled)


class DataFrameEmbeddings(WordEmbeddings):
    """Holds embedding vectors indexed by string keys using a pandas DataFrame."""

    def __init__(self, path):
        """Initializes the DataFrameEmbeddings given a data_frame.
        The data frame is expected to use words as index and the embedding
        vector as columns.
        """
        super().__init__(path)
        self.data_frame = None

    def embedding_for(self, word):
        """Returns the embedding vector for `word`. `word` must be in the vocabulary."""
        return self.data_frame.loc[word].values

    @property
    def words(self):
        """Returns all the words available"""
        return self.data_frame.index.values

    def load(self):
        self.data_frame = pd.read_hdf(self.path)

    def as_dict(self):
        return self.data_frame.T.to_dict(orient="list")

    def as_key_values(self):
        return self.data_frame.index.values, self.data_frame.values


def create_relation_dataset(embeddings, out_relations_path, out_labels_path,
                            relation_paths, out_false_relations_path=None, max_per_class=None,
                            unknown_word=None):
    """Creates relation embeddings and their labels from word vectors in `embeddings`
    for relations found in `relation_paths` at `out_relations_path` and `out_labels_path`
    respectively. Also creates the same amount of false relations at
    `out_false_relations_path`."""
    relation_count = len(relation_paths)

    relation_embeddings = OrderedDict()
    relation_labels = OrderedDict()

    total_relations = 0
    invalid_relations = 0  # TODO / bug: words starting with "nan" parsed as float by pandas
    unavailable_relations = 0

    def get_relation_embedding(word_a, word_b):
        """
        emb_a = []
        emb_b = []

        for word in word_a.split("_"):
            emb_a.append(embeddings.embedding_for(word))
        emb_a = np.mean(np.array(emb_a), axis=0)

        for word in word_b.split("_"):
            emb_b.append(embeddings.embedding_for(word))
        emb_b = np.mean(np.array(emb_b), axis=0)
        """

        # Use unknown_word if word could not be found
        try:
            emb_a = embeddings.embedding_for(word_a)
        except:
            if unknown_word is not None:
                emb_a = embeddings.embedding_for(unknown_word)
            else:
                emb_a = None

        try:
            emb_b = embeddings.embedding_for(word_b)
        except:
            if unknown_word is not None:
                emb_b = embeddings.embedding_for(unknown_word)
            else:
                emb_b = None

        if emb_a is None or emb_b is None:
            return None

        return np.hstack((emb_a, emb_b))

    for relation_id, relation_path in enumerate(relation_paths):
        print("Processing relations at %s" % relation_path)
        pairs = pd.read_csv(relation_path, sep=" ", header=None, usecols=[0, 1],
                            dtype=str, encoding="utf-8")

        class_relations = 0 # How many relations of this class have been added
        for row in tqdm(pairs.itertuples()):
            total_relations += 1

            relation_tuple = (row[1], row[2])

            if not isinstance(relation_tuple[0], str) or not isinstance(relation_tuple[1], str):
                #print("Skipping pair", relation_tuple)
                invalid_relations += 1
                continue

            if relation_tuple not in relation_embeddings:
                # Use unknown_word if word could not be found
                relation_embedding = get_relation_embedding(relation_tuple[0], relation_tuple[1])

                if relation_embedding is None:
                    unavailable_relations += 1
                    continue

                if np.isnan(relation_embedding).any():
                    invalid_relations += 1
                    continue

                relation_embeddings[relation_tuple] = relation_embedding
                relation_labels[relation_tuple] = np.zeros(relation_count)
                class_relations += 1
                if max_per_class is not None and class_relations >= max_per_class:
                    break

            relation_labels[relation_tuple][relation_id] = 1

    found_relations = total_relations - invalid_relations - unavailable_relations

    print("-- Relations statistics")
    print("\t", total_relations, "total relations")
    print("\t", found_relations, "found relations (%.2f%%)" %
          (100 * found_relations / total_relations))
    print("\t", invalid_relations, "invalid relations (%.2f%%)" %
          (100 * invalid_relations / total_relations))
    print("\t", unavailable_relations, "unavailable relations (%.2f%%)" %
          (100 * unavailable_relations / total_relations))

    '''
    print("Creating relation tree")
    relation_tree = {}
    for relation in tqdm(relation_embeddings):
        relation_tree.setdefault(relation[0], []).append(relation[1])

    def tree_find(tree, root_word, word_to_find):
        """Returns whether `tree` contains `word_to_find`"""
        checked_tree_words = []
        tree_words_to_check = [root_word]

        while tree_words_to_check:
            current_words_to_check = tree_words_to_check
            tree_words_to_check = []

            # Remember the words we already checked
            checked_tree_words += current_words_to_check

            for word in current_words_to_check:
                # Return true if the word was found in the tree
                if word == word_to_find:
                    return True

                # Add all unchecked words to the list of words to check
                if word in tree:
                    tree_words_to_check += [other_word for other_word in tree[word]
                                            if other_word not in checked_tree_words and
                                            other_word not in tree_words_to_check]

        # No matches found
        return False
    '''

    def are_related(word_a, word_b):
        """Returns whether `word_a` <relation> `word_b`"""
        #return tree_find(relation_tree, word_a, word_b)
        return word_a == word_b or (word_a, word_b) in relation_embeddings

    if out_false_relations_path:
        print("Creating false relations")
        false_relation_embeddings = {}

        all_words = []
        for relation in tqdm(relation_embeddings):
            all_words.append(relation[0])
            all_words.append(relation[1])
        all_words = list(set(all_words))
        word_count = len(all_words)

        for relation_tuple in tqdm(relation_embeddings):
            left_word, right_word = relation_tuple
            left_false_rel = None
            
            # Make left fake
            while not left_false_rel or left_false_rel in false_relation_embeddings or are_related(left_false_rel[0], left_false_rel[1]):
                left_false_rel = (left_word, all_words[np.random.randint(word_count)])

            left_false_emb = get_relation_embedding(left_false_rel[0], left_false_rel[1])
            false_relation_embeddings[left_false_rel] = left_false_emb
            
            # Make right fake
            right_false_rel = None
            while not right_false_rel or right_false_rel in false_relation_embeddings or are_related(right_false_rel[0], right_false_rel[1]):
                right_false_rel = (all_words[np.random.randint(word_count)], right_word)

            right_false_emb = get_relation_embedding(right_false_rel[0], right_false_rel[1])
            false_relation_embeddings[right_false_rel] = right_false_emb

        # Write false relation embeddings
        print("Saving false relations to", out_false_relations_path)
        write_hdf_chunked_from_dict(
            false_relation_embeddings, out_false_relations_path, "embeddings")
    """
    print("Creating false relations using random embedding words")

    # Create fake pair using words from all embedding words
    all_words = list(embeddings.words)
    word_count = len(all_words)
    for _ in tqdm(range(len(relation_embeddings)//2)):
        while True:
            relation_tuple = (all_words[np.random.randint(word_count)],
                              all_words[np.random.randint(word_count)])
            if (relation_tuple[0] != relation_tuple[1] and
                    relation_tuple not in relation_embeddings and
                    relation_tuple not in false_relation_embeddings):
                break

        relation_embedding = np.hstack((
            embeddings.embedding_for(relation_tuple[0]),
            embeddings.embedding_for(relation_tuple[1])))
        false_relation_embeddings[relation_tuple] = relation_embedding
    """

    """
    print("Creating false relations using random relation words")

    # Create fake pair using words from real word pairs
    all_words = []
    for relation in tqdm(relation_embeddings.keys()):
        all_words.append(relation[0])
        all_words.append(relation[1])
    all_words = list(set(all_words))
    word_count = len(all_words)
    for _ in tqdm(range(len(relation_embeddings))):
        while True:
            relation_tuple = (all_words[np.random.randint(word_count)],
                              all_words[np.random.randint(word_count)])
            if (relation_tuple[0] != relation_tuple[1] and
                    relation_tuple not in relation_embeddings and
                    relation_tuple not in false_relation_embeddings):
                break

        relation_embedding = np.hstack((
            embeddings.embedding_for(relation_tuple[0]),
            embeddings.embedding_for(relation_tuple[1])))
        false_relation_embeddings[relation_tuple] = relation_embedding
    """

    # Write the embeddings in chunks becauce they might not fit into memory
    # twice
    print("Relation embeddings count:", len(relation_embeddings))
    print("Saving embeddings to", out_relations_path)
    write_hdf_chunked_from_dict(
        relation_embeddings, out_relations_path, "embeddings")

    # Write the labels all at once
    print("Saving labels to", out_labels_path)
    write_hdf_from_dict(relation_labels, out_labels_path, "labels")

    
