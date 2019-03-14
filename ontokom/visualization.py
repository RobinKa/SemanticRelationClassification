from itertools import chain
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def get_tsne(word_embeddings, perplexity=30, n_iter=5000, learning_rate=100, verbose=1):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                learning_rate=learning_rate, verbose=verbose)
    embeddings_tsne = tsne.fit_transform(word_embeddings)
    return embeddings_tsne


def show_embeddings_tsne(embeddings, word_count=1000, size=(100, 100), save_path=None,
                         clusters=None, **tsne_args):
    if clusters is not None:
        cluster_indices = set(clusters[0])
        words_per_cluster = word_count // len(cluster_indices)
        words = list(chain(*(clusters.loc[clusters[0] == cluster_index].head(
            words_per_cluster).index.tolist() for cluster_index in cluster_indices)))
    else:
        words = list(embeddings.words)[:word_count]

    word_embeddings = [embeddings.embedding_for(word) for word in words]

    print("Training TSNE")
    embeddings_tsne = get_tsne(word_embeddings, **tsne_args)

    print("Creating figure")
    plt.figure(figsize=size)

    print("Plotting")
    if clusters is not None:
        colors = cm.rainbow(np.linspace(0, 1, len(cluster_indices)))
        cluster_colors = {cluster_index: color
                          for cluster_index, color in zip(cluster_indices, colors)}
        word_clusters = clusters[0].loc[words].values

        for word, embedding, word_cluster in zip(words, embeddings_tsne, word_clusters):
            plt.scatter(*embedding, c=cluster_colors[word_cluster])
            plt.annotate(word, xy=embedding, xytext=(5, 2), textcoords="offset points",
                         ha="right", va="bottom")
    else:
        for word, embedding in zip(words, embeddings_tsne):
            plt.scatter(*embedding)
            plt.annotate(word, xy=embedding, xytext=(5, 2), textcoords="offset points",
                         ha="right", va="bottom")

    if save_path is not None:
        print("Saving figure")
        plt.savefig(save_path)
    else:
        print("Showing figure")
        plt.show()
