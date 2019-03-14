from sklearn.metrics import calinski_harabaz_score, silhouette_score
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class EmbeddingClusterer:
    """Clusters embedding vectors into a number of clusters determined
    by different metrics."""
    @staticmethod
    def _metric_scores(emb, clusters):
        yield calinski_harabaz_score(emb, clusters)
        yield silhouette_score(emb, clusters, sample_size=10000)

    @staticmethod
    def _get_clusters(emb, num_clusters):
        # Bug in sklearn, sometimes raises IndexError
        # https://github.com/scikit-learn/scikit-learn/issues/7705
        try:
            return MiniBatchKMeans(n_clusters=num_clusters, max_iter=10000).fit_predict(emb)
        except:
            return MiniBatchKMeans(n_clusters=num_clusters, max_iter=10000).fit_predict(emb)

    def cluster(self, embeddings, min_clusters=5, max_clusters=100):
        """Trains a new clustering model for `embeddings`"""
        print("Calculating clusters")
        cluster_scores = []

        for num_clusters in tqdm(range(min_clusters, max_clusters + 1)):
            clusters = EmbeddingClusterer._get_clusters(
                embeddings, num_clusters)
            for metric_id, score in enumerate(EmbeddingClusterer._metric_scores(embeddings,
                                                                                clusters)):
                if num_clusters == min_clusters:
                    cluster_scores.append([])
                cluster_scores[metric_id].append(score)

        cluster_scores = np.array(cluster_scores)
        num_metrics = len(cluster_scores)

        np.savetxt("clusterscores.txt", cluster_scores)

        print("Cluster scores")
        for num_clusters, scores in enumerate(cluster_scores.T):
            print("\t", num_clusters + min_clusters, scores)

        # (num_clusters, num_metrics)
        sorted_num_clusters = np.argsort(-cluster_scores, axis=1).T

        num_votes = np.zeros(max_clusters - min_clusters, dtype=np.int32)
        for voted_clusters in sorted_num_clusters:
            num_votes[voted_clusters] += 1
            best_num_clusters = np.argwhere(num_votes == num_metrics)
            if len(best_num_clusters) > 0:
                best_num_clusters = best_num_clusters[0][0] + min_clusters
                break

        print("Best:", best_num_clusters)

        clusters = EmbeddingClusterer._get_clusters(
            embeddings, best_num_clusters)

        return clusters
