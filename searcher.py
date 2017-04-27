import numpy as np

class Searcher:
    def __init__(self, indexStore):
        self.indexStore = indexStore

    def chi2Distance(self, histA, histB, eps=1e-10):
        return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    def search(self, queryFeatures):
        results = {}

        for (photoName, features) in self.indexStore.items():
            results[photoName] = self.chi2Distance(features, queryFeatures)

        results = sorted([(score, photoName) for (photoName, score) in results.items()])
        return results
