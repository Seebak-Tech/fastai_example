class _FastaiClassifierModelWrapper:
    def __init__(self, learner):
        self.learner = learner

    def predict(self, data):
        import pandas as pd

        dl = self.learner.dls.test_dl(data)
        #  dl.num_workers = 0
        preds, _ = self.learner.get_preds(dl=dl)

        # converting probabilities to classes
        class_idxs = preds.argmax(axis=1)
        results = [(c.numpy(), self.learner.dls.vocab[c]) for c in class_idxs]

        return pd.DataFrame(results, columns=["class", "label"])


def _load_pyfunc(path):
    import os
    from fastai.learner import load_learner

    learn = load_learner(os.path.abspath(path))

    return _FastaiClassifierModelWrapper(learn)
