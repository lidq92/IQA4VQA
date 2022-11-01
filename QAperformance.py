import numpy as np
from ignite.metrics.metric import Metric
from scipy.stats import pearsonr, spearmanr, kendalltau


class QAPerformance(Metric):
    def reset(self):
        self._yp = []
        self._y  = []

    def update(self, output):
        y_pred, y = output
        self._yp.extend(list(y_pred.to('cpu').numpy()))
        self._y.extend(list(y.to('cpu').numpy()))

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        pq = np.reshape(np.asarray(self._yp), (-1,))

        SROCC = spearmanr(sq, pq)[0]
        KROCC = kendalltau(sq, pq)[0]
        # TODO: 4-parameter logistic nonlinear mapping (for cross-dataset evaluation)
        PLCC = pearsonr(sq, pq)[0]
        # RMSE = np.sqrt(np.power(sq-pq, 2).mean())
        return {'SROCC': SROCC,
                'KROCC': KROCC,
                'PLCC': PLCC,
                # 'RMSE': RMSE,
                'sq': sq,
                'pq': pq}
