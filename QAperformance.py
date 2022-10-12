import numpy as np
from scipy import stats
from ignite.metrics.metric import Metric


class QAPerformance(Metric):
    def reset(self):
        self._yp = []
        self._y  = []

    def update(self, output):
        y_pred, y = output
        self._yp.extend(list(y_pred.to('cpu').numpy()))
        self._y.extend(list(y.to('cpu').numpy()))
        # self._yp.extend([t.item() for t in y_pred])
        # self._y.extend([t.item() for t in y])

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        pq = np.reshape(np.asarray(self._yp), (-1,))

        SROCC = stats.spearmanr(sq, pq)[0]
        KROCC = stats.stats.kendalltau(sq, pq)[0]
        # TODO: nonlinear mapping
        PLCC = stats.pearsonr(sq, pq)[0]
        # RMSE = np.sqrt(np.power(sq-pq, 2).mean())
        return {'SROCC': SROCC,
                'KROCC': KROCC,
                'PLCC': PLCC,
                # 'RMSE': RMSE,
                'sq': sq,
                'pq': pq}
