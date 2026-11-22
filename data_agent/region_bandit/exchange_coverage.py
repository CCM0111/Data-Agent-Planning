"""Exact sparse removal losses and reversible coverage updates for local swaps."""
import numpy as np
from ..marginal_coverage import GroupCoverageState


class ExchangeCoverageState(GroupCoverageState):
    def removal_losses(self):
        """F(S)-F(S\{i}) for every selected i; tied representatives lose zero."""
        selected = np.flatnonzero(self.used)
        losses = np.zeros(len(self.ids))
        if not len(selected):
            return losses
        edges = self.K[:, selected].tocoo()
        first = np.zeros(len(self.ids))
        np.maximum.at(first, edges.row, edges.data)
        is_top = edges.data == first[edges.row]
        top_counts = np.bincount(edges.row[is_top], minlength=len(self.ids))
        second = np.zeros(len(self.ids))
        np.maximum.at(second, edges.row[~is_top], edges.data[~is_top])
        unique = is_top & (top_counts[edges.row] == 1)
        rows = edges.row[unique]
        np.add.at(losses, selected[edges.col[unique]], self.a[rows]*(first[rows]-second[rows]))
        return losses

    def remove(self, i):
        if not self.used[i]:
            raise ValueError('Cannot remove an unselected sample')
        self.used[i] = False
        lo, hi = self.cols.indptr[i:i+2]
        affected = self.cols.indices[lo:hi]
        new = (self.K[affected][:, self.used].max(axis=1).toarray().ravel()
               if self.used.any() else np.zeros(len(affected)))
        for j, coverage in zip(affected, new):
            old = self.coverage[j]
            if coverage == old:
                continue
            start, stop = self.K.indptr[j:j+2]
            neighbors, values = self.K.indices[start:stop], self.K.data[start:stop]
            self.gain[neighbors] += self.a[j]*(np.maximum(values-coverage, 0.)-np.maximum(values-old, 0.))
            self.coverage[j] = coverage
