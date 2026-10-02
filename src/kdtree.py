from sorting import merge_sort

class KDTree:
    def __init__(self, X, leaf_size=10):
        self.X = X
        self.leaf = leaf_size
        self.root = self._build(list(range(len(X))), 0)

    def _build(self, idx, depth):
        # 1. leaf when small, else split at the median of one axis
        if len(idx) <= self.leaf:
            return ("leaf", idx)
        axis = depth % self.X.shape[1]
        idx = merge_sort(idx, key=lambda i: self.X[i, axis])
        mid = len(idx) // 2
        split = self.X[idx[mid], axis]
        return ("node", axis, split,
                self._build(idx[:mid], depth + 1),
                self._build(idx[mid:], depth + 1))

    def query(self, q, k):
        # 2. k nearest as a sorted list of (squared distance, index)
        best = []
        self._search(self.root, q, k, best)
        return best

    def _search(self, node, q, k, best):
        if node[0] == "leaf":
            for i in node[1]:
                d2 = float(((self.X[i] - q) ** 2).sum())
                if len(best) < k or d2 < best[-1][0]:
                    # 3. insert keeping the list sorted, k is small
                    best.append((d2, i))
                    j = len(best) - 1
                    while j > 0 and best[j][0] < best[j - 1][0]:
                        best[j], best[j - 1] = best[j - 1], best[j]
                        j -= 1
                    if len(best) > k:
                        best.pop()
            return
        _, axis, split, left, right = node
        near, far = (left, right) if q[axis] < split else (right, left)
        self._search(near, q, k, best)
        # 4. visit the far side only if the split plane is closer than the k-th best
        if len(best) < k or (q[axis] - split) ** 2 < best[-1][0]:
            self._search(far, q, k, best)