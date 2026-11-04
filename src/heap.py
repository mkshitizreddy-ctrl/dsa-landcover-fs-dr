class IndexedMaxHeap:
    def __init__(self, n):
        self.heap = []
        self.pos = [-1] * n
        self.key = [0.0] * n

    def __len__(self):
        return len(self.heap)

    def __contains__(self, i):
        return self.pos[i] != -1

    def peek(self):
        top = self.heap[0]
        return top, self.key[top]

    def _swap(self, a, b):
        h = self.heap
        h[a], h[b] = h[b], h[a]
        self.pos[h[a]] = a
        self.pos[h[b]] = b

    def _up(self, k):
        while k > 0:
            parent = (k - 1) // 2
            if self.key[self.heap[k]] <= self.key[self.heap[parent]]:
                break
            self._swap(k, parent)
            k = parent

    def _down(self, k):
        n = len(self.heap)
        while True:
            left, right, largest = 2 * k + 1, 2 * k + 2, k
            if left < n and self.key[self.heap[left]] > self.key[self.heap[largest]]:
                largest = left
            if right < n and self.key[self.heap[right]] > self.key[self.heap[largest]]:
                largest = right
            if largest == k:
                break
            self._swap(k, largest)
            k = largest

    def push(self, i, key):
        if self.pos[i] != -1:
            raise ValueError("id already in heap")
        self.key[i] = key
        self.heap.append(i)
        self.pos[i] = len(self.heap) - 1
        self._up(len(self.heap) - 1)

    def pop(self):
        top = self.heap[0]
        key = self.key[top]
        self._swap(0, len(self.heap) - 1)
        self.heap.pop()
        self.pos[top] = -1
        if self.heap:
            self._down(0)
        return top, key

    def update(self, i, key):
        old = self.key[i]
        self.key[i] = key
        if key > old:
            self._up(self.pos[i])
        else:
            self._down(self.pos[i])

    @classmethod
    def build(cls, keys):
        h = cls(len(keys))
        h.heap = list(range(len(keys)))
        h.pos = list(range(len(keys)))
        h.key = list(keys)
        for k in range(len(keys) // 2 - 1, -1, -1):
            h._down(k)
        return h
