class IndexedMaxHeap:
    # 1. items are feature ids 0..n-1, each with a float key
    def __init__(self, n):
        self.heap = []        # 2. heap array of ids
        self.pos = [-1] * n   # 3. position of each id in heap (-1 = absent)
        self.key = [0.0] * n  # 4. key of each id

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
        # 5. move up while larger than parent
        while k > 0:
            p = (k - 1) // 2
            if self.key[self.heap[k]] <= self.key[self.heap[p]]:
                break
            self._swap(k, p)
            k = p

    def _down(self, k):
        # 6. move down while a child is larger
        n = len(self.heap)
        while True:
            l, r, m = 2 * k + 1, 2 * k + 2, k
            if l < n and self.key[self.heap[l]] > self.key[self.heap[m]]:
                m = l
            if r < n and self.key[self.heap[r]] > self.key[self.heap[m]]:
                m = r
            if m == k:
                break
            self._swap(k, m)
            k = m

    def push(self, i, key):
        # 7. O(log n)
        if self.pos[i] != -1:
            raise ValueError("id already in heap")
        self.key[i] = key
        self.heap.append(i)
        self.pos[i] = len(self.heap) - 1
        self._up(len(self.heap) - 1)

    def pop(self):
        # 8. O(log n): swap root with last, remove, fix root
        top = self.heap[0]
        k = self.key[top]
        self._swap(0, len(self.heap) - 1)
        self.heap.pop()
        self.pos[top] = -1
        if self.heap:
            self._down(0)
        return top, k

    def update(self, i, key):
        # 9. O(log n): change key of an item already in the heap
        old = self.key[i]
        self.key[i] = key
        if key > old:
            self._up(self.pos[i])
        else:
            self._down(self.pos[i])

    @classmethod
    def build(cls, keys):
        # 10. O(n) bottom-up heapify
        h = cls(len(keys))
        h.heap = list(range(len(keys)))
        h.pos = list(range(len(keys)))
        h.key = list(keys)
        for k in range(len(keys) // 2 - 1, -1, -1):
            h._down(k)
        return h