import heapq, random, sys
sys.path.insert(0, "src")
from heap import IndexedMaxHeap

def drain(h):
    out = []
    while len(h):
        out.append(h.pop()[1])
    return out

n = 200
keys = [random.random() for _ in range(n)]

# 1. push all, pop order must match heapq
h = IndexedMaxHeap(n)
for i, k in enumerate(keys):
    h.push(i, k)
assert drain(h) == heapq.nlargest(n, keys)

# 2. O(n) build gives the same order
assert drain(IndexedMaxHeap.build(keys)) == heapq.nlargest(n, keys)

# 3. update: raise one key, lower another
h = IndexedMaxHeap.build(keys)
h.update(5, 10.0)
assert h.pop() == (5, 10.0)
h.update(7, -1.0)
expected = [(-1.0 if i == 7 else k) for i, k in enumerate(keys) if i != 5]
assert drain(h) == sorted(expected, reverse=True)

print("heap tests passed")