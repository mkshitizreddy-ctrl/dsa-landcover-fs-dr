
import heapq
import random
import sys

sys.path.insert(0, "src")
from heap import IndexedMaxHeap


def drain(heap):
    result = []
    while len(heap):
        result.append(heap.pop()[1])
    return result


def test_push_and_pop_order():
    rng = random.Random(42)
    keys = [rng.random() for _ in range(200)]

    heap = IndexedMaxHeap(len(keys))
    for i, key in enumerate(keys):
        heap.push(i, key)

    assert drain(heap) == heapq.nlargest(len(keys), keys)


def test_linear_build_and_pop_order():
    rng = random.Random(42)
    keys = [rng.random() for _ in range(200)]

    assert drain(IndexedMaxHeap.build(keys)) == heapq.nlargest(
        len(keys), keys
    )


def test_update_keys():
    rng = random.Random(42)
    keys = [rng.random() for _ in range(200)]

    heap = IndexedMaxHeap.build(keys)
    heap.update(5, 10.0)

    assert heap.pop() == (5, 10.0)

    heap.update(7, -1.0)
    expected = [
        -1.0 if i == 7 else key
        for i, key in enumerate(keys)
        if i != 5
    ]

    assert drain(heap) == sorted(expected, reverse=True)
