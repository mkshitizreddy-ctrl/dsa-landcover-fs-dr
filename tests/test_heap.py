
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


def test_equal_keys_pop_lowest_id_first():
    keys = [0.5, 0.9, 0.5, 0.9, 0.5, 0.1, 0.9]
    built = IndexedMaxHeap.build(keys)
    pushed = IndexedMaxHeap(len(keys))
    for i in reversed(range(len(keys))):
        pushed.push(i, keys[i])
    expected = [1, 3, 6, 0, 2, 4, 5]
    assert [built.pop()[0] for _ in keys] == expected
    assert [pushed.pop()[0] for _ in keys] == expected


def test_update_respects_the_tie_rule():
    heap = IndexedMaxHeap.build([0.2, 0.5, 0.5])
    heap.update(0, 0.5)
    assert [heap.pop()[0] for _ in range(3)] == [0, 1, 2]
