def merge_sort(a, key=None, reverse=False):
    # 1. stable merge sort, O(n log n)
    if key is None:
        key = lambda v: v
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    L = merge_sort(a[:mid], key, reverse)
    R = merge_sort(a[mid:], key, reverse)
    out, i, j = [], 0, 0
    while i < len(L) and j < len(R):
        # 2. take right only if strictly better, so ties keep order
        better = key(R[j]) > key(L[i]) if reverse else key(R[j]) < key(L[i])
        if better:
            out.append(R[j])
            j += 1
        else:
            out.append(L[i])
            i += 1
    out.extend(L[i:])
    out.extend(R[j:])
    return out