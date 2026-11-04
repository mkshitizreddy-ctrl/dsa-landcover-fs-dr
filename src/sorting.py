def merge_sort(a, key=None, reverse=False):
    if key is None:
        key = lambda v: v
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], key, reverse)
    right = merge_sort(a[mid:], key, reverse)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        right_first = key(right[j]) > key(left[i]) if reverse else key(right[j]) < key(left[i])
        if right_first:
            out.append(right[j])
            j += 1
        else:
            out.append(left[i])
            i += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out
