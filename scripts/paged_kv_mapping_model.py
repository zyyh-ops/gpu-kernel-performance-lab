"""CPU-only semantic model of a simplified slot_mapping-to-physical-KV address.

This module deliberately contains no Triton, Torch, CUDA, or benchmarking.
Passing its tests NEVER proves the GPU kernel works.
"""

def translate_slots(slot_mapping, page_size, num_blocks):
    """Map each distinct logical token to (block, slot-in-block) coordinates.

    Sentinel -1, repeated slots, and out-of-range slots are invalid in this lab.
    """
    if page_size <= 0 or num_blocks <= 0:
        raise ValueError('page_size and num_blocks must be positive')
    if not slot_mapping:
        raise ValueError('Empty token sequence is not supported by the GPU lab')
    capacity = page_size * num_blocks
    seen = set()
    result = []
    for slot in slot_mapping:
        if type(slot) is not int or not 0 <= slot < capacity:
            raise ValueError('invalid or out-of-bounds physical slot')
        if slot in seen:
            raise ValueError('duplicate physical slot')
        seen.add(slot)
        result.append((slot // page_size, slot % page_size))
    return result


def write_reference(vectors, slot_mapping, page_size, num_blocks, untouched=None):
    """Simulate K *or* V write using plain Python feature vectors."""
    if len(vectors) != len(slot_mapping) or not vectors:
        raise ValueError('vectors must correspond 1:1 to token mappings')
    features = len(vectors[0])
    if features == 0 or any(len(v) != features for v in vectors):
        raise ValueError('all feature vectors must have the same positive width')
    coords = translate_slots(slot_mapping, page_size, num_blocks)
    cache = [[[untouched] * features for _ in range(page_size)] for _ in range(num_blocks)]
    for t, (block, offset) in enumerate(coords):
        cache[block][offset] = list(vectors[t])
    return cache
