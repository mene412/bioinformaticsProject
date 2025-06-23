from datasketch import MinHash # type: ignore
from config import NUM_PERM


def jaccard_distance(set1, set2):
    # TODO check che faccia giusta l'intersezione e l'unione
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - inter / union if union > 0 else 1

# minhashes occupa (n * M) spazio. n array da M
def compute_minhashes(samples):
    minhashes = {}
    for sid, seqs in samples.items(): # sampleID + sample
        mh = MinHash(num_perm=NUM_PERM)
        # O(M), M = numero reads per sample
        for seq in seqs:
            mh.update(seq.encode('utf-8'))
        minhashes[sid] = mh
    return minhashes