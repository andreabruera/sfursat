import numpy
import scipy

from scipy import spatial, stats

def curel(words, vecs):
    combs = set([tuple(sorted([a, b])) for a in words for b in words])
    sims = list()
    for w_one, w_two in combs:
        sim = 1 -scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        sims.append(sim)
    return sims
def seqrel(words, vecs):
    combs = [(words[i], words[i+1]) for i in range(len(words)-1)]
    sims = list()
    for w_one, w_two in combs:
        sim = 1 -scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        sims.append(sim)
    return sims
def switches_and_clusters(words, vecs):
    all_combs = set([tuple(sorted([a, b])) for a in words for b in words])
    sims = list()
    for w_one, w_two in all_combs:
        sim = 1 -scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        sims.append(sim)
    threshold = numpy.median(sims)
    clusters = 1
    switches = 0
    combs = [(words[i], words[i+1]) for i in range(len(words)-1)]
    for w_one, w_two in combs:
        sim = 1 - scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        if sim > threshold:
            pass
        else:
            clusters += 1
            switches += 1
    return switches, clusters
def temporal_analysis(words, vecs, rts):
    combs = [(words[i], words[i+1]) for i in range(len(words)-1)]
    sims = list()
    for w_one, w_two in combs:
        sim = scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        sims.append(sim)
    assert len(sims) == len(rts[1:])
    corr = scipy.stats.pearsonr(sims, rts[1:])[0]
    return corr
