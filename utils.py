import numpy
import scipy

from scipy import spatial, stats

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def curel(words, vecs, mode='language'):
    combs = set([tuple(sorted([a, b])) for a in words for b in words])
    sims = list()
    for w_one, w_two in combs:
        if mode == 'language':
            sim = 1 - scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        elif mode == 'fig':
            sim = 8 - levenshtein(w_one, w_two)
        sims.append(sim)
    return sims

def seqrel(words, vecs, mode='language'):
    combs = [(words[i], words[i+1]) for i in range(len(words)-1)]
    sims = list()
    for w_one, w_two in combs:
        if mode == 'language':
            sim = 1 - scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        elif mode == 'fig':
            sim = 8 - levenshtein(w_one, w_two)
        sims.append(sim)
    return sims

def switches_and_clusters(words, vecs, threshold, mode='language'):
    '''
    ### threshold
    all_combs = set([tuple(sorted([a, b])) for a in words for b in words])
    sims = list()
    for w_one, w_two in all_combs:
        sim = 1 -scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        sims.append(sim)
    threshold = numpy.median(sims)
    '''
    ### clusters
    clusters = 1
    switches = 0
    combs = [(words[i], words[i+1]) for i in range(len(words)-1)]
    for w_one, w_two in combs:
        if mode == 'language':
            sim = 1 - scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        elif mode == 'fig':
            sim = 8 - levenshtein(w_one, w_two)
        if sim >= threshold:
            pass
        else:
            clusters += 1
            switches += 1
    #§switches = switches / len(combs)
    return switches, clusters

def temporal_analysis(words, vecs, rts, mode='language'):
    combs = [(words[i], words[i+1]) for i in range(len(words)-1)]
    sims = list()
    for w_one, w_two in combs:
        if mode == 'language':
            sim = scipy.spatial.distance.cosine(vecs[w_one], vecs[w_two])
        elif mode == 'fig':
            sim = 8 - levenshtein(w_one, w_two)
        sims.append(sim)
    assert len(sims) == len(rts[1:])
    corr = scipy.stats.pearsonr(sims, rts[1:])[0]
    return corr

def manual_correction(word, stopwords, manual_corr_toks):
    #if word in manual_corr_toks.keys():
    #    word = manual_corr_toks[word]
    ### removing stopwords
    word = ' '.join([w for w in word.split() if w.lower() not in stopwords])
    return word
