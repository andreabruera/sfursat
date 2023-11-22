import numpy
import os
import pickle

os.makedirs('pickles', exist_ok=True)

concept_net = dict()
with open(os.path.join('..', '..', 'dataset', 'word_vectors', 'numberbatch-19.08.txt')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split(' ')
        lang_word = line[0].split('/')
        lang = lang_word[-2]
        word = lang_word[-1]
        if lang == 'de':
            vec = numpy.array(line[1:], dtype=numpy.float64)
            concept_net[word] = vec
with open(os.path.join('pickles', 'conceptnet_de.pkl'), 'wb') as o:
    pickle.dump(concept_net, o)
