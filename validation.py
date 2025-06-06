import fasttext
import matplotlib
import mne
import numpy
import os
import pickle
import random
import re
import scipy
import sklearn
import spacy

from matplotlib import font_manager, pyplot
from mne import stats
from scipy import spatial, stats
from tqdm import tqdm

from utils import curel, seqrel, manual_correction, switches_and_clusters, temporal_analysis
from utf_utils import transform_german_word

def check_vocab(corr_toks, vocab):
    assert type(corr_toks) in (set, list)
    check = list()
    for full_w in corr_toks:
        counter = 0
        if len(full_w.split()) == 1:
            if full_w not in vocab.keys():
                check.append(False)
            else:
                check.append(True)
        else:
            within_w = list()
            for w in full_w.split():
                if w not in vocab.keys():
                    within_w.append(False)
                else:
                    within_w.append(True)
            if False not in within_w:
                check.append(True)
            else:
                check.append(False)
    if True not in check:
        missing = True
    else:
        missing = False
    return missing


### read dataset
lines = list()
total_rows = 0
with open(os.path.join('data', 'sandra_old.csv')) as i:
    for l_i, l in enumerate(i):
        l = re.sub(r'\'|\"', r'', l)
        line = l.strip().split(',')
        if l_i == 0:
            header = line.copy()
            full_dataset = {h : list() for h in header}
            continue
        if line[header.index('Condition')].strip() != 'Categories':
            continue
        if len(line[header.index('Word')].strip()) < 2:
            continue
        assert len(line) == len(header)
        for val, h in zip(line, header):
            full_dataset[h].append(val)
        total_rows += 1

rts = {int(sub) : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['Age'])
                         }
                     for task in set(full_dataset['Condition'])
                     }
                     for sub in set(full_dataset['Subj'])
                     }
fluencies = {int(sub) : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['Age'])
                         }
                     for task in set(full_dataset['Condition'])
                     }
                     for sub in set(full_dataset['Subj'])
                     }
accuracies = {int(sub) : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['Age'])
                         }
                     for task in set(full_dataset['Condition'])
                     }
                     for sub in set(full_dataset['Subj'])
                     }

cats = set()
for row in tqdm(range(total_rows)):
    sub = int(full_dataset['Subj'][row])
    task = full_dataset['Condition'][row]
    cond = full_dataset['Age'][row]
    cat = full_dataset['Category'][row].strip().lower()
    if task != 'Categories':
        continue
    else:
        cats.add(cat)
    try:
        rt = float(full_dataset['Onset'][row])*0.0001
    except ValueError:
        rt = numpy.nan
    word = full_dataset['Word'][row].strip().lower()
    try:
        accuracy = int(full_dataset['Correct'][row].strip())
    except ValueError:
        accuracy = 0
    if len(word) == 1:
        print(word)
    if cat not in rts[sub][task][cond].keys():
        rts[sub][task][cond][cat] = list()
        fluencies[sub][task][cond][cat] = list()
        accuracies[sub][task][cond][cat] = list()
    ### log10 for rts
    rts[sub][task][cond][cat].append(numpy.log10(1+rt))
    fluencies[sub][task][cond][cat].append(word)
    accuracies[sub][task][cond][cat].append(accuracy)
print(cats)

case = 'uncased'
lang = 'de'
f = 'wac'
min_count = 10

base_folder = os.path.join('..', '..', 'counts',
                       lang, 
                       f,
                       )
with open(os.path.join(
                        base_folder,
                       '{}_{}_{}_word_freqs.pkl'.format(
                                                         lang, 
                                                         f,
                                                         case
                                                         ),
                       ), 'rb') as i:
    freqs = pickle.load(i)
with open(os.path.join(
                        base_folder,
                       '{}_{}_{}_word_pos.pkl'.format(
                                                         lang, 
                                                         f,
                                                         case
                                                         ),
                       ), 'rb') as i:
    pos = pickle.load(i)
vocab_file = os.path.join(
                        base_folder,
                       '{}_{}_{}_vocab_min_{}.pkl'.format(
                                                           lang, 
                                                           f,
                                                           case,
                                                           min_count
                                                           ),
                       )
with open(vocab_file, 'rb') as i:
    vocab = pickle.load(i)
print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
print('total size of the vocabulary: {:,} words'.format(max(vocab.values())))
coocs_file = os.path.join(base_folder,
              '{}_{}_forward-coocs_{}_min_{}_win_5.pkl'.format(
                                                                 lang,
                                                                 f,
                                                                 case,
                                                                 min_count
                                                                 )
                  )
with open(coocs_file, 'rb') as i:
    coocs = pickle.load(i)

ft = fasttext.load_model(os.path.join('/', 'data', 'u_bruera_software', 'word_vectors','de', 'cc.de.300.bin'))
ft_vocab = {w : w for w in ft.words}

### read stopwords
stopwords = list()
with open(os.path.join('data', 'german_stopwords.txt')) as i:
    for l in i:
        line = l.strip().lower()
        stopwords.append(line)

vecs = dict()
word_vecs = dict()
lemma_vecs = dict()
corr_vecs = dict()
to_be_checked = list()

for row in tqdm(range(total_rows)):
    task = full_dataset['Condition'][row]
    if 'Categories' in task:
        word = full_dataset['Word'][row].strip().lower()
        ### automatic correction + multiple orthographical versions
        corr_word = transform_german_word(word, ft_vocab)
        corr_toks = set([w for c_w in corr_word for w in c_w.split() if w!=''])
        assert len(corr_toks) > 0
        ### words to be checked
        is_missing = check_vocab(corr_toks, ft_vocab)
        if is_missing:
            to_be_checked.append(word)

        ### extracting word vectors

        ### words without automatic correction
        word_vecs[word] = numpy.average([ft.get_word_vector(w) for w in word], axis=0)
        ### words after automatic correction
        corr_vecs[word] = numpy.average([ft.get_word_vector(w) for w in corr_toks], axis=0)
        '''
        ### lemmas after automatic corrections
        lemma_corr_toks = [w.lemma_ for c_w in corr_toks for w in spacy_model(c_w)]
        lemma_vecs[word] = numpy.average([ft.get_word_vector(w) for w in lemma_corr_toks], axis=0)
        '''

proto_ws = dict()

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['Categories'].items():
        for cat, words in cond_data.items():
            curr_accs = accuracies[_]['Categories'][cond][cat]
            for a, w in zip(curr_accs, words):
                if a == 0:
                    continue
                try:
                    proto_ws[cat].add(w)
                except KeyError:
                    proto_ws[cat] = set([w])

proto_vecs = {cat : numpy.average([word_vecs[w] for w in ws], axis=0) for cat, ws in proto_ws.items()}

proto_results = dict()
sims = dict()

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['Categories'].items():
        if cond not in proto_results.keys():
            proto_results[cond] = dict()
        for cat, words in cond_data.items():
            if cat not in proto_results[cond].keys():
                proto_results[cond][cat] = list()
            curr_rts = rts[_]['Categories'][cond][cat]
            curr_sims = list()
            for w in words:
                try:
                    sim = sims[(cat, w)]
                except KeyError:
                    sim = scipy.spatial.distance.cosine(proto_vecs[cat], word_vecs[w])
                    sims[(cat, w)] = sim
                curr_sims.append(sim)
            corr = scipy.stats.spearmanr(curr_rts, curr_sims).statistic
            proto_results[cond][cat].append(corr)

seq_results = dict()
len_results = dict()
freq_results = dict()
logfreq_results = dict()
cooc_results = dict()
surp_results = dict()

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['Categories'].items():
        if cond not in seq_results.keys():
            seq_results[cond] = dict()
            len_results[cond] = dict()
            freq_results[cond] = dict()
            logfreq_results[cond] = dict()
            cooc_results[cond] = dict()
            surp_results[cond] = dict()
        for cat, words in cond_data.items():
            if cat not in seq_results[cond].keys():
                seq_results[cond][cat] = list()
                len_results[cond][cat] = list()
                freq_results[cond][cat] = list()
                logfreq_results[cond][cat] = list()
                cooc_results[cond][cat] = list()
                surp_results[cond][cat] = list()
            curr_rts = rts[_]['Categories'][cond][cat]
            curr_sims = list()
            for w_i, w in enumerate(words):
                if w_i == 0:
                    continue
                else:
                    start_vec = word_vecs[words[w_i-1]]
                sim = scipy.spatial.distance.cosine(start_vec, word_vecs[w])
                curr_sims.append(sim)
            ### fasttext
            corr = scipy.stats.spearmanr(curr_rts[1:], curr_sims).statistic
            seq_results[cond][cat].append(corr)
            ### length
            lens = [len(w) for w in words[1:]]
            corr = scipy.stats.spearmanr(curr_rts[1:], lens).statistic
            len_results[cond][cat].append(corr)
            ### frequency
            curr_sims = list()
            for w_i, w in enumerate(words):
                if w_i == 0:
                    continue
                try:
                    curr_sims.append(freqs[w])
                except KeyError:
                    curr_sims.append(0)
            #corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.array(curr_sims)).statistic
            corr = scipy.stats.pearsonr(curr_rts[1:], -numpy.array(curr_sims)).statistic
            freq_results[cond][cat].append(corr)
            #corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.log2(1+numpy.array(curr_sims))).statistic
            corr = scipy.stats.pearsonr(curr_rts[1:], -numpy.log2(1+numpy.array(curr_sims))).statistic
            logfreq_results[cond][cat].append(corr)
            ### surprisal
            curr_sims = list()
            for w_i, w in enumerate(words):
                if w_i == 0:
                    continue
                else:
                    start_w = words[w_i-1]
                try:
                    assert vocab[w] != 0
                    assert vocab[start_w] != 0
                except (AssertionError, KeyError):
                    curr_sims.append(0)
                    continue
                try:
                    curr_sims.append(coocs[vocab[start_w]][vocab[w]])
                except KeyError:
                    curr_sims.append(0)
            #corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.array(curr_sims)).statistic
            corr = scipy.stats.pearsonr(curr_rts[1:], -numpy.array(curr_sims)).statistic
            cooc_results[cond][cat].append(corr)
            #corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.log2(1+numpy.array(curr_sims))).statistic
            corr = scipy.stats.pearsonr(curr_rts[1:], -numpy.log2(1+numpy.array(curr_sims))).statistic
            surp_results[cond][cat].append(corr)

import pdb; pdb.set_trace()

### writing to file words with wrong spelling
with open(os.path.join('data', 'to_be_checked.tsv'), 'w') as o:
    o.write('original_transcription\tcorrected_spelling\tother_variants_(e.g._split_compounds)\n')
    for w in set(to_be_checked):
        o.write('{}\tx\tx\tx\tx\n'.format(w))

### averaging word vectors
vecs = {w : numpy.average(
                          [
                           #word_vecs[w],
                           #lemma_vecs[w],
                           corr_vecs[w],
                           ], axis=0) for w in word_vecs.keys()}
'''
### z-scoring word vectors
vecs_means = numpy.average(numpy.array([v for v in vecs.values()]), axis=0)
vecs_stds = numpy.std(numpy.array([v for v in vecs.values()]), axis=0)
assert vecs_means.shape == (300,)
assert vecs_stds.shape == (300,)
vecs = {k : (v-vecs_means)/vecs_stds for k,v in vecs.items()}
assert set([v.shape for v in vecs.values()]) == {(300,)}
'''

all_rts = {cond : dict() for cond in set(full_dataset['cond'])}
log_all_rts = {cond : dict() for cond in set(full_dataset['cond'])}
curels = {cond : dict() for cond in set(full_dataset['cond'])}
seqrels = {cond : dict() for cond in set(full_dataset['cond'])}
switches = {cond : dict() for cond in set(full_dataset['cond'])}
temporal_correlations = {cond : dict() for cond in set(full_dataset['cond'])}

### computing thresholds
thresholds = {'overall' : list()}
cond_thresholds = dict()
subject_thresholds = {s : {'overall' : list()} for s in fluencies.keys()}
for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['Categories'].items():
        for cat, words in cond_data.items():
            thresholds['overall'].extend(seqrel(words, vecs))
            subject_thresholds[_]['overall'].extend(seqrel(words, vecs))
            if cat not in thresholds.keys():
                thresholds[cat] = list()
            if cat not in subject_thresholds.keys():
                subject_thresholds[_][cat] = list()
            if cond not in cond_thresholds.keys():
                cond_thresholds[cond] = dict()
                cond_thresholds[cond]['overall'] = list()
            if cat not in cond_thresholds[cond].keys():
                cond_thresholds[cond][cat] = list()
            thresholds[cat].extend(seqrel(words, vecs))
            subject_thresholds[_][cat].extend(seqrel(words, vecs))
            cond_thresholds[cond]['overall'].extend(seqrel(words, vecs))
            cond_thresholds[cond][cat].extend(seqrel(words, vecs))
thresholds = {k : numpy.median(v) for k, v in thresholds.items()}
subject_thresholds = {s : {k : numpy.median(v) for k, v in thresh.items()} for s , thresh in subject_thresholds.items()}
cond_thresholds = {s : {k : numpy.median(v) for k, v in thresh.items()} for s , thresh in cond_thresholds.items()}

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['Categories'].items():
        for cat, words in cond_data.items():
            if cat not in curels[cond].keys():
                curels[cond][cat] = list()
                seqrels[cond][cat] = list()
                switches[cond][cat] = list()
                temporal_correlations[cond][cat] = list()
                all_rts[cond][cat] = list()
                log_all_rts[cond][cat] = list()
            all_rts[cond][cat].extend(rts[_]['Categories'][cond][cat])
            log_all_rts[cond][cat].extend([numpy.log(val+1) for val in rts[_]['Categories'][cond][cat]])
            curels[cond][cat].append(numpy.nanmean(curel(words, vecs)))
            seqrels[cond][cat].append(numpy.nanmean(seqrel(words, vecs)))
            ### overall threshold as in Kim et al. 2019
            #switches[cond][cat].append(switches_and_clusters(words, vecs, thresholds['overall'])[0])
            ### category-specific threshold as Ocalam et al. 2022
            ### first is subject
            switches[cond][cat].append((_, switches_and_clusters(words, vecs, thresholds[cat])[0]))
            current_rts = rts[_]['Categories'][cond][cat]
            temporal_correlations[cond][cat].append(temporal_analysis(words, vecs, current_rts))

os.makedirs('pkls', exist_ok=True)
for d, n in [(switches, 'switches'), (all_rts, 'raw_rts')]:
    with open(os.path.join('pkls', '{}.pkl'.format(n)), 'wb') as i:
        pickle.dump(d, i)

txt_folder = os.path.join('results', 'semantic_fluency')
os.makedirs(txt_folder, exist_ok=True)
### correlation between switches and RTs
with open(os.path.join(txt_folder, 'RT-switches_correlations.tsv'), 'w') as o:
    o.write('condition\tpearson_correlation\tuncorrected_p-value\n')
    for area, area_switches in switches.items():
        keys = list(area_switches.keys())
        corr_switches = [numpy.average(area_switches[k]) for k in keys]
        corr_rts = [numpy.average(all_rts[area][k]) for k in keys]
        corr_log_rts = [numpy.average(log_all_rts[area][k]) for k in keys]
        rt_corr = scipy.stats.pearsonr(corr_switches, corr_rts)
        o.write('RTs\t{}\t{}\t{}\n'.format(area, round(rt_corr.statistic, 4), rt_corr.pvalue))
        log_rt_corr = scipy.stats.pearsonr(corr_switches, corr_log_rts)
        o.write('log(1+RT)\t{}\t{}\t{}\n'.format(area, round(log_rt_corr.statistic, 4), log_rt_corr.pvalue))

