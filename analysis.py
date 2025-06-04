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

spacy_model = spacy.load('de_core_news_lg')
ft = fasttext.load_model(os.path.join('data', 'models', 'cc.de.300.bin'))
#ft = fasttext.load_model(os.path.join('..', '..', 'dataset', 'word_vectors', 'de', 'cc.de.300.bin'))
ft_vocab = {w : w for w in ft.words}

### read dataset
lines = list()
with open(os.path.join('data', 'all_tasks.tsv')) as i:
    for l_i, l in enumerate(i):
        l = re.sub(r'\'|\"', r'', l)
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            full_dataset = {h : list() for h in header}
            continue
        for val, h in zip(line, header):
            full_dataset[h].append(val)
total_rows = l_i

### reading manual correction
manual_corr_toks = dict()
with open(os.path.join('data', 'to_be_checked_corrected.tsv')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = [w.strip() for w in l.strip().split('\t')]
        line = [re.sub('\s+', r' ', w) for w in line]
        ### corrected spelling
        if len(line[1]) > 1:
            corr_word = transform_german_word(line[1], ft_vocab)
            corr_toks = set([w for c_w in corr_word for w in c_w.split()])
            is_missing = check_vocab(corr_toks, ft_vocab)
            if not is_missing:
                manual_corr_toks[line[0]] = line[1]
                continue
        continue
        if len(line[2]) > 1:
            corr_word = transform_german_word(line[2], ft_vocab)
            corr_toks = set([w for c_w in corr_word for w in c_w.split()])
            is_missing = check_vocab(corr_toks, ft_vocab)
            if not is_missing:
                manual_corr_toks[line[0]] = line[2]
                continue
        if len(line[3]) >1:
            corr_word = transform_german_word(line[3], ft_vocab)
            corr_toks = set([w for c_w in corr_word for w in c_w.split()])
            is_missing = check_vocab(corr_toks, ft_vocab)
            if not is_missing:
                manual_corr_toks[line[0]] = line[3]
                continue
        ### other name
        if len(line[4]) > 1:
            corr_word = transform_german_word(line[4], ft_vocab)
            corr_toks = set([w for c_w in corr_word for w in c_w.split()])
            is_missing = check_vocab(corr_toks, ft_vocab)
            if not is_missing:
                manual_corr_toks[line[0]] = line[4]
                continue
        ## split compound
        alternative_spellings = [w for w in line[5:] if w!='x']
        if len(alternative_spellings) > 0:
            joint_w = ' '.join(alternative_spellings)
            #capital_joint_w = ' '.join([w.capitalize() for w in alternative_spellings])
            corr_word = transform_german_word(joint_w, ft_vocab)
            corr_toks = set([w for c_w in corr_word for w in c_w.split()])
            is_missing = check_vocab(corr_toks, ft_vocab)
            if not is_missing:
                manual_corr_toks[line[0]] = joint_w

### metrics vs difficulty
difficulties = dict()
with open(os.path.join('data', 'category_ranking.tsv')) as i:
    for l_i, l in enumerate(i):
        if l_i==0:
            continue
        line = l.strip().split('\t')
        difficulties[line[0]] = float(line[2])

### read stopwords
stopwords = list()
with open(os.path.join('data', 'german_stopwords.txt')) as i:
    for l in i:
        line = l.strip().lower()
        stopwords.append(line)

rts = {int(sub) : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['cond'])
                         }
                     for task in set(full_dataset['task'])
                     }
                     for sub in set(full_dataset['participant'])
                     }
fluencies = {int(sub) : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['cond'])
                         }
                     for task in set(full_dataset['task'])
                     }
                     for sub in set(full_dataset['participant'])
                     }

cats = set()
for row in tqdm(range(total_rows)):
    sub = int(full_dataset['participant'][row])
    task = full_dataset['task'][row]
    cond = full_dataset['cond'][row]
    cat = full_dataset['item'][row]
    if 'sem' in task:
        cats.add(cat)
    rt = float(full_dataset['rt'][row])
    word = full_dataset['response'][row].strip()
    word = manual_correction(word, stopwords, manual_corr_toks)
    if cat not in rts[sub][task][cond].keys():
        rts[sub][task][cond][cat] = list()
        fluencies[sub][task][cond][cat] = list()
    rts[sub][task][cond][cat].append(rt)
    fluencies[sub][task][cond][cat].append(word)
print(cats)

vecs = dict()
word_vecs = dict()
lemma_vecs = dict()
corr_vecs = dict()
to_be_checked = list()

for row in tqdm(range(total_rows)):
    task = full_dataset['task'][row]
    if 'sem' in task:
        word = full_dataset['response'][row].strip()
        ### manual correction
        word = manual_correction(word, stopwords, manual_corr_toks)
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
    for cond, cond_data in sub_data['sem_fluency'].items():
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
    for cond, cond_data in sub_data['sem_fluency'].items():
        for cat, words in cond_data.items():
            if cat not in curels[cond].keys():
                curels[cond][cat] = list()
                seqrels[cond][cat] = list()
                switches[cond][cat] = list()
                temporal_correlations[cond][cat] = list()
                all_rts[cond][cat] = list()
                log_all_rts[cond][cat] = list()
            all_rts[cond][cat].extend(rts[_]['sem_fluency'][cond][cat])
            log_all_rts[cond][cat].extend([numpy.log(val+1) for val in rts[_]['sem_fluency'][cond][cat]])
            curels[cond][cat].append(numpy.nanmean(curel(words, vecs)))
            seqrels[cond][cat].append(numpy.nanmean(seqrel(words, vecs)))
            ### overall threshold as in Kim et al. 2019
            #switches[cond][cat].append(switches_and_clusters(words, vecs, thresholds['overall'])[0])
            ### category-specific threshold as Ocalam et al. 2022
            ### first is subject
            switches[cond][cat].append((_, switches_and_clusters(words, vecs, thresholds[cat])[0]))
            current_rts = rts[_]['sem_fluency'][cond][cat]
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

