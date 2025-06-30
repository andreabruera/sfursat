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

def two_samples_permutation(one, two, hyp, perms=1000):
    possibilities = ['one>two', 'two<one', 'two_tailed']
    assert hyp in possibilities
    double = [v for v in one] + [v for v in two]
    real_diff = numpy.nanmean(one)-numpy.nanmean(two)
    iters = list()
    for _ in range(perms):
        iter_random = random.sample(double, k=len(double))
        iter_one = iter_random[:len(one)]
        iter_two = iter_random[len(one):]
        iter_diff = numpy.nanmean(iter_one)-numpy.nanmean(iter_two)
        iters.append(iter_diff)
    if hyp == 'two_tailed':
        contra_hyp = sum([1 if abs(d)>abs(real_diff) else 0 for d in iters])
    if hyp == 'one>two':
        contra_hyp = sum([1 if d>real_diff else 0 for d in iters])
    if hyp == 'two<one':
        contra_hyp = sum([1 if d<real_diff else 0 for d in iters])
    p_val = (contra_hyp+1) / (perms+1)
    denom_d = numpy.sqrt((numpy.nanstd(one)**2 + numpy.nanstd(two)**2)/2)
    effect_size = real_diff / denom_d
    return {'delta' : real_diff, 'p_val' : p_val, 'effect_size' : effect_size}

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

case = 'uncased'
lang = 'de'
f = 'wac'
#f = 'opensubs'
#f = 'cc100'
min_count = 10
#min_count = 100

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
              #'{}_{}_forward-coocs_{}_min_{}_win_20.pkl'.format(
              '{}_{}_forward-coocs_{}_min_{}_win_5.pkl'.format(
                                                                 lang,
                                                                 f,
                                                                 case,
                                                                 min_count
                                                                 )
                  )
with open(coocs_file, 'rb') as i:
    coocs = pickle.load(i)

#spacy_model = spacy.load('de_core_news_lg')
ft = fasttext.load_model(os.path.join('/', 'data', 'u_bruera_software', 'word_vectors','de', 'cc.de.300.bin'))
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

sorted_difficulties = [v[0] for v in sorted(difficulties.items(), key=lambda item : item[1], reverse=True)]

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
accuracies = {int(sub) : {
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
    accuracy = int(full_dataset['correct'][row].strip())
    if len(word) == 1:
        print(word)
    word = manual_correction(word, stopwords, manual_corr_toks)
    if cat not in rts[sub][task][cond].keys():
        rts[sub][task][cond][cat] = list()
        fluencies[sub][task][cond][cat] = list()
        accuracies[sub][task][cond][cat] = list()
    #rts[sub][task][cond][cat].append(rt)
    ### log10 for rts
    rts[sub][task][cond][cat].append(numpy.log10(1+rt))
    fluencies[sub][task][cond][cat].append(word)
    accuracies[sub][task][cond][cat].append(accuracy)
print(cats)

vecs = dict()
word_vecs = dict()
lemma_vecs = dict()
corr_vecs = dict()
to_be_checked = list()
w_versions = dict()

for row in tqdm(range(total_rows)):
    task = full_dataset['task'][row]
    if 'sem' in task:
        word = full_dataset['response'][row].strip()
        ### manual correction
        word = manual_correction(word, stopwords, manual_corr_toks)
        ### automatic correction + multiple orthographical versions
        corr_word = transform_german_word(word, ft_vocab)
        corr_toks = set([w for c_w in corr_word for w in c_w.split() if w!=''])
        #print(corr_toks)
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
        if len(word.split()) > 1:
            versions_word = word.split()[-1]
        else:
            versions_word = word
        versions_corr_word = transform_german_word(versions_word, ft_vocab)
        versions_corr_toks = set([w for c_w in versions_corr_word for w in c_w.split() if w!=''])
        w_versions[word] = versions_corr_toks

proto_ws = dict()

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['sem_fluency'].items():
        for cat, words in cond_data.items():
            curr_accs = accuracies[_]['sem_fluency'][cond][cat]
            for a, w in zip(curr_accs, words):
                if a == 0:
                    continue
                try:
                    proto_ws[cat].add(w)
                except KeyError:
                    proto_ws[cat] = set([w])

proto_vecs = {cat : numpy.average([corr_vecs[w] for w in ws], axis=0) for cat, ws in proto_ws.items()}

proto_results = dict()
sims = dict()

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['sem_fluency'].items():
        if cond not in proto_results.keys():
            proto_results[cond] = dict()
        for cat, words in cond_data.items():
            if cat not in proto_results[cond].keys():
                proto_results[cond][cat] = list()
            curr_rts = rts[_]['sem_fluency'][cond][cat]
            curr_sims = list()
            for w in words:
                try:
                    sim = sims[(cat, w)]
                except KeyError:
                    sim = scipy.spatial.distance.cosine(proto_vecs[cat], corr_vecs[w])
                    sims[(cat, w)] = sim
                curr_sims.append(sim)
            corr = scipy.stats.spearmanr(curr_rts, curr_sims).statistic
            proto_results[cond][cat].append(corr)

results = {
           'word_length' : dict(),
           'neg_word_frequency' : dict(),
           'word_surprisal' : dict(),
           'fasttext' : dict(),
           }

for _, sub_data in tqdm(fluencies.items()):
    for k in results.keys():
        results[k][_]  = dict()
    for cond, cond_data in sub_data['sem_fluency'].items():
        if cond not in results['word_length'][_].keys():
            for k_one in results.keys():
                results[k_one][_][cond] = dict()
        for cat, words in cond_data.items():
            if cat not in results['word_length'][_][cond].keys():
                for k_one in results.keys():
                    results[k_one][_][cond][cat] = dict()
            curr_rts = rts[_]['sem_fluency'][cond][cat]
            curr_sims = list()
            for w_i, w in enumerate(words):
                if w_i == 0:
                    continue
                else:
                    start_vec = corr_vecs[words[w_i-1]]
                start_vec = corr_vecs[words[0]]
                sim = scipy.spatial.distance.cosine(start_vec, corr_vecs[w])
                curr_sims.append(sim)
            ### fasttext
            ### rts
            corr = scipy.stats.spearmanr(curr_rts[1:], curr_sims).statistic
            results['fasttext'][_][cond][cat]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], curr_sims).statistic
            results['fasttext'][_][cond][cat]['order'] = corr
            ### length
            lens = [len(w) for w in words[1:]]
            ### rts
            corr = scipy.stats.spearmanr(curr_rts[1:], lens).statistic
            results['word_length'][_][cond][cat]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], lens).statistic
            results['word_length'][_][cond][cat]['order'] = corr
            ### frequency
            curr_sims = list()
            for w_i, original_w in enumerate(words):
                if w_i == 0:
                    continue
                w = w_versions[original_w]
                counter = 0
                for tok in w:
                    try:
                        counter += freqs[tok]
                    except KeyError:
                        continue
                curr_sims.append(counter)
            ### rts
            corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.array(curr_sims)).statistic
            results['neg_word_frequency'][_][cond][cat]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], -numpy.array(curr_sims)).statistic
            results['neg_word_frequency'][_][cond][cat]['order'] = corr
            ### log freq
            #corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.log2(1+numpy.array(curr_sims))).statistic
            #corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], -numpy.log2(1+numpy.array(curr_sims))).statistic
            ### surprisal
            curr_sims = list()
            for w_i, w in enumerate(words):
                if w_i == 0:
                    continue
                else:
                    start_w = words[w_i-1]
                counter = 0
                for tok_one in w_versions[start_w]:
                    for tok_two in w_versions[w]:
                        try:
                            assert vocab[tok_one] != 0
                            assert vocab[tok_two] != 0
                        except (AssertionError, KeyError):
                            continue
                        try:
                            counter += coocs[vocab[tok_one]][vocab[tok_two]]
                        except KeyError:
                            continue
                curr_sims.append(counter)
            ### rts
            corr = scipy.stats.pearsonr(curr_rts[1:], -numpy.log2(1+numpy.array(curr_sims))).statistic
            results['word_surprisal'][_][cond][cat]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], -numpy.log2(1+numpy.array(curr_sims))).statistic
            results['word_surprisal'][_][cond][cat]['order'] = corr
            ### simple co-occurrences
            #corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.array(curr_sims)).statistic
            #corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], -numpy.array(curr_sims)).statistic

for mode in ['order', 'rts']:
    for model, model_data in results.items():

        comp = {s : {cond : [cat_data[mode] for cat, cat_data in cond_data.items()] for cond, cond_data in s_data.items()} for s, s_data in model_data.items()}
        sham_corr = numpy.nanmean([numpy.nanmean(comp[v]['sham']) for v in sorted(comp.keys())])
        ifg_corr = numpy.nanmean([numpy.nanmean(comp[v]['IFG']) for v in sorted(comp.keys())])
        presma_corr = numpy.nanmean([numpy.nanmean(comp[v]['preSMA']) for v in sorted(comp.keys())])
        dual_corr = numpy.nanmean([numpy.nanmean(comp[v]['dual']) for v in sorted(comp.keys())])
        print('\n')
        print('{}, {}'.format(model, mode))
        print('average correlation with sham: {}'.format(sham_corr))
        print('average correlation with IFG: {}'.format(ifg_corr))
        print('average correlation with preSMA: {}'.format(presma_corr))
        print('average correlation with dual: {}'.format(dual_corr))
        print('\n')
        for sham, other_cond in [
                      ('sham', 'IFG'),
                      ('sham', 'preSMA'),
                      ('sham', 'dual'),
                      ]:
            '''
            res = scipy.stats.wilcoxon(
                                       [c for v in sorted(comp.keys()) for c in comp[v][sham]], 
                                       [c for v in sorted(comp.keys()) for c in comp[v][other_cond]],
                                       )
            print('wilcoxon - {}, {}: {}'.format(sham, other_cond, res.pvalue))
            '''
            res = two_samples_permutation(
                                  [c for v in sorted(comp.keys()) for c in comp[v][sham]], 
                                  [c for v in sorted(comp.keys()) for c in comp[v][other_cond]],
                                  hyp='two_tailed',
                                  )
            print('permutation - {}, {}: {}'.format(sham, other_cond, res['p_val']))
            print('\n')
        print('\n\n')

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

