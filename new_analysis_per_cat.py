import fasttext
import pingouin
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


### read dataset
lines = list()
#total_rows = list()
total_rows = -1
with open(os.path.join('data', 'all_tasks.tsv')) as i:
    for l_i, l in enumerate(i):
        l = re.sub(r'\'|\"', r'', l)
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            full_dataset = {h : list() for h in header}
            continue
        if line[header.index('task')] != 'sem_fluency':
            continue
        for val, h in zip(line, header):
            full_dataset[h].append(val)
        total_rows += 1
#total_rows = l_i

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

rts = {cat : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['cond'])
                         }
                     for task in set(full_dataset['task'])
                     }
                     #for sub in set(full_dataset['participant'])
                     for cat in set(full_dataset['item'])
                     }
fluencies = {cat : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['cond'])
                         }
                     for task in set(full_dataset['task'])
                     }
                     #for sub in set(full_dataset['participant'])
                     for cat in set(full_dataset['item'])
                     }
accuracies = {cat : {
                     task : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                }
                         for cond in set(full_dataset['cond'])
                         }
                     for task in set(full_dataset['task'])
                     }
                     #for sub in set(full_dataset['participant'])
                     for cat in set(full_dataset['item'])
                     }

cats = set()
for row in range(total_rows):
    sub = int(full_dataset['participant'][row])
    task = full_dataset['task'][row]
    cond = full_dataset['cond'][row]
    cat = full_dataset['item'][row]
    assert 'sem' in task
    cats.add(cat)
    rt = float(full_dataset['rt'][row])
    word = full_dataset['response'][row].strip()
    accuracy = int(full_dataset['correct'][row].strip())
    if len(word) == 1:
        print(word)
    word = manual_correction(word, stopwords, manual_corr_toks)
    if sub not in rts[cat][task][cond].keys():
        rts[cat][task][cond][sub] = list()
        fluencies[cat][task][cond][sub] = list()
        accuracies[cat][task][cond][sub] = list()
    #rts[sub][task][cond][cat].append(rt)
    ### log10 for rts
    rts[cat][task][cond][sub].append(numpy.log10(1+rt))
    fluencies[cat][task][cond][sub].append(word)
    accuracies[cat][task][cond][sub].append(accuracy)
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

#for _, sub_data in tqdm(fluencies.items()):
for cat, cat_data in tqdm(fluencies.items()):
    for cond, cond_data in cat_data['sem_fluency'].items():
        for sub, words in cond_data.items():
            curr_accs = accuracies[cat]['sem_fluency'][cond][sub]
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

#for _, sub_data in tqdm(fluencies.items()):
for cat, cat_data in tqdm(fluencies.items()):
    for cond, cond_data in cat_data['sem_fluency'].items():
        if cond not in proto_results.keys():
            proto_results[cond] = dict()
        for sub, words in cond_data.items():
            if cat not in proto_results[cond].keys():
                proto_results[cond][cat] = list()
            curr_rts = rts[cat]['sem_fluency'][cond][sub]
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

#for _, sub_data in tqdm(fluencies.items()):
for cat, cat_data in tqdm(fluencies.items()):
    for k in results.keys():
        results[k][cat]  = dict()
    for cond, cond_data in cat_data['sem_fluency'].items():
        if cond not in results['word_length'][cat].keys():
            for k_one in results.keys():
                results[k_one][cat][cond] = dict()
        for sub, words in cond_data.items():
            if sub not in results['word_length'][cat][cond].keys():
                for k_one in results.keys():
                    results[k_one][cat][cond][sub] = dict()
            curr_rts = rts[cat]['sem_fluency'][cond][sub]
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
            results['fasttext'][cat][cond][sub]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], curr_sims).statistic
            results['fasttext'][cat][cond][sub]['order'] = corr
            ### length
            lens = [len(w) for w in words[1:]]
            ### rts
            corr = scipy.stats.spearmanr(curr_rts[1:], lens).statistic
            results['word_length'][cat][cond][sub]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], lens).statistic
            results['word_length'][cat][cond][sub]['order'] = corr
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
            results['neg_word_frequency'][cat][cond][sub]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], -numpy.array(curr_sims)).statistic
            results['neg_word_frequency'][cat][cond][sub]['order'] = corr
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
            results['word_surprisal'][cat][cond][sub]['rts'] = corr
            ### order
            corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], -numpy.log2(1+numpy.array(curr_sims))).statistic
            results['word_surprisal'][cat][cond][sub]['order'] = corr
            ### simple co-occurrences
            #corr = scipy.stats.spearmanr(curr_rts[1:], -numpy.array(curr_sims)).statistic
            #corr = scipy.stats.spearmanr([_ for _ in range(len(curr_rts[1:]))], -numpy.array(curr_sims)).statistic

for mode in ['order', 'rts']:
    out_f = os.path.join('new_results', 'cat_per_cat', mode)
    os.makedirs(out_f, exist_ok=True)
    for model, model_data in results.items():

        comp = {cat : {cond : [sub_data[mode] for sub, sub_data in cond_data.items()] for cond, cond_data in cat_data.items()} for cat, cat_data in model_data.items()}
        with open(os.path.join(out_f, 'cat-per-cat_{}_{}.tsv'.format(mode, model)), 'w') as o:
            o.write('mode\tmodel\tcat\tdifficulty\tsham_corr\tifg_corr\tifg_raw_p\tifg_effsize\tpresma_corr\tpresma_raw_p\tpresma_effsize\tdual_corr\tdual_raw_p\tdual_effsize\n')
            for cat, cat_data in comp.items():
                o.write('{}\t{}\t{}\t{}\t'.format(mode, model, cat, sorted_difficulties.index(cat)))
                sham_corr = numpy.nanmean(cat_data['sham'])
                o.write('{}\t'.format(sham_corr))
                corrs = dict()
                ifg_corr = numpy.nanmean(cat_data['IFG'])
                corrs['IFG'] = ifg_corr
                presma_corr = numpy.nanmean(cat_data['preSMA'])
                corrs['preSMA'] = presma_corr
                dual_corr = numpy.nanmean(cat_data['dual'])
                corrs['dual'] = dual_corr
                print('\n')
                print('{}, {} - {}'.format(model, mode, cat))
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
                                          cat_data['sham'],
                                          cat_data[other_cond],
                                          hyp='two_tailed',
                                          )
                    o.write('{}\t{}\t'.format(corrs[other_cond], res['p_val']))
                    eff_size = pingouin.compute_effsize(
                                                        cat_data['sham'],
                                                        cat_data[other_cond],
                                                        eftype='hedges',
                                                        )
                    o.write('{}\t'.format(eff_size))
                    print('permutation - {}, {}: {}'.format(sham, other_cond, res['p_val']))
                    print('\n')
                o.write('\n')
                print('\n\n')
