import charsplit
import fasttext
import matplotlib
import mne
import numpy
import os
import pickle
import re
import scipy
import spacy

from matplotlib import font_manager, pyplot
from mne import stats
from scipy import spatial, stats
from tqdm import tqdm

from utils import curel, seqrel, switches_and_clusters, temporal_analysis
from utf_utils import transform_german_word

spacy_model = spacy.load('de_core_news_lg')

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

for row in tqdm(range(total_rows)):
    sub = int(full_dataset['participant'][row])
    task = full_dataset['task'][row]
    cond = full_dataset['cond'][row]
    cat = full_dataset['item'][row]
    rt = float(full_dataset['rt'][row])
    word = full_dataset['response'][row].strip()
    if cat not in rts[sub][task][cond].keys():
        rts[sub][task][cond][cat] = list()
        fluencies[sub][task][cond][cat] = list()
    rts[sub][task][cond][cat].append(rt)
    fluencies[sub][task][cond][cat].append(word)

word_vecs = dict()
lemma_vecs = dict()
corr_vecs = dict()
ft = fasttext.load_model(os.path.join('..', '..', 'dataset', 'word_vectors', 'de', 'cc.de.300.bin'))
ft_vocab = {w : w for w in ft.words}
### load conceptnet
vecs = dict()
with open(os.path.join('pickles', 'conceptnet_de.pkl'), 'rb') as i:
    conceptnet = pickle.load(i)
for row in tqdm(range(total_rows)):
    task = full_dataset['task'][row]
    if 'sem' in task:
        word = full_dataset['response'][row].strip()
        word_vecs[word] = numpy.average([ft.get_word_vector(w) for w in word.split()], axis=0)
        corr_word = transform_german_word(word, ft_vocab)
        corr_toks = [w for c_w in corr_word for w in c_w.split()]
        #if len(corr_toks) > 1:
        #print(corr_toks)
        corr_vecs[word] = numpy.average([ft.get_word_vector(w) for w in corr_toks], axis=0)
        lemma_corr_toks = [w.lemma_ for c_w in corr_word for w in spacy_model(c_w)]
        lemma_vecs[word] = numpy.average([ft.get_word_vector(w) for w in lemma_corr_toks], axis=0)
        #lemma = ' '.join([w.lemma_ for w in spacy_model(word)]).lower()
        #lemma_vecs[word] = numpy.average([ft.get_word_vector(w) for w in lemma.split()], axis=0)
        #print(1-scipy.spatial.distance.cosine(word_vecs[word], lemma_vecs[word]))
vecs = {w : numpy.average(
                          [
                           #word_vecs[w], 
                           #lemma_vecs[w], 
                           corr_vecs[w],
                           ], axis=0) for w in word_vecs.keys()}
'''
missing_words = set()
splitter = charsplit.Splitter()
for row in tqdm(range(total_rows)):
    task = full_dataset['task'][row]
    if 'sem' in task:
        word = full_dataset['response'][row].strip()
        if word.lower() not in conceptnet.keys():
            new_word = re.sub('\W', '_', transform_german_word(word, conceptnet))
            if new_word not in conceptnet.keys():
                missing_words.add(word)
        #    split = splitter.split_compound(word)[0][1:]
        #    counter = [sub.lower() in conceptnet.keys() for sub in split]
        #    if True not in counter:
        #        missing_words.append(word)
        #    else:
        #        vec = numpy.average([conceptnet[sub] for sub in split if sub in conceptnet.keys()], axis=0)
        #else:
        #    vec = conceptnet[word]
        #vecs[word] = vec
with open('to_be_checked.tsv', 'w') as o:
    for w in missing_words:
        o.write(w)
        o.write('\n')
import pdb; pdb.set_trace()
'''

curels = {cond : dict() for cond in set(full_dataset['cond'])}
seqrels = {cond : dict() for cond in set(full_dataset['cond'])}
switches = {cond : dict() for cond in set(full_dataset['cond'])}
temporal_correlations = {cond : dict() for cond in set(full_dataset['cond'])}

### computing thresholds
thresholds = {'overall' : list()}
for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['sem_fluency'].items():
        for cat, words in cond_data.items():
            thresholds['overall'].extend(seqrel(words, vecs))
            if cat not in thresholds.keys():
                thresholds[cat] = list()
            thresholds[cat].extend(seqrel(words, vecs))
thresholds = {k : numpy.median(v) for k, v in thresholds.items()}

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['sem_fluency'].items():
        for cat, words in cond_data.items():
            if cat not in curels[cond].keys():
                curels[cond][cat] = list()
                seqrels[cond][cat] = list()
                switches[cond][cat] = list()
                temporal_correlations[cond][cat] = list()
            curels[cond][cat].append(numpy.nanmean(curel(words, vecs)))
            seqrels[cond][cat].append(numpy.nanmean(seqrel(words, vecs)))
            switches[cond][cat].append(switches_and_clusters(words, vecs, thresholds['overall'])[0])
            current_rts = rts[_]['sem_fluency'][cond][cat]
            temporal_correlations[cond][cat].append(temporal_analysis(words, vecs, current_rts))

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'
colors = ['teal', 'goldenrod', 'magenta', 'grey']
out_folder = 'plots'

for metric, results in [('CuRel', curels), ('SeqRel', seqrels), ('Switches', switches)]:
    overall_folder = os.path.join(out_folder, 'overall')
    os.makedirs(overall_folder, exist_ok=True)
    ### plotting overall averages
    fig, ax = pyplot.subplots(constrained_layout=True)
    title = 'Across-categories averages for {}'.format(metric)
    #xs = list(results.keys())
    xs = ['IFG', 'preSMA', 'dual', 'sham']
    ys = [[val for v in results[k].values() for val in v] for k in xs]
    parts = ax.violinplot(ys, positions=range(len(ys)), showmeans=True, showextrema=False,)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontweight='bold')
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        #m = numpy.mean(pc.get_paths()[0].vertices[:, 0])
        #pc.get_paths()[0].vertices[:, 0] = numpy.clip(pc.get_paths()[0].vertices[:, 0], -numpy.inf, m)
    ax.scatter(range(len(xs)), [numpy.average([val for v in results[k].values() for val in v]) for k in xs],zorder=3, color='white', marker='_')
    ax.set_ylabel('Across-categories average {}'.format(metric))
    ax.set_title(title)
    ### p-values
    p_vals = list()
    for k_one, v_one in results.items():
        one = [val for v in v_one.values() for val in v]
        for k_two, v_two in results.items():
            two = [val for v in v_two.values() for val in v]
            if k_one == k_two:
                continue
            key = tuple(sorted([k_one, k_two]))
            if key not in [p[0] for p in p_vals]:
                p = scipy.stats.ttest_ind(
                                      one, 
                                      two, 
                                      #permutations=4096, 
                                      #alternative=direction,
                                      ).pvalue
                p_vals.append([key, p])
    ### fdr correction
    correct_ps = mne.stats.fdr_correction([p[1] for p in p_vals])[1]
    with open(os.path.join(overall_folder, '{}_p-vals_comparisons.tsv'.format(metric)), 'w') as o:
        o.write('comparison\tuncorrected_p-value\tFDR-corrected_p-value\n')
        for a, b in zip(p_vals, correct_ps):
            o.write('{}\t{}\t{}\n'.format(a[0], a[1], b))

    pyplot.savefig(os.path.join(overall_folder, '{}_average.jpg'.format(metric)))
    pyplot.clf()
    pyplot.close()
    ### per-category plots
    for cat in results['sham'].keys():
        cat_folder = os.path.join(out_folder, cat)
        os.makedirs(cat_folder, exist_ok=True)
        ### plotting overall averages
        fig, ax = pyplot.subplots(constrained_layout=True)
        title = '{} scores for {}'.format(metric, cat)
        #xs = list(results.keys())
        xs = ['IFG', 'preSMA', 'dual', 'sham']
        ys = [results[k][cat] for k in xs]
        parts = ax.violinplot(ys, positions=range(len(ys)), showmeans=True, showextrema=False,)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontweight='bold')
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)
            #m = numpy.mean(pc.get_paths()[0].vertices[:, 0])
            #pc.get_paths()[0].vertices[:, 0] = numpy.clip(pc.get_paths()[0].vertices[:, 0], -numpy.inf, m)
        ax.scatter(range(len(xs)), [numpy.average(results[k][cat]) for k in xs],zorder=3, color='white', marker='_')
        ax.set_ylabel('{}'.format(metric))
        ax.set_title(title)
        ### p-values
        p_vals = list()
        for k_one, v_one in results.items():
            one = v_one[cat]
            for k_two, v_two in results.items():
                two = v_two[cat]
                if k_one == k_two:
                    continue
                key = tuple(sorted([k_one, k_two]))
                if key not in [p[0] for p in p_vals]:
                    p = scipy.stats.ttest_ind(
                                          one, 
                                          two, 
                                          #permutations=4096, 
                                          #alternative=direction,
                                          ).pvalue
                    p_vals.append([key, p])
        ### fdr correction
        correct_ps = mne.stats.fdr_correction([p[1] for p in p_vals])[1]
        with open(os.path.join(cat_folder, '{}_{}_p-vals_comparisons.tsv'.format(cat, metric)), 'w') as o:
            o.write('comparison\tuncorrected_p-value\tFDR-corrected_p-value\n')
            for a, b in zip(p_vals, correct_ps):
                o.write('{}\t{}\t{}\n'.format(a[0], a[1], b))

        pyplot.savefig(os.path.join(cat_folder, '{}_{}.jpg'.format(cat, metric)))
        pyplot.clf()
        pyplot.close()

### temporal analysis
temporal_folder = os.path.join(out_folder, 'temporal')
os.makedirs(temporal_folder, exist_ok=True)
### plotting overall averages
fig, ax = pyplot.subplots(constrained_layout=True)
title = 'Across-categories averages for temporal correlations with RTs'
#xs = list(results.keys())
xs = ['IFG', 'preSMA', 'dual', 'sham']
ys = [[val for v in temporal_correlations[k].values() for val in v] for k in xs]
parts = ax.violinplot(ys, positions=range(len(ys)), showmeans=True, showextrema=False,)
ax.set_xticks(range(len(xs)))
ax.set_xticklabels(xs, fontweight='bold')
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(1)
    #m = numpy.mean(pc.get_paths()[0].vertices[:, 0])
    #pc.get_paths()[0].vertices[:, 0] = numpy.clip(pc.get_paths()[0].vertices[:, 0], -numpy.inf, m)
ax.scatter(range(len(xs)), [numpy.average([val for v in temporal_correlations[k].values() for val in v]) for k in xs],zorder=3, color='white', marker='_')
ax.set_ylabel('Across-categories average temporal correlation with RTs')
ax.set_title(title)
### p-values
p_vals = list()
for k_one, v_one in temporal_correlations.items():
    one = [val for v in v_one.values() for val in v]
    for k_two, v_two in temporal_correlations.items():
        two = [val for v in v_two.values() for val in v]
        if k_one == k_two:
            continue
        key = tuple(sorted([k_one, k_two]))
        if key not in [p[0] for p in p_vals]:
            p = scipy.stats.ttest_ind(
                                  one, 
                                  two, 
                                  #permutations=4096, 
                                  #alternative=direction,
                                  ).pvalue
            p_vals.append([key, p])
### fdr correction
correct_ps = mne.stats.fdr_correction([p[1] for p in p_vals])[1]
with open(os.path.join(temporal_folder, 'temporal_p-vals_comparisons.tsv'), 'w') as o:
    o.write('comparison\tuncorrected_p-value\tFDR-corrected_p-value\n')
    for a, b in zip(p_vals, correct_ps):
        o.write('{}\t{}\t{}\n'.format(a[0], a[1], b))

pyplot.savefig(os.path.join(temporal_folder, 'temporal_average.jpg'))
pyplot.clf()
pyplot.close()
