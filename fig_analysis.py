import charsplit
import fasttext
import matplotlib
import mne
import numpy
import os
import pickle
import re
import scipy
import sklearn
import spacy

from matplotlib import font_manager, pyplot
from mne import stats
from scipy import spatial, stats
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from utils import curel, seqrel, switches_and_clusters, temporal_analysis
from utf_utils import transform_german_word

### read dataset
lines = list()
with open(os.path.join('data', 'fig_fluency_all.tsv')) as i:
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
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                } 
                         for cond in set(full_dataset['cond'])
                         } 
                     for sub in set(full_dataset['participant'])
                     }
fluencies = {int(sub) : {
                         cond : {
                                #cat : list() for cat in set(full_dataset['item'])
                                } 
                         for cond in set(full_dataset['cond'])
                         } 
                     for sub in set(full_dataset['participant'])
                     }

vecs = dict()
vocab = set()
for row in tqdm(range(total_rows)):
    sub = int(full_dataset['participant'][row])
    cond = full_dataset['cond'][row]
    rt = float(full_dataset['rt'][row])
    cat = full_dataset['design'][row]
    fig = ''.join([f.strip().replace('bar_{}_'.format(cat), '') for f in full_dataset['name'][row].split(',')])
    for f in fig:
        vocab.add(f)
    if cat not in rts[sub][cond].keys():
        rts[sub][cond][cat] = list()
        fluencies[sub][cond][cat] = list()
    rts[sub][cond][cat].append(rt)
    fluencies[sub][cond][cat].append(fig)

curels = {cond : dict() for cond in set(full_dataset['cond'])}
seqrels = {cond : dict() for cond in set(full_dataset['cond'])}
switches = {cond : dict() for cond in set(full_dataset['cond'])}
temporal_correlations = {cond : dict() for cond in set(full_dataset['cond'])}

### computing thresholds
thresholds = {'overall' : list()}
cond_thresholds = dict()
subject_thresholds = {s : {'overall' : list()} for s in fluencies.keys()}
for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data.items():
        for cat, words in cond_data.items():
            thresholds['overall'].extend(seqrel(words, vecs, mode='fig'))
            subject_thresholds[_]['overall'].extend(seqrel(words, vecs, mode='fig'))
            if cat not in thresholds.keys():
                thresholds[cat] = list()
            if cat not in subject_thresholds.keys():
                subject_thresholds[_][cat] = list()
            if cond not in cond_thresholds.keys():
                cond_thresholds[cond] = dict()
                cond_thresholds[cond]['overall'] = list()
            if cat not in cond_thresholds[cond].keys():
                cond_thresholds[cond][cat] = list()
            thresholds[cat].extend(seqrel(words, vecs,mode='fig'))
            subject_thresholds[_][cat].extend(seqrel(words, vecs,mode='fig'))
            cond_thresholds[cond]['overall'].extend(seqrel(words, vecs,mode='fig'))
            cond_thresholds[cond][cat].extend(seqrel(words, vecs,mode='fig'))
thresholds = {k : numpy.median(v) for k, v in thresholds.items()}
subject_thresholds = {s : {k : numpy.median(v) for k, v in thresh.items()} for s , thresh in subject_thresholds.items()}
cond_thresholds = {s : {k : numpy.median(v) for k, v in thresh.items()} for s , thresh in cond_thresholds.items()}


for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data.items():
        for cat, words in cond_data.items():
            if cat not in curels[cond].keys():
                curels[cond][cat] = list()
                seqrels[cond][cat] = list()
                switches[cond][cat] = list()
                temporal_correlations[cond][cat] = list()
            curels[cond][cat].append(numpy.nanmean(curel(words, vecs,mode='fig')))
            seqrels[cond][cat].append(numpy.nanmean(seqrel(words, vecs,mode='fig')))
            ### overall threshold as in Kim et al. 2019
            #switches[cond][cat].append(switches_and_clusters(words, vecs, thresholds['overall'], mode='fig')[0])
            ### category-specific threshold as Ocalam et al. 2022
            switches[cond][cat].append(switches_and_clusters(words, vecs, thresholds[cat],mode='fig')[0])
            current_rts = rts[_][cond][cat]
            temporal_correlations[cond][cat].append(temporal_analysis(words, vecs, current_rts,mode='fig'))

### Font setup
# Using Helvetica as a font
font_folder = '/import/cogsci/andrea/dataset/fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'
colors = ['teal', 'goldenrod', 'magenta', 'grey']
colors_dict = {
               'IFG' : 'teal', 
               'preSMA' : 'goldenrod', 
               'dual' : 'magenta', 
               'sham' : 'grey',
               }
out_folder = os.path.join('plots', 'figural_fluency')
os.makedirs(out_folder, exist_ok=True)

### violin plot
for metric, results in [('CuRel', curels), ('SeqRel', seqrels), ('Switches', switches)]:
    overall_folder = os.path.join(out_folder, 'overall')
    os.makedirs(overall_folder, exist_ok=True)
    ### plotting overall averages
    fig, ax = pyplot.subplots(constrained_layout=True)
    title = 'Figural fluency - across-categories averages for {}'.format(metric)
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
    avgs = list()
    medians = list()
    for k_one, v_one in results.items():
        one = [val for v in v_one.values() for val in v]
        for k_two, v_two in results.items():
            two = [val for v in v_two.values() for val in v]
            if k_one == k_two:
                continue
            key = [k_one, k_two]
            if 'sham' in key:
                alternative = 'greater' if k_one=='sham' else 'less'
                if sorted(key) not in [sorted(p[0]) for p in p_vals]:
                    p = scipy.stats.ttest_ind(
                                          one, 
                                          two, 
                                          #permutations=4096, 
                                          alternative=alternative,
                                          ).pvalue
                    p_vals.append([[k_one, k_two], round(p, 4)])
                    avgs.append([round(numpy.average(one), 3), round(numpy.average(two), 3)])
                    medians.append([numpy.median(one), numpy.median(two)])
    ### fdr correction
    correct_ps = mne.stats.fdr_correction([p[1] for p in p_vals])[1]
    with open(os.path.join(overall_folder, 'fig_{}_p-vals_comparisons.tsv'.format(metric)), 'w') as o:
        o.write('comparison\tuncorrected_p-value\tFDR-corrected_p-value\taverages\n')
        for a, b, c in zip(p_vals, correct_ps, avgs):
            o.write('{}\t{}\t{}\t{}\n'.format(a[0], a[1], b, c))

    pyplot.savefig(os.path.join(overall_folder, 'fig_{}_average.jpg'.format(metric)))
    pyplot.clf()
    pyplot.close()

### temporal analysis
temporal_folder = os.path.join(out_folder, 'temporal')
os.makedirs(temporal_folder, exist_ok=True)
### plotting overall averages
fig, ax = pyplot.subplots(constrained_layout=True)
title = 'Figural fluency - across-categories averages for temporal correlations with RTs'
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
                                  permutations=4096, 
                                  #alternative=direction,
                                  ).pvalue
            p_vals.append([key, p])
### fdr correction
correct_ps = mne.stats.fdr_correction([p[1] for p in p_vals])[1]
with open(os.path.join(temporal_folder, 'fig_temporal_p-vals_comparisons.tsv'), 'w') as o:
    o.write('comparison\tuncorrected_p-value\tFDR-corrected_p-value\n')
    for a, b in zip(p_vals, correct_ps):
        o.write('{}\t{}\t{}\n'.format(a[0], a[1], b))

pyplot.savefig(os.path.join(temporal_folder, 'fig_temporal_average.jpg'))
pyplot.clf()
pyplot.close()
