import fasttext
import matplotlib
import mne
import numpy
import os
import re
import scipy
import spacy

from matplotlib import font_manager, pyplot
from mne import stats
from scipy import spatial
from tqdm import tqdm

from utils import curel, seqrel, switches_and_clusters

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
for row in range(total_rows):
    sub = int(full_dataset['participant'][row])
    task = full_dataset['task'][row]
    cond = full_dataset['cond'][row]
    if cond == 'NA':
        import pdb; pdb.set_trace()
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
ft = fasttext.load_model(os.path.join('..', '..', 'dataset', 'word_vectors', 'de', 'cc.de.300.bin'))
for row in tqdm(range(total_rows)):
    task = full_dataset['task'][row]
    if 'sem' in task:
        word = full_dataset['response'][row].strip()
        lemma = ' '.join([w.lemma_ for w in spacy_model(word)]).lower()
        word_vecs[word] = ft.get_word_vector(word)
        lemma_vecs[word] = ft.get_word_vector(lemma)
        #print(1-scipy.spatial.distance.cosine(word_vecs[word], lemma_vecs[word]))
vecs = {w : numpy.average([word_vecs[w], lemma_vecs[w]], axis=0) for w in word_vecs.keys()}

curels = {cond : dict() for cond in set(full_dataset['cond'])}
seqrels = {cond : dict() for cond in set(full_dataset['cond'])}
switches = {cond : dict() for cond in set(full_dataset['cond'])}

for _, sub_data in tqdm(fluencies.items()):
    for cond, cond_data in sub_data['sem_fluency'].items():
        for cat, words in cond_data.items():
            if cat not in curels[cond].keys():
                curels[cond][cat] = list()
                seqrels[cond][cat] = list()
                switches[cond][cat] = list()
            curels[cond][cat].append(numpy.nanmean(curel(words, vecs)))
            seqrels[cond][cat].append(numpy.nanmean(seqrel(words, vecs)))
            switches[cond][cat].append(switches_and_clusters(words, vecs)[0])

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
os.makedirs(out_folder, exist_ok=True)

### plotting overall averages
for metric, results in [('CuRel', curels), ('SeqRel', seqrels), ('Switches', switches)]:
    fig, ax = pyplot.subplots(constrained_layout=True)
    title = 'Across-categories averages for {}'.format(metric)
    xs = list(results.keys())
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
    ax.set_ylabel('Across-categories average {}'.format(metric))
    ax.set_title(title)
    pyplot.savefig(os.path.join(out_folder, '{}_average.jpg'.format(metric)))
    pyplot.clf()
    pyplot.close()
