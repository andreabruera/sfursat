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
#ft = fasttext.load_model(os.path.join('data', 'cc.de.300.bin'))
ft = fasttext.load_model(os.path.join('..', '..', 'dataset', 'word_vectors', 'de', 'cc.de.300.bin'))
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
            switches[cond][cat].append(switches_and_clusters(words, vecs, thresholds[cat])[0])
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
colors = ['grey', '#0072B2', '#E69F00', '#D55E00']
scatter_colors = ['silver', 'darkturquoise', 'wheat', 'darkorange']
colors_dict = {
               'IFG' : '#0072B2',
               'preSMA' : '#E69F00',
               'dual' : '#D55E00',
               'sham' : 'grey',
               }



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


for marker in ['less_bars', 'more_bars']:
    out_folder = os.path.join('plots', 'semantic_fluency', marker)
    os.makedirs(out_folder, exist_ok=True)

    ### bar + individual points
    for metric, results in [
                            #('CuRel', curels), 
                            #('SeqRel', seqrels), 
                            ('switches', switches), 
                            ('RT', all_rts), 
                            ('log(1+RT)', log_all_rts),
                            ]:
        plot_results = {area : {k : numpy.average(v) for k, v in a_results.items()} for area, a_results in results.items()}
        ### plotting overall averages
        fig, ax = pyplot.subplots(constrained_layout=True)
        title = 'Average {} across semantic categories'.format(metric)
        xs = ['sham', 'IFG', 'preSMA', 'dual']
        ### average category scatters
        ys = [[val for _, val in plot_results[k].items()] for k in xs]
        line_y = numpy.average(ys[xs.index('dual')])+(3*numpy.std(ys[xs.index('dual')]))
        ys = [{_ : val for _, val in plot_results[k].items()} for k in xs]
        corrections = {cat : random.randrange(-300, 300)/1000 for cat in results['sham'].keys()}
        for i in range(len(xs)):
            ax.scatter(
                       [i+corrections[cat] for cat, y in ys[i].items()],
                       [y for cat, y in ys[i].items()],
                       color=scatter_colors[i],
                       edgecolors='white',
                       alpha=0.7,
                       zorder=2.5,
                       s=30,
                       )
        ### drawing lines
        combs = [
                 ('sham', 'IFG', 0, 1),
                 ('IFG', 'preSMA', 1, 2),
                 ('preSMA', 'dual', 2, 3),
                 ]
        for cat, corr in corrections.items():
            for one, two, pos_one, pos_two in combs:
                ones = ys[xs.index(one)]
                twos = ys[xs.index(two)]
                ax.plot(
                        [pos_one+corrections[cat], pos_two+corrections[cat]],
                        [ones[cat], twos[cat]],
                        alpha=0.1,
                        color='black',
                        zorder=2.
                        )
        ys = [[val for _, val in plot_results[k].items()] for k in xs]
        '''
        ### all subjects scatters
        ys = [[val for _, v in results[k].items() for val in v] for k in xs]
        line_y = max(ys[xs.index('sham')])-scipy.stats.sem(ys[xs.index('sham')])
        for i in range(len(xs)):
            ax.scatter(
                       [numpy.array(i)+corrections for y in ys[i]],
                       ys[i],
                       color=scatter_colors[i],
                       edgecolors='white',
                       alpha=0.7,
                       zorder=3.
                       )
        ### no scatter + clip
        ys = [[val for _, val in plot_results[k].items()] for k in xs]
        line_y = numpy.average(ys[xs.index('sham')])+(3*scipy.stats.sem(ys[xs.index('sham')]))
        ### no scatter no clip
        ys = [[val for _, val in plot_results[k].items()] for k in xs]
        line_y = numpy.average(ys[xs.index('sham')])+(3*scipy.stats.sem(ys[xs.index('sham')]))
        '''
        ### bar
        for i in range(len(xs)):
            ax.bar(
                   i,
                   numpy.average(ys[i]),
                   color=colors[i],
                   zorder=1.5
                   )
            if i==0:
                label='SEM'
            else:
                label=''
            ax.errorbar(
                   i,
                   numpy.average(ys[i]),
                   color='black',
                   capsize=5.,
                   yerr=scipy.stats.sem(ys[i]),
                   label=label,
                   zorder=3.
                   )
        ### clipping?
        #ax.set_ylim(
        #            ymin=numpy.average(ys[xs.index('dual')])-(2*numpy.std(ys[xs.index('dual')])),
        #            ymax=numpy.average(ys[xs.index('dual')])+(2*numpy.std(ys[xs.index('dual')]))
        #            )
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontweight='bold')
        ax.set_ylabel(
                      'Average {}'.format(metric),
                      )
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
                    if 'RT' in metric:
                        alternative = 'less' if k_one=='sham' else 'greater'
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
        ### plotting ps if significant
        counter = 1
        for p in p_vals:
            key_one = p[0][0]
            key_two = p[0][1]
            p_val = p[1]
            if p_val < 0.05:
                v = scipy.stats.sem(ys[xs.index('sham')])*.5
                pos_v = v*counter*4
                start = xs.index(key_one)
                end = xs.index(key_two)
                #print([start, end])
                ax.plot(
                        [start, start],
                        [line_y-pos_v-v, line_y-pos_v-(v*2)],
                        color='black',
                        )
                ax.plot(
                        [end, end],
                        [line_y-pos_v-v, line_y-pos_v-(v*2)],
                        color='black',
                        )
                ax.plot(
                        [start, end],
                        [line_y-pos_v-v, line_y-pos_v-v],
                        color='black',
                        )
                if counter == 1:
                    label = 'p<0.05'
                else:
                    label = ''
                ax.scatter(
                        (start+end)*.5,
                        line_y-pos_v+(v*.5),
                        color='black',
                        marker='*',
                        label=label,
                        )
                counter += 1
        ax.legend()
        ### fdr correction
        correct_ps = mne.stats.fdr_correction([p[1] for p in p_vals])[1]
        with open(os.path.join(txt_folder, '{}_p-vals_comparisons.tsv'.format(metric)), 'w') as o:
            o.write('comparison\tuncorrected_p-value\tFDR-corrected_p-value\taverages\n')
            for a, b, c in zip(p_vals, correct_ps, avgs):
                o.write('{}\t{}\t{}\t{}\n'.format(a[0], a[1], b, c))

        pyplot.savefig(
                       os.path.join(out_folder, '{}_average.jpg'.format(metric)),
                       dpi=300,)
        pyplot.clf()
        pyplot.close()

    ### scores X category difficulty
    xs = [k[0] for k in sorted(difficulties.items(), key=lambda item : item[1], reverse=True)]

    for metric, results in [
                            #('CuRel', curels), 
                            #('SeqRel', seqrels), 
                            ('switches', switches), 
                            ('RT', all_rts), 
                            ('log(1+RT)', log_all_rts),
                            ]:
        ### plotting overall averages
        fig, ax = pyplot.subplots(constrained_layout=True, figsize=(16, 9))
        title = 'Average {} per category'.format(metric)
        dual_ys = [numpy.average(results['dual'][k]) for k in xs]
        sham_ys = [numpy.average(results['sham'][k]) for k in xs]
        test_dual_ys = [results['dual'][k] for k in xs]
        test_sham_ys = [results['sham'][k] for k in xs]
        min_val = min(sham_ys+dual_ys)-scipy.stats.sem(sham_ys)
        if 'RT' in metric:
            alt = 'greater'
        else:
            alt = 'less'
        ps = list()
        for one, two in zip(test_dual_ys, test_sham_ys):
            p = scipy.stats.ttest_ind(
                                  one,
                                  two,
                                  alternative=alt,
                                  ).pvalue
            ps.append(p)
        #print([metric, ps])
        assert numpy.array(ps).shape == numpy.array(xs).shape
        ifg_ys = [numpy.average(results['IFG'][k]) for k in xs]
        presma_ys = [numpy.average(results['preSMA'][k]) for k in xs]
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(
                           xs, 
                           fontweight='bold', 
                           rotation=45,
                           fontsize=15,
                           ha='right',
                           )
        ax.set_ylabel(
                      'Average {}'.format(metric),
                      fontsize=20,
                      )
        ax.set_xlabel(
                      'Categories (easier -> harder)',
                      fontsize=20,
                      )
        ax.set_title(
                     title,
                     fontsize=23.,
                     )
        ### ps
        ps_five_xs = [i for i, p in enumerate(ps) if p<=0.05]
        if len(ps_five_xs) > 0:
            #print(ps_five_xs)
            ax.scatter(
                   ps_five_xs, 
                   [min_val for i in ps_five_xs], 
                   color='black', 
                   marker='*', 
                   s=60,
                   label='p<0.05'
                   )
        ps_approach_xs = [i for i, p in enumerate(ps) if p<=0.1 and p>0.05]
        if len(ps_approach_xs) > 0:
            ax.scatter(
                   ps_approach_xs, 
                   [min_val for i in ps_approach_xs], 
                   color='black', 
                   marker='^', 
                   s=60,
                   label='p<0.1'
                   )
        ps_approach_xs = [i for i, p in enumerate(ps) if p<=0.2 and p>0.1]
        if len(ps_approach_xs) > 0:
            ax.scatter(
                   ps_approach_xs, 
                   [min_val for i in ps_approach_xs], 
                   color='black', 
                   marker='2', 
                   s=60,
                   label='p<0.2'
                   )
        ### dual
        ax.plot(range(len(xs)), dual_ys, color=colors_dict['dual'], label='dual')
        ax.scatter(
                   range(len(xs)), 
                   dual_ys, 
                   color=colors_dict['dual'], 
                   marker='s', 
                   edgecolors='white', 
                   #linewidths=15,
                   s=50,
                   zorder=3.
                   )
        '''
        ### IFG
        ax.plot(range(len(xs)), ifg_ys, color=colors_dict['IFG'], label='IFG', alpha=0.6,)
        ax.scatter(
                   range(len(xs)), 
                   ifg_ys, 
                   color=colors_dict['IFG'], 
                   marker='8', 
                   edgecolors='white', 
                   #linewidths=15,
                   alpha=0.6,
                   s=50,
                   zorder=3.
                   )
        ### preSMA
        ax.plot(range(len(xs)), presma_ys, color=colors_dict['preSMA'], label='preSMA', alpha=0.6,)
        ax.scatter(
                   range(len(xs)), 
                   presma_ys, 
                   color=colors_dict['preSMA'], 
                   marker='v', 
                   edgecolors='white', 
                   #linewidths=15,
                   alpha=0.6,
                   s=50,
                   zorder=3.
                   )
        '''
        ### sham
        ax.plot(range(len(xs)), sham_ys, color=colors_dict['sham'], label='sham', linestyle='--')
        ax.scatter(
                   range(len(xs)), 
                   sham_ys, 
                   color=colors_dict['sham'], 
                   marker='D', 
                   edgecolors='white', 
                   #linewidths=15,
                   s=50,
                   zorder=3.,
                   )
        ax.legend(fontsize=23)
        pyplot.savefig(os.path.join(out_folder, '{}_difficulties.jpg'.format(metric)))
        pyplot.clf()
        pyplot.close()
