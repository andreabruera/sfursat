import matplotlib
import mne
import numpy
import os
import pickle
import random
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats

txt_folder = os.path.join('results', 'semantic_fluency')

with open(os.path.join('pkls', 'switches.pkl'), 'rb') as i:
    switches = pickle.load(i)

with open('switches.csv', 'w') as o:
    o.write('condition,subject,category,switches\n')
    for cond, cond_data in switches.items():
        for cat, cat_data in cond_data.items():
            for sub, swtc in cat_data:
                o.write('{},{},{},{}\n'.format(cond, sub, cat, swtc))
switches = {_ : {__ : [v[1] for v in vs] for __, vs in conds.items()} for _, conds in switches.items()}

### Font setup
# Using Helvetica as a font
font_folder = '../fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'
colors = ['grey', '#0072B2', '#E69F00', '#D55E00']
scatter_colors = ['silver', 'darkturquoise', 'wheat', 'darkorange']
colors = ['silver', 'darkturquoise', 'wheat', 'darkorange']
colors_dict = {
               #'IFG' : '#0072B2',
               'IFG' : '#8491af',
               #'preSMA' : '#E69F00',
               'preSMA' : '#626f8c',
               #'dual' : '#D55E00',
               'dual' : '#424f6a',
               #'sham' : 'grey',
               'sham' : '#a8b4d4',
               }
scatter_colors = ["#a8b4d4", "#8491af", "#626f8c", "#424f6a"]
colors = ["#a8b4d4", "#8491af", "#626f8c", "#424f6a"]

for marker in ['less_bars', 'more_bars']:
    out_folder = os.path.join('plots', 'semantic_fluency', marker)
    os.makedirs(out_folder, exist_ok=True)

    ### bar + individual points
    for metric, results in [
                            #('CuRel', curels), 
                            #('SeqRel', seqrels), 
                            ('switches', switches), 
                            #('RT', all_rts), 
                            #('log(1+RT)', log_all_rts),
                            ]:
        plot_results = {area : {k : numpy.average(v) for k, v in a_results.items()} for area, a_results in results.items()}
        ### plotting overall averages
        fig, ax = pyplot.subplots(constrained_layout=True, figsize=(4, 5))
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
        if marker == 'more_bars':
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
                   zorder=1.5,
                   #width=0.4,
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
        ax.set_xticklabels([x[0].capitalize()+x[1:] for x in xs],)
        #fontweight='bold')
        ax.set_ylabel(
                      'Average {}'.format(metric),
                      fontsize=12,
                      )
        #ax.set_title(title)
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
        ax.legend(ncols=2, loc=9, framealpha=0.95)
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
                            #('RT', all_rts), 
                            #('log(1+RT)', log_all_rts),
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
