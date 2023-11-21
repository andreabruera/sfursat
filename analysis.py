import fasttext
import numpy
import os
import re
import scipy

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
                                cat : list() for cat in set(full_dataset['item'])
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
                                cat : list() for cat in set(full_dataset['item'])
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
    cat = full_dataset['item'][row]
    rt = float(full_dataset['rt'][row])
    word = full_dataset['response'][row].strip()
    rts[sub][task][cond][cat].append(rt)
    fluencies[sub][task][cond][cat].append(word)
import pdb; pdb.set_trace()
