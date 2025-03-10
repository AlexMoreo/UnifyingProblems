from glob import glob
import pandas as pd
from util import count_successes


input_dir = './results/calibration/label_shift/repeats_100'

results = pd.concat([pd.read_csv(result_file) for result_file in glob(input_dir+'/*.csv')])

for classifier in ['nb']:#, 'mlp', 'nb']:
    results_sel = results[results['classifier']==classifier]
    counts = count_successes(results_sel, baselines=['CPCS-S', 'Head2Tail-S', 'LasCal-S', 'TransCal-S'], value='ece')
    for method, count in counts.items():
        print(f'{method=}:')
        for times, perc in count.items():
            if times == 0 or times=='ave': continue
            print(f'\t>{times} baselines: {perc*100:.2f}%')
        print(f'\tAVE={count["ave"]:.3f}')

