from glob import glob
import pandas as pd
from util import count_successes


input_dir = './results/quantification/label_shift/repeats_100_samplesize_250'

results = pd.concat([pd.read_csv(result_file) for result_file in glob(input_dir+'/*.csv')])

counts = count_successes(results, baselines=['Naive', 'CC', 'PCC', 'PACC', 'KDEy', 'EMQ'], value='ae')
print(counts)

