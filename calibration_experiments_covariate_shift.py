import torch
from os.path import join
import numpy as np
from model.classifier_calibrators import *
from util import cal_error
from scipy.special import softmax

source = 'imdb'
target = 'imdb'
model = 'distilbert-base-uncased'

path=f'./neural_training/embeds/{source}/distilbert-base-uncased'

train_hidden = torch.load(join(path,'source.train.hidden_states.pt')).numpy()
train_logits = torch.load(join(path,'source.train.logits.pt')).numpy()
train_y = torch.load(join(path,'source.train.labels.pt')).numpy()

valid_hidden = torch.load(join(path,'source.validation.hidden_states.pt')).numpy()
valid_logits = torch.load(join(path,'source.validation.logits.pt')).numpy()
valid_y = torch.load(join(path,'source.validation.labels.pt')).numpy()

if target==source:
    target_prefix='source'
else:
    target_prefix=f'target_{target}'
# test_hidden = torch.load(join(path,'source.test.hidden_states.pt')).numpy()
# test_logits = torch.load(join(path,'source.test.logits.pt')).numpy()
# test_y = torch.load(join(path,'source.test.labels.pt')).numpy()
test_hidden = torch.load(join(path,f'{target_prefix}.test.hidden_states.pt')).numpy()
test_logits = torch.load(join(path,f'{target_prefix}.test.logits.pt')).numpy()
test_y = torch.load(join(path,f'{target_prefix}.test.labels.pt')).numpy()

np.random.seed(0)
rand_order = np.random.permutation(len(test_y))
sel = rand_order[:1000]
test_hidden=test_hidden[sel]
test_logits=test_logits[sel]
test_y=test_y[sel]

print(f'#train={len(train_y)}')
print(f'#valid={len(valid_y)}')
print(f'#test={len(test_y)}')

ece_before = cal_error(test_logits, test_y, arelogits=True)

# valid_posteriors = softmax(valid_logits, axis=1)
# test_posteriors = softmax(test_logits, axis=1)
# calibrator = EM(train_prevalence=np.mean(train_y))
# calibrator = PACCcal(softmax(valid_logits, axis=1), valid_y)
# calibrator = TransCalCalibrator(prob2logits=False)
# calibrator = HeadToTailCalibrator(prob2logits=False)
# calibrator = CpcsCalibrator(prob2logits=False)
calibrator = LasCalCalibration(prob2logits=False)

if isinstance(calibrator, CalibratorCompound):
    calib_posteriors = calibrator.calibrate(
        Ftr=train_hidden, ytr=train_y,
        Fsrc=valid_hidden, Zsrc=valid_logits, ysrc=valid_y,
        Ftgt=test_hidden, Ztgt=test_logits
    )
elif isinstance(calibrator, CalibratorSourceTarget):
    calib_posteriors = calibrator.calibrate(
        Zsrc=valid_logits, ysrc=valid_y, Ztgt=test_logits
    )
elif isinstance(calibrator, CalibratorSimple):
    calib_posteriors = calibrator.calibrate(P=softmax(test_logits, axis=1))

ece_after = cal_error(calib_posteriors, test_y, arelogits=False)

print(f'{ece_before=:.5f}')
print(f'{ece_after=:.5f}')




