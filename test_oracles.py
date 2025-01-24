from sklearn.linear_model import LogisticRegression

from model import *
import quapy as qp

dataset=qp.datasets.UCI_BINARY_DATASETS[0]

data = qp.datasets.fetch_UCIBinaryDataset(dataset)
train, test = data.train_test

test = test.sampling(len(test), 0.8, 0.2, random_state=1)

Xtr, ytr = train.Xy
Xte, yte = test.Xy

# ---------------------------------------
print('-'*80)
print('CAP Oracles')
print('-'*80)
lr = LogisticRegression()
lr.fit(Xtr, ytr)

true_acc = (lr.predict(Xte)==yte).mean()

print(f'{true_acc=:.4f}')

ocap = OracleCAP(h=lr)
ocap_acc = ocap.predict_accuracy(Xte, yte)
print(f'{ocap_acc=:.4f}')

ocap_from_q = OracleCAPfromQuantification(h=lr, oracle_quantifier=OracleQuantifier())
ocap_from_qacc = ocap_from_q.predict_accuracy(Xte, yte)
print(f'{ocap_from_qacc=:.4f}')

ocap_from_c = OracleCAPfromCalibration(h=lr, oracle_calibrator=OracleCalibrator(h=lr))
ocap_from_c = ocap_from_c.predict_accuracy(Xte, yte)
print(f'{ocap_from_c=:.4f}')

# ---------------------------------------
print('-'*80)
print('Quantification Oracles')
print('-'*80)

true_prev = test.prevalence()
print(f'true_prev={true_prev[1]:.4f}')

oquant = OracleQuantifier()
oquant_prev = oquant.quantify(Xte, yte)
print(f'{oquant_prev=:.4f}')

oquant_from_cap = OracleQuantifierFromCAP(OracleCAP(h=lr))
oquant_from_cap_prev = oquant_from_cap.quantify(Xte, yte)
print(f'{oquant_from_cap_prev=:.4f}')

oquant_from_cal = OracleQuantifierFromCalibrator(OracleCalibrator(h=lr))
oquant_from_cal_prev = oquant_from_cal.quantify(Xte, yte)
print(f'{oquant_from_cal_prev=:.4f}')