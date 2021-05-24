from sklearn.svm import SVC
import pandas as pd
from util import get_input_label_split, get_accuracy, get_precision
from SVM import SVM

df = pd.read_csv('../heart.csv')
train_sz = int(len(df) * 0.8)
df_train = df[:train_sz]
df_test = df[train_sz:]

clf1 = SVM(max_iter=10000, kernel_type='rbf', C=1.0, epsilon=0.001)

clf2 = SVC(kernel='linear')
train_x, train_y = get_input_label_split(df_train, 'exng')


clf1.fit(train_x, train_y)
clf2.fit(train_x, train_y)

test_x, test_y = get_input_label_split(df_test, 'exng')


pred1 = clf1.predict(test_x)

print(f"accuracy score for test SVM: {get_accuracy(pred1, test_y)}")
print(f"precision score for test SVM: {get_precision(pred1, test_y)}")

pred2 = clf2.predict(test_x)

print(f"accuracy score for SVC: {get_accuracy(pred2, test_y)}")
print(f"precision score for SVC: {get_precision(pred2, test_y)}")