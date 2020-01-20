import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import auc

#use to plot ROC curve 

# load model
model = load_model('model.h5')
# summarize model.
model.summary()

#test model
video = 'Muppets-02-04-04'
X = np.load('images_'+video+'.npy')#[0:10]
y = np.load('labels_'+video+'.npy')#[0:10]

y_pred = model.predict(X).argmax(axis=1)#.ravel()
print(y_pred)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Classifier (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

