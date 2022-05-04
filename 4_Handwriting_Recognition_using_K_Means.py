import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

# fig = plt.figure(figsize=(6, 6))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# for i in range(64):
#     ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
#     ax.text(0, 7, str(digits.target[i]))
# plt.show()

model = KMeans(n_clusters = 10, random_state=42)
model.fit(digits.data)

fig2 = plt.figure(figsize=(8, 3))
fig2.suptitle('centroids')
for i in range(10):
  ax = fig2.add_subplot(2, 5, 1 + i,  xticks=[], yticks=[])
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
  ax.text(0, 7, 'i'+str(i))
plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.38,5.61,10.59,5.87,0.00,0.00,0.00,0.00,10.21,11.48,11.97,7.13,0.00,0.00,0.00,0.00,1.53,0.51,11.09,4.84,0.00,0.00,0.00,0.00,0.00,0.00,12.75,2.93,0.00,0.00,0.00,0.00,0.00,0.77,12.75,2.17,0.00,0.00,0.00,0.00,0.00,1.28,12.75,1.28,0.00,0.00,0.00,0.00,0.00,0.00,4.08,0.00,0.00,0.00],
[0.00,0.38,6.88,6.25,3.83,6.38,6.63,0.13,0.00,6.51,11.86,11.35,11.48,10.71,12.75,2.43,0.00,0.77,1.02,0.00,0.00,0.00,12.76,2.56,0.00,0.00,0.00,0.00,0.00,1.79,12.75,1.91,0.00,0.00,0.00,0.00,2.68,11.86,10.84,0.26,0.00,0.00,0.00,4.33,11.99,7.14,0.77,0.00,0.00,1.28,10.07,12.62,12.62,8.42,4.59,0.13,0.00,8.80,12.75,9.05,6.38,8.03,12.24,6.50],
[0.00,0.00,5.49,6.12,4.72,3.32,0.26,0.00,0.00,2.04,12.75,9.69,10.97,12.50,5.61,0.00,0.00,1.79,12.75,5.23,2.81,9.69,6.37,0.00,0.00,0.00,7.65,11.73,12.75,12.75,7.39,0.00,0.00,0.00,0.00,0.00,0.00,6.76,10.20,0.00,0.00,0.00,2.68,0.13,0.00,5.10,10.20,0.00,0.00,0.00,12.62,4.59,1.53,9.69,9.70,0.00,0.00,0.00,8.93,12.75,12.75,10.96,1.66,0.00],
[0.00,0.00,0.00,1.28,0.26,0.00,0.00,0.00,0.00,0.00,2.68,12.24,3.96,0.00,0.00,0.00,0.00,0.25,10.71,8.41,4.09,0.00,0.00,0.00,0.00,5.74,12.49,8.41,12.75,9.31,0.00,0.00,0.00,10.20,12.50,12.50,10.33,2.55,0.00,0.00,0.00,0.00,0.00,11.10,4.97,0.00,0.00,0.00,0.00,0.00,1.53,12.37,1.15,0.00,0.00,0.00,0.00,0.00,0.00,0.51,0.00,0.00,0.00,0.00]
])

fig3 = plt.figure(figsize=(8, 3))
fig3.suptitle('new_samples')
for i in range(len(new_samples)):
  ax = fig3.add_subplot(2, 5, 1 + i,  xticks=[], yticks=[])
  ax.imshow(new_samples[i].reshape((8, 8)), cmap=plt.cm.binary)
  ax.text(0, 7, 'i'+str(i))
plt.show()

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(3, end='')
  elif new_labels[i] == 1:
    print(0, end='')
  elif new_labels[i] == 2:
    print(8, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(9, end='')
  elif new_labels[i] == 5:
    print(2, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(7, end='')
  elif new_labels[i] == 8:
    print(6, end='')
  elif new_labels[i] == 9:
    print(5, end='')
