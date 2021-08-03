import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.feature_names)
#print(breast_cancer_data.target,breast_cancer_data.target_names)
#The very first data point tagged as malignant(0).

training_data , validation_data , training_labels , validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target,  test_size = 0.2, train_size = 0.8, random_state = 100)
#print(len(training_data),len(training_labels))

k = 0
accuracy = []
for i in range(100):
  k += 1
  classifier = KNeighborsClassifier(k)
  classifier.fit(training_data, training_labels)
  print(classifier.score(validation_data,validation_labels))
  accuracy.append(classifier.score(validation_data,validation_labels))
print('Maximal accuracy is ',max(accuracy),'its the ',(accuracy.index(max(accuracy))+1), 'k')

plt.plot(range(1,101), accuracy)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
