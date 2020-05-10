from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from operator import mul
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB # or any other NB model
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def pre_processing(line):
    modified_line = []
    final_line = []
    for w in word_tokenize(line):
        if w not in stop_words:
            modified_line.append(w)
    for word in modified_line:
        final_line.append(porter.stem(word))
    return final_line


count_vect = CountVectorizer()
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update(["I", "?", ".", ";", "'", ",", "[", "]", "(", ")"])
file = pd.read_csv("classes_labeled.csv", index_col=0)
column_names = ['Positive', 'Negative', 'Neutral']
for name in column_names:
    file[name] = file[name].apply(lambda x: pre_processing(x))

# file.to_csv("processed_labeled.csv")
train, test = train_test_split(file, test_size=0.2)
new_df = file.stack().reset_index()
del new_df['level_0']
new_df.columns = ['Class', 'Document']
new_df['Class'] = new_df['Class'].apply(lambda x: 0 if x == 'Positive' else 1 if x == 'Negative' else 2)
new_df['Document'] = new_df['Document'].apply(lambda x: ' '.join(x))
x = new_df['Document']
y = new_df['Class']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
# train_x_counts = count_vect.fit_transform(train_x)
# test_x_counts = count_vect.transform(test_x)
object_list = []
svm0 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1))])
object_list.append(svm0)
svm01 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1, gamma = 0.001))])
object_list.append(svm01)
svm1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1))])
object_list.append(svm1)
svm11 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1, gamma=0.001))])
object_list.append(svm11)
svm2 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=10))])
object_list.append(svm2)
svm21 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=10, gamma=0.001))])
object_list.append(svm21)
svm3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=10))])
object_list.append(svm3)
svm31 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=10, gamma = 0.001))])
object_list.append(svm31)
svm4 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=100))])
object_list.append(svm4)
svm41 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=100, gamma=0.001))])
object_list.append(svm41)
svm5 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=100))])
object_list.append(svm5)
svm51 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=100, gamma = 0.001))])
object_list.append(svm51)
svm6 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1000))])
object_list.append(svm6)
svm61 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="linear", C=1000, gamma = 0.001))])
object_list.append(svm61)
svm7 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1000))])
object_list.append(svm7)
svm71 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('svm', SVC(kernel="rbf", C=1000, gamma = 0.001))])
object_list.append(svm71)
# x = [svm0,svm1, svm2, svm3, svm4, svm5]
# def highestSVM(x):
#     all_accuracies = {}
#     for item in x:
#         svm = item.fit(train_x,train_y)
#         y_pred = svm.predict(test_x)
#         all_accuracies[item] = accuracy_score(test_y, y_pred)
#     max_accuracy = max(all_accuracies, key=all_accuracies.get)
#     return max_accuracy
# x = [svm0, svm1, svm11, svm01, svm2, svm21]
print(svm11.steps[-1][1].gamma)
with open("Results.txt", "w+") as results:
    for item in object_list:
        svm = item.fit(train_x, train_y)
        # Predict the response for test dataset
        results.write("-"*30 + "\n")
        y_pred = item.predict(test_x)
        svc = item.steps[-1][1]
        kernel = svc.kernel
        c = svc.C
        gamma = svc.gamma
        results.write("Kernel = " + str(kernel) + ", C = " + str(c) + " and gamma = " + str(gamma) + "\n")
        results.write("Accuracy :" + str(accuracy_score(test_y, y_pred)) + "\n")
        target_names = ['Positive', 'Negative', 'Neutral']
        conf_mat = confusion_matrix(test_y, y_pred, labels=[0, 1, 2])
        results.write("Confusion matrix : \n")
        results.write(str(conf_mat) + "\n")
        results.write(str(classification_report(test_y, y_pred, target_names=target_names)) + "\n")
# positive_indices = []
# negative_indices = []
# neutral_indices = []
# for x, y in zip(list(test_x.index.values), list(y_pred)):
#     if y == 0:
#         positive_indices.append(x)
#     if y == 1:
#         negative_indices.append(x)
#     if y == 2:
#         neutral_indices.append(x)
# print(positive_indices)
# positive = new_df[new_df.index.isin(neutral_indices)]["Document"].values
# # train["Positive"] = train["Positive"].apply(lambda q: ' '.join(q))
# # positive = train["Positive"].values
# # print(positive)
# stop_words.update(STOPWORDS)
# stop_words.update(["book", "nook", "read", "n't", "kindl","camera","use","button","len","one"])
# positive_wc = WordCloud(width = 3000,
#     height = 2000,stopwords = stop_words, background_color='white').generate(str(positive))
# fig = plt.figure(figsize = (40, 30))
# plt.imshow(positive_wc, interpolation = 'bilinear')
# plt.axis('off')
# plt.tight_layout(pad = 0)
# plt.show()
