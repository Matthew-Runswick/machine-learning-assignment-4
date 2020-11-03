import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import sklearn.preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from matplotlib.patches import Patch
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

#Part (i)
df = pd.read_csv("assignment4_data1.csv" , comment='#')
data1_x1=df.iloc[:,0]
data1_x2=df.iloc[:,1]
data1_y = df.iloc[:,2]
data1_X = np.column_stack((data1_x1,data1_x2))

#Part A

colours = np.where(data1_y==1,'r','b')
plt.scatter(data1_x1, data1_x2, c=colours)
plt.show()

P = [1, 2, 3, 4, 5, 6]
all_predictions = []
for i in P:
    poly = sklearn.preprocessing.PolynomialFeatures(degree=i)
    X_new_features_data1 = poly.fit_transform(data1_X)
    x_range_poly = poly.fit_transform(data1_X)

    C_values=[0.1, 1, 10, 100, 1000]
    mean_accuracy = []
    accuracy_std_error = []
    mean_F1 = []
    F1_std_error = []

    for c in C_values:
        new_model = LogisticRegression(penalty="l2", C=c)
        new_model.fit(X_new_features_data1, data1_y)
        predictions = new_model.predict(x_range_poly)
        scores = cross_val_score(new_model, x_range_poly, data1_y, cv=5, scoring="accuracy")
        mean_accuracy.append(np.array(scores).mean())
        accuracy_std_error.append(np.array(scores).std())
        scores = cross_val_score(new_model, x_range_poly, data1_y, cv=5, scoring="precision")
        mean_F1.append(np.array(scores).mean())
        F1_std_error.append(np.array(scores).std())
        print("model C = ", c)
        print("intercept ", new_model.intercept_)
        print("Coef ", new_model.coef_)
        all_predictions.append(predictions)


    fig = plt.figure()
    plt.title("C Value vs Model Accuracy and Precision, P={}".format(i))
    plt.errorbar(C_values, mean_accuracy, yerr=accuracy_std_error, c='r')
    plt.errorbar(C_values, mean_F1, yerr=F1_std_error, c='b')
    plt.xscale("log")
    plt.xlabel("C Values")
    plt.ylabel("Metric %")

plt.show()

col = []
col1 = []
col2 = []
col3 = []
for i in range(0, len(all_predictions[0])):
    if all_predictions[1][i] == -1: #P=1 C=1
        col.append('#FFFF00')    
    else:
        col.append('#00FFFF')

    if all_predictions[7][i] == -1: #P=2 C=10
        col1.append('#FFFF00')    
    else:
        col1.append('#00FFFF')

    if all_predictions[12][i] == -1: #P=3 C=10
        col2.append('#FFFF00')    
    else:
        col2.append('#00FFFF')

    if all_predictions[27][i] == -1: #P=6 C=10
        col3.append('#FFFF00')    
    else:
        col3.append('#00FFFF')

legend_elements = [
    Patch(facecolor='r', label='Value=1'),
    Patch(facecolor='b', label='Value=-1'),
    Patch(facecolor='#00FFFF', label='Predicted=1'),
    Patch(facecolor='#FFFF00', label='Predicted=-1'),]
#make into a for loop
fig = plt.figure()
plt.scatter(data1_x1, data1_x2, c=colours, marker='o',  s=40)
plt.scatter(data1_x1, data1_x2, c=col, marker='+', alpha=0.8, s=24)
plt.legend(handles=legend_elements, loc='upper left')
plt.title("P=1 C=1")

fig = plt.figure()
plt.scatter(data1_x1, data1_x2, c=colours, marker='o',  s=40)
plt.scatter(data1_x1, data1_x2, c=col1, marker='+', alpha=0.8, s=24)
plt.legend(handles=legend_elements, loc='upper left')
plt.title("P=2 C=10")

fig = plt.figure()
plt.scatter(data1_x1, data1_x2, c=colours, marker='o',  s=40)
plt.scatter(data1_x1, data1_x2, c=col2, marker='+', alpha=0.8, s=24)
plt.legend(handles=legend_elements, loc='upper left')
plt.title("P=3 C=10")

fig = plt.figure()
plt.scatter(data1_x1, data1_x2, c=colours, marker='o',  s=40)
plt.scatter(data1_x1, data1_x2, c=col3, marker='+', alpha=0.8, s=24)
plt.legend(handles=legend_elements, loc='upper left')
plt.title("P=6 C=10")

plt.show()

#Part B
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
knn_predictions = []
mean_accuracy = []
accuracy_std_error = []
mean_F1 = []
F1_std_error = []

for k in k_values:
    new_model = KNeighborsClassifier(n_neighbors=k,weights='uniform').fit(data1_X, data1_y)
    knn_predictions.append(new_model.predict(data1_X))
    scores = cross_val_score(new_model, data1_X, data1_y, cv=5, scoring="accuracy")
    mean_accuracy.append(np.array(scores).mean())
    accuracy_std_error.append(np.array(scores).std())
    scores = cross_val_score(new_model, data1_X, data1_y, cv=5, scoring="precision")
    mean_F1.append(np.array(scores).mean())
    F1_std_error.append(np.array(scores).std())

fig = plt.figure()
plt.title("K Value vs Model Accuracy and Precision")
plt.errorbar(k_values, mean_accuracy, yerr=accuracy_std_error, c='r')
plt.errorbar(k_values, mean_F1, yerr=F1_std_error, c='b')
plt.xlabel("K Values")
plt.ylabel("Metric %")

col4 = []
for i in range(0, len(knn_predictions[3])):
    if knn_predictions[3][i] == -1: #k=4
        col4.append('#FFFF00')    
    else:
        col4.append('#00FFFF')

fig = plt.figure()
plt.scatter(data1_x1, data1_x2, c=colours, marker='o',  s=40)
plt.scatter(data1_x1, data1_x2, c=col4, marker='+', alpha=0.8, s=24)
plt.legend(handles=legend_elements, loc='upper left')
plt.title("K=4")
plt.show()

#Part C
#do the confussion matrix for the basline models, the logistic regression and the KNN model
baseline_random_predictions = []
baseline_most_popular_predictions = []
options = [-1, 1]
for x in range(0, len(data1_y)):
    baseline_random_predictions.append(random.choice(options))
    baseline_most_popular_predictions.append(1)

baseline_random_metrics = confusion_matrix(data1_y, baseline_random_predictions)
baseline_most_popular_metrics = confusion_matrix(data1_y, baseline_most_popular_predictions)
knn_metrics = confusion_matrix(data1_y, knn_predictions[6]) #k=7
logistic_regression_metrics = confusion_matrix(data1_y, all_predictions[7]) # P=2, C=10
print ("baseline popular metrics", baseline_most_popular_metrics)
print ("baseline random metrics", baseline_random_metrics)
print ("knn_metrics", knn_metrics)
print ("logistic_regression_metrics", logistic_regression_metrics)

#Part D
chosen_logistic_regression_model = LogisticRegression(penalty="l2", C=10).fit(data1_X, data1_y)
chosen_knn_model = KNeighborsClassifier(n_neighbors=7,weights='uniform').fit(data1_X, data1_y)
knn_y_scores = chosen_knn_model.predict_proba(data1_X)

fpr_lr, tpr_lr, _ = roc_curve(data1_y,chosen_logistic_regression_model.decision_function(data1_X))
fpr_knn, tpr_knn, _ = roc_curve(data1_y,knn_y_scores[:,1])

fig = plt.figure()
plt.plot(fpr_lr,tpr_lr, color='green', linestyle='-', label="lr")
plt.plot(fpr_knn,tpr_knn, color='red', linestyle='-', label="knn")
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label="random")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.show()


#legend code for error plots 
# legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
#make the error bars slightly transulcent so we can see the overlap
#clean up graphs
#clean up code and add meaningful variable names