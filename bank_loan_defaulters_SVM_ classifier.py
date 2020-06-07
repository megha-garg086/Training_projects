import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelBinarizer
from sklearn.svm import SVC

def load_data():
    df = pd.read_csv('data_loan.csv')
    # Importing the dataset
    df = df.fillna(0.)
    X = df.iloc[:, 1: ]
    print(X)
    y = df["loan_status"]
    print(y)
    return X, y

def split_transform(X, y):
    #Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    print(len(X_train), len(X_test), len(y_train), y_test)
    return X_train, X_test, y_train, y_test

def data_plot(y_train):
    pos = y_train[y_train.values == 0].shape[0]
    neg = y_train[y_train.values == 1].shape[0]
    print(f"Positive examples = {pos}")
    print(f"Negative examples = {neg}")
    print(f"Proportion of positive to negative examples = {(neg/pos) * 100:.2f}%")
    plt.figure(figsize=(8, 6))
    sns.countplot(y_train)
    plt.xticks((0, 1), ["Paid fully", "Default"])
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title("Class counts", y=1, fontdict={"fontsize": 20})
    plt.show()

def data_resample(X_train, y_train):
    # Resampling using Smote
    sm = SMOTE(sampling_strategy=1, random_state=87)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    data_plot(y_train)
    return X_train, y_train

def data_modeling(X_train, y_train, X_test):
    classifier = SVC(kernel = 'rbf', random_state=87)

    boosting = AdaBoostClassifier(base_estimator=classifier, n_estimators=10, algorithm='SAMME')
    smote_rfc = RandomForestClassifier(n_estimators= 100,
                                    criterion="entropy", random_state=47)

    '''ensemble = VotingClassifier(estimators=[('lr', boosting),('RandomForest', smote_rfc)],
                           voting='soft', weights=[1,1]).fit(X_train,y_train)'''

    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker='.', label='SVM')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    return roc_auc_score(y_test, y_pred, average=average)

def data_evaluation(y_test, y_pred):
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    res_accuracy = accuracy_score(y_test, y_pred)
    print("Accuray Scroe :" + str(res_accuracy))
    # f1 score
    res_f1score = f1_score(y_test, y_pred)
    print("F1 Score :" + str(res_f1score))

    res_recall = recall_score(y_test, y_pred)
    print("recall Score :" + str(res_recall))

    auc = multiclass_roc_auc_score(y_test, y_pred, average="macro")
    print("Area under curve : ", auc)



if __name__ == '__main__':
    #load data from CSV file
    X, y = load_data()
    #Split and transofrm Data
    X_train, X_test, y_train, y_test = split_transform(X, y)
    #plot data count
    data_plot(y_train)
    # resampling imbalanced data
    X_train, y_train = data_resample(X_train, y_train)
    #make predictions on test data
    predictions = data_modeling(X_train, y_train, X_test)
    #Evaluate test data
    data_evaluation(y_test, predictions)
