import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz


#creiamo la prima riga che conterrà l'intestazione di ogni colonna
column_index = ['AGE', 'WORKCLASS', 'FNLWGT', 'EDUCATION', 'EDUCATION-NUM', 'MARITAL-STATUS', 'OCCUPATION', 
                'RELATIONSHIP', 'RACE', 'SEX', 'CAPITAL-GAIN', 'CAPITAL-LOSS', 'HOURS-PER-WEEK', 'NATIVE-COUNTRY', 'CLASSE']
#carichiamo i dati del training set dal file, usiamo 'r' perche' indica il percorso raw, altrimenti legge '\' come carattere di escape
#specifichiamo che il nostro file non contiene un'intestazione, la aggiungiamo noi con l'attributo 'names=column_index'
training_data =  pd.read_csv(r'C:\Users\simos\OneDrive\Desktop\uniPd\Tesi\adult\adult.data', header = None, names=column_index)

#print("DATASET COMPLETO prima della decodifica:\n",training_data)  #for testing

#PROBLEMA: sckit learn non riesce ad operare con dati categorici, bisogna trasformarli in dati numerici
#salvo le colonne con i dati categorici da trasformare
categorical_col = ['WORKCLASS', 'EDUCATION', 'MARITAL-STATUS', 'OCCUPATION', 'RELATIONSHIP', 'RACE', 'SEX', 'NATIVE-COUNTRY']
#seleziono i dati categorici, quelli appartenenti alle colonne selezionate nella riga precedente
categorical_training_data = training_data[categorical_col]
#print("DATI CATEGORICI:\n",categorical_training_data)  #for testing
#creo oggetto di OneHotEncoder per fare la decodifica dei dati categorici
encoder = OneHotEncoder(handle_unknown='ignore')

#creo una struttura dati che contiene i dati categorici decodificati
encoded_categorical_data = encoder.fit_transform(categorical_training_data)
encoded_categorical_data_df = pd.DataFrame(encoded_categorical_data.toarray(), columns=encoder.get_feature_names_out(input_features=categorical_col))
training_data = pd.concat([training_data.drop(categorical_col, axis=1), encoded_categorical_data_df], axis=1)
#la colonna 'CLASSE' contiene solo 2 tipi di elementi, '>50K' e '<=50K'. Con la seguente riga rendo di tipo numerico gli elementi
#che assumono valore 1 se =='>50K', 0 altrimenti
training_data['CLASSE'] = (training_data['CLASSE'] == '>50K').astype(int)

#print("DATASET COMPLETO dopo la decodifica:\n",training_data)  #for testing

#creo il training data, eliminando la colonna che rappresenta il label set
X_train = training_data.drop('CLASSE', axis=1)
#creazione label set, usando l'intera colonna 'classe'
Y_train = training_data['CLASSE']

#salvo i nomi delle colonne del training set
training_columns = X_train.columns
#nome colonne del label set
label_column = ['CLASSE']


#scarico il test set, elimino la prima riga che contiene informazioni non inerenti al set
test_data = pd.read_csv(r'C:\Users\simos\OneDrive\Desktop\uniPd\Tesi\adult\adult.test', header=None, names=column_index).drop(0)

#seleziono i dati categorici, quelli appartenenti alle colonne selezionate
categorical_test_data = test_data[categorical_col]
#print("DATI CATEGORICI:\n",categorical_test_data)

#creo una struttura dati che contiene i dati categorici decodificati
encoded_categorical_test_data = encoder.transform(categorical_test_data)
encoded_categorical_test_data_df = pd.DataFrame(encoded_categorical_test_data.toarray(), columns=encoder.get_feature_names_out(input_features=categorical_col))
test_data = pd.concat([test_data.drop(categorical_col, axis=1), encoded_categorical_test_data_df], axis=1)
test_data['CLASSE'] = (test_data['CLASSE'] == '>50K').astype(int)
#training set per il test set
X_test = test_data.drop('CLASSE', axis=1)
#label set per il test set
Y_test = test_data['CLASSE']


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)
class_report = classification_report(Y_test, Y_pred)

print(f'accuracy: {accuracy}')
print('Matrice di confusione:')
print(conf_matrix)
print('Report di classificazione:')
print(class_report)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=training_columns,
                                class_names=label_column,
                                filled=True, rounded=True,
                                special_characters=True)

graph =graphviz.Source(dot_data)
graph.format = 'pdf'
graph.render("decision_tree")


"""
#TEST su pandas

print("DATASET COMPLETO:\n",training_data)
print("TRAINING SET:\n",X_train)
print("LABEL SET TRAINING:\n",Y_train)

print("DATASET COMPLETO:\n",test_data)
print("TEST SET:\n",X_test)
print("LABEL SET TEST:\n",Y_test)
"""
