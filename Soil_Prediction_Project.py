import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import ipywidgets as widgets
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import KFold

url = "https://raw.githubusercontent.com/Icegreen04/pbl/main/pbl%20dataset.csv"
dataset = pd.read_csv(url)
dataset

dataset.isnull().sum()

dataset.info()

dataset.duplicated().sum()

dataset.describe()

dataset['Crop'].unique()

bplot = sns.countplot(y='Crop',data=dataset, palette="muted")
bplot.set_ylabel('Crop', fontsize=10)
bplot.set_xlabel('Count', fontsize=10)
bplot.tick_params(labelsize=8)

sns.pairplot(data=dataset, hue = 'Crop')

scatter_matrix(dataset.drop('Crop', axis='columns'), figsize=(16, 16), marker='.', alpha=0.4, color='pink')

plt.show()

encoded_dataset = pd.get_dummies(dataset)


corr = encoded_dataset.corr()
print(corr)

sns.heatmap(corr,annot=True,cbar=True , cmap='coolwarm')

r = dataset.Crop.astype('category')
response = dict(enumerate(r.cat.categories))
dataset['response']=r.cat.codes

y=dataset.response
X=dataset[['Nitrogen','Phosphorus','Potassium','Temperature','pH','Rainfall']]

#Convert labels into categories codes and then declaring set x, y variables

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=10)

len(X_train)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_svm = svm.SVC(kernel='rbf', C=10)


model_svm.fit(X_train_scaled, y_train)


y_pred = model_svm.predict(X_test_scaled)


accuracy = round(accuracy_score(y_test, y_pred),3)
print("Accuracy: ",accuracy)

precision = round(precision_score(y_test, y_pred, average='weighted',zero_division=1),3)
recall = round(recall_score(y_test, y_pred, average='weighted'),3)
f1 = round(f1_score(y_test, y_pred, average='weighted'),3)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:",f1)

kfold = KFold(n_splits=5, shuffle=True, random_state=10)

scores = []
for train_index, test_index in kfold.split(X):
    X_train3, y_train3 = X.iloc[train_index], y.iloc[train_index]
    X_test3, y_test3 = X.iloc[test_index], y.iloc[test_index]
    model_svm.fit(X_train3, y_train3)

    y_pred = model_svm.predict(X_test3)
    score = accuracy_score(y_test3, y_pred)

    scores.append(score)

mean_accuracy = round(sum(scores) / len(scores),3)

print("Mean Accuracy:",mean_accuracy)

model_dt = DecisionTreeClassifier(max_depth = 10, criterion = 'gini', random_state=10)

model_dt = DecisionTreeClassifier(max_depth = 10, criterion = 'gini', random_state=10)

model_dt.fit(X_train_scaled,y_train)

y_pred = model_dt.predict(X_test_scaled)

accuracy = round(accuracy_score(y_test, y_pred),3)
print("Accuracy: ",accuracy)

kfold = KFold(n_splits=5, shuffle=True, random_state=10)

scores = []
for train_index, test_index in kfold.split(X):
    X_train2, y_train2 = X.iloc[train_index], y.iloc[train_index]
    X_test2, y_test2 = X.iloc[test_index], y.iloc[test_index]

    model_dt.fit(X_train2, y_train2)

    y_pred = model_dt.predict(X_test2)
    score = accuracy_score(y_test2, y_pred)

    scores.append(score)

mean_accuracy = round(sum(scores) / len(scores),3)

print("Mean accuracy:",mean_accuracy)

model_rf = RandomForestClassifier(max_depth=6, n_estimators=120, random_state=10)

model_rf.fit(X_train_scaled, y_train)

y_pred = model_rf.predict(X_test_scaled)

accuracy = round(accuracy_score(y_test, y_pred),3)
print("Accuracy: ",accuracy)

precision = round(precision_score(y_test, y_pred, average='weighted',zero_division=1),3)
recall = round(recall_score(y_test, y_pred, average='weighted'),3)
f1 = round(f1_score(y_test, y_pred, average='weighted'),3)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:",f1)

# K-fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=10)

#k-fold iteration
scores = []
for train_index, test_index in kfold.split(X):
    X_train_k, y_train_k = X.iloc[train_index], y.iloc[train_index]
    X_test_k, y_test_k = X.iloc[test_index], y.iloc[test_index]
    model_rf.fit(X_train_k, y_train_k)

    y_pred = model_rf.predict(X_test_k)
    score = accuracy_score(y_test_k, y_pred)

    #Append the accuracy score for this fold to the list of scores
    scores.append(score)

#Mean accuracy across all folds
mean_accuracy = round(sum(scores) / len(scores),3)

print("Mean Accuracy:",mean_accuracy)

plt.figure(figsize=(5, 3), dpi=80)
importance_sorted = sorted(zip(model_rf.feature_importances_, X_train.columns), reverse=True)
feature_importance = [imp for imp, _ in importance_sorted]
feature_names = [name for _, name in importance_sorted]

c_features = len(feature_importance)
plt.bar(range(c_features), feature_importance, color = "green")
plt.ylabel("Feature Importance")
plt.xlabel("Feature Name")
plt.xticks(np.arange(c_features), feature_names, rotation=45)


district_widget = widgets.Combobox(
    options=tuple(dataset['District_Name'].unique()),
    description='District:',
    placeholder='Select district',
    ensure_option=True
)

soil_color_widget = widgets.Combobox(
    description='Soil Color:',
    placeholder='Select soil color',
    ensure_option=True
)

nitrogen_widget = widgets.Combobox(
    description='Nitrogen:',
    placeholder='Select nitrogen value',
    ensure_option=True
)

phosphorus_widget = widgets.Combobox(
    description='Phosphorus:',
    placeholder='Select phosphorus value',
    ensure_option=True
)

potassium_widget = widgets.Combobox(
    description='Potassium:',
    placeholder='Select potassium value',
    ensure_option=True
)

ph_widget = widgets.Combobox(
    description='pH:',
    placeholder='Select pH value',
    ensure_option=True
)

rainfall_widget = widgets.Combobox(
    description='Rainfall:',
    placeholder='Select rainfall value',
    ensure_option=True
)

temperature_widget = widgets.Combobox(
    description='Temperature:',
    placeholder='Select temperature value',
    ensure_option=True
)

recommend_widget = widgets.Output()



def update_soil_color_options(change):
    district = change.new
    if district:
        soil_colors = dataset[dataset['District_Name'] == district]['Soil_color'].unique()
        soil_color_widget.options = tuple(soil_colors)
    else:
        soil_color_widget.options = ()

def update_nitrogen_options(change):
    district = district_widget.value
    soil_color = soil_color_widget.value
    if district and soil_color:
        nitrogen_values = dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Nitrogen'].unique()
        nitrogen_values = [str(value) for value in nitrogen_values]
        nitrogen_widget.options = tuple(nitrogen_values)
    else:
        nitrogen_widget.options = ()

def update_phosphorus_options(change):
    district = district_widget.value
    soil_color = soil_color_widget.value
    if district and soil_color:
        phosphorus_values = dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Phosphorus'].unique()
        phosphorus_values = [str(value) for value in phosphorus_values]
        phosphorus_widget.options = tuple(phosphorus_values)
    else:
        phosphorus_widget.options = ()

def update_potassium_options(change):
    district = district_widget.value
    soil_color = soil_color_widget.value
    if district and soil_color:
        potassium_values = dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Potassium'].unique()
        potassium_values = [str(value) for value in potassium_values]
        potassium_widget.options = tuple(potassium_values)
    else:
        potassium_widget.options = ()

def update_ph_options(change):
    district = district_widget.value
    soil_color = soil_color_widget.value
    if district and soil_color:
        ph_values = dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['pH'].unique()
        ph_values = [str(value) for value in ph_values]
        ph_widget.options = tuple(ph_values)
    else:
        ph_widget.options = ()

def update_rainfall_options(change):
    district = district_widget.value
    soil_color = soil_color_widget.value
    if district and soil_color:
        rainfall_values = dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Rainfall'].unique()
        rainfall_values = [str(value) for value in rainfall_values]
        rainfall_widget.options = tuple(rainfall_values)
    else:
        rainfall_widget.options = ()

def update_temperature_options(change):
    district = district_widget.value
    soil_color = soil_color_widget.value
    if district and soil_color:
        temperature_values = dataset[(dataset['District_Name'] == district) & (dataset['Soil_color'] == soil_color)]['Temperature'].unique()
        temperature_values = [str(value) for value in temperature_values]
        temperature_widget.options = tuple(temperature_values)
    else:
        temperature_widget.options = ()



def train_model(change):

    district = district_widget.value
    soil_color = soil_color_widget.value
    nitrogen = float(nitrogen_widget.value)
    phosphorus = float(phosphorus_widget.value)
    potassium = float(potassium_widget.value)
    pH = float(ph_widget.value)
    rainfall = float(rainfall_widget.value)
    temperature = float(temperature_widget.value)


    input_data = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, pH, rainfall, temperature, district, soil_color]],
        columns=['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature', 'District_Name', 'Soil_color']
    )


    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = encoder.fit_transform(dataset[['District_Name', 'Soil_color']])
    input_data_encoded = encoder.transform(input_data[['District_Name', 'Soil_color']])


    X_train, X_test, y_train, y_test = train_test_split(X_encoded, dataset['Crop'], test_size=0.2, random_state=42)


    model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
    model_crop.fit(X_train, y_train)


    predicted_crop = model_crop.predict(input_data_encoded)


    recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop[0]]['Fertilizer'].values[0]

    with recommend_widget:
        recommend_widget.clear_output()
        print("Recommended Crop:", predicted_crop[0])
        print("Recommended Fertilizer:", recommended_fertilizer)


district_widget.observe(update_soil_color_options, names='value')
district_widget.observe(update_nitrogen_options, names='value')
soil_color_widget.observe(update_nitrogen_options, names='value')


district_widget.observe(update_phosphorus_options, names='value')
soil_color_widget.observe(update_phosphorus_options, names='value')

district_widget.observe(update_potassium_options, names='value')
soil_color_widget.observe(update_potassium_options, names='value')

district_widget.observe(update_ph_options, names='value')
soil_color_widget.observe(update_ph_options, names='value')

district_widget.observe(update_rainfall_options, names='value')
soil_color_widget.observe(update_rainfall_options, names='value')

district_widget.observe(update_temperature_options, names='value')
soil_color_widget.observe(update_temperature_options, names='value')




button = widgets.Button(description='Train Model')

button.on_click(train_model)


widgets.VBox([district_widget, soil_color_widget, nitrogen_widget,phosphorus_widget,potassium_widget,ph_widget,rainfall_widget,temperature_widget,button,recommend_widget])