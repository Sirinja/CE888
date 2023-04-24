#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score


# In[2]:


root_dir='C:/Users/Admin/Documents/1.MSc_Data science/Data science&decision making/Assignment/EyeT/EyeT'

directory = root_dir
all_files = os.listdir(root_dir)

test_groups=[file for file in all_files if "letter" in file and file.endswith(".csv")]
control_groups=[file for file in all_files if file not in test_groups and file.endswith(".csv")]


# In[3]:


test=[]
control=[]
for i in test_groups:
    file_path = os.path.join(directory, i)
    t = pd.read_csv(file_path,low_memory=False)
    test.append(t)

for j in control_groups:
    file_path = os.path.join(directory, j)
    c = pd.read_csv(file_path,low_memory=False)
    control.append(c)

    
testgroup_concat = pd.concat(test, axis=0, ignore_index=True)
controlgroup_concat = pd.concat(control, axis=0, ignore_index=True)


# In[4]:


testgroup_concat['Project name']='Test group experiment'
controlgroup_concat['Project name']='Control group experiment'

print('The member of project name in test group experiment are',testgroup_concat['Project name'].unique())
print('The member of project name in control group experiment are',controlgroup_concat['Project name'].unique())


# In[5]:


df=pd.concat([testgroup_concat,controlgroup_concat])


# ## Features Extraction

# ### The change in pupil diameter

# In[6]:


pupil=df[['Participant name','Recording name','Pupil diameter right','Pupil diameter left']].dropna().replace(',', '.', regex=True).replace("0.00", np.NaN)


# In[7]:


pupil.iloc[:,2:]=pupil.iloc[:,2:].astype(float)
pupil.set_index(['Participant name', 'Recording name'], inplace=True)


# In[8]:


#find the different of pupil diameter
diff_pupil = pupil.diff()

diff_pupil.iloc[0]=0

diff_pupil.reset_index()

#Absolute the each different value for find only the change value
diff_pupil[['Pupil diameter right','Pupil diameter left']]=abs(diff_pupil[['Pupil diameter right','Pupil diameter left']])


# In[9]:


#find mean of the differece of pupil diameter

diff_pupil_mean=diff_pupil[['Pupil diameter right','Pupil diameter left']].groupby(['Participant name','Recording name']).mean()

diff_pupil_mean.rename(columns={'Pupil diameter right':'mean diff rt.pupil','Pupil diameter left':'mean diff lt.pupil'},inplace=True)


# In[10]:


#find max of the differece of pupil diameter

diff_pupil_max=diff_pupil[['Pupil diameter right','Pupil diameter left']].groupby(['Participant name','Recording name']).max()
diff_pupil_max.rename(columns={'Pupil diameter right':'max diff rt.pupil','Pupil diameter left':'max diff lt.pupil'},inplace=True)


# In[11]:


#find standard deviation of the differece of pupil diameter
diff_pupil_std=diff_pupil[['Pupil diameter right','Pupil diameter left']].groupby(['Participant name','Recording name']).std()
diff_pupil_std.rename(columns={'Pupil diameter right':'std diff rt.pupil','Pupil diameter left':'std diff lt.pupil'},inplace=True)


# In[12]:


merged_df_pupil = pd.concat([diff_pupil_mean, diff_pupil_max, diff_pupil_std], axis=1)


# ### Fixation duration

# In[13]:


eye_movement=df[['Participant name','Recording name','Eye movement type','Gaze event duration','Event']].set_index(['Participant name','Recording name'])


# In[14]:


#find the fixation duration in each record and participants
fixation_time = []
fixation_duration = []
fixation_index= []
fixation_start = None

for i, eye in eye_movement.iterrows():
    if eye['Eye movement type'] == 'Fixation':
        if fixation_start is None:
            fixation_start = eye['Gaze event duration']
        elif eye['Eye movement type'] != 'Fixation':
            fixation_index.append(i)
            fixation_duration.append(eye['Gaze event duration'])
            fixation_start = eye['Gaze event duration']
    elif fixation_start is not None:
        fixation_time.append(fixation_start)
        fixation_index.append(i)
        fixation_duration.append(eye['Gaze event duration'])
        fixation_start = None
        
if fixation_start is not None:
    fixation_time.append(fixation_start)
    fixation_index.append(i)
    fixation_duration.append(eye['Gaze event duration'])   


fixation=pd.concat([pd.DataFrame(fixation_index).rename(columns={0:'Participant name',1:'Recording name'})
                    ,pd.DataFrame(fixation_duration).rename(columns={0:'Duration'})],axis=1).set_index(['Participant name','Recording name'])


# In[15]:


##divide the fixation duration in each period

index=[]
period = []
ptype = None

for i,eye in fixation.iterrows():
    if eye['Duration'] <= 150:
        ptype = 'short'
        period.append(ptype)
        index.append(i)
    elif eye['Duration'] >= 900:
        ptype = 'long'
        period.append(ptype)
        index.append(i)
    else:
        ptype = 'medium'
        period.append(ptype)
        index.append(i)

period=pd.DataFrame(period).rename(columns={0:'Period'})

fixation_period = pd.concat([fixation.reset_index(), period],axis=1)


# In[16]:


#Encode period from categorical value to numerical value

from sklearn.preprocessing import OneHotEncoder

cat = {'short': 0, 'medium': 1, 'long': 2}

# Convert the categorical variable to numeric codes
fixation_period['Period_codes'] = fixation_period['Period'].map(cat)

# Create an instance of the OneHotEncoder class
enc = OneHotEncoder()

# Fit and transform the categorical variable in the DataFrame
encoded_data = enc.fit_transform(fixation_period[['Period_codes']]).toarray()

encode_duration=pd.DataFrame(encoded_data).rename(columns={0:'short',1:'medium',2:'long'})


# In[17]:


#Merge all fixation features 

merged_fixation = pd.concat([fixation_period,encode_duration], axis=1)
merged_fixation.drop(columns=['Period_codes', 'Period'], inplace=True)


# In[18]:


#Group fixation features by Participant name and Recording name
mean_fixation=merged_fixation.groupby(['Participant name','Recording name']).mean().rename(columns={'Duration':'mean duration',
                                                                                                    'short':'mean short',
                                                                                                   'medium':'mean medium',
                                                                                                   'long':'mean long'})


std_fixation=merged_fixation.groupby(['Participant name','Recording name']).std().rename(columns={'Duration':'std duration',
                                                                                                    'short':'std short',
                                                                                                   'medium':'std medium',
                                                                                                   'long':'std long'})

merged_df_fixation = pd.concat([mean_fixation,std_fixation], axis=1)


# ### Saccade

# In[19]:


#find the saccade count
saccade_counts = {}

for record, data in eye_movement.groupby(['Participant name','Recording name']):
    saccade_count = 0
    prev_movement_type = None
    for idx, row in data.iterrows():
        if prev_movement_type == 'Fixation' and row['Eye movement type'] == 'Saccade':
            saccade_count += 1
        prev_movement_type = row['Eye movement type']
    saccade_counts[record] = saccade_count

df_saccade = pd.DataFrame.from_dict(saccade_counts, orient='index', columns=['Saccade Count']).reset_index()
df_saccade[['Participant name', 'Recording name']] = pd.DataFrame(df_saccade['index'].tolist(), index=df_saccade.index)
df_saccade = df_saccade.drop('index', axis=1).set_index(['Participant name','Recording name'])


# ### Merge all features

# In[20]:


X_data=pd.concat([merged_df_pupil,merged_df_fixation,df_saccade],axis=1).reset_index()


# In[21]:


X_data


# ## Target

# ### Empathy score

# In[22]:


#Download target dataset
score_dir='C:/Users/Admin/Documents/1.MSc_Data science/Data science&decision making/Assignment/Questionaire/'

score_pre=pd.read_csv(score_dir+'Questionnaire_datasetIA.csv',encoding="cp1252")
score_post=pd.read_csv(score_dir+'Questionnaire_datasetIB.csv',encoding="cp1252")


# In[23]:


#Choose total score to be target
score=pd.DataFrame({'Participant name':X_data['Participant name'].unique(),
            'Score_original':score_post['Total Score original'],
            'Score_extended':score_post['Total Score extended']})


# In[24]:


data= pd.merge(X_data, score, on='Participant name')


# ### Merge features and target

# In[25]:


data= pd.merge(X_data, score, on='Participant name')

X = data.drop(['Participant name','Recording name', 'Score_original', 'Score_extended'], axis=1).fillna(0) 
#fill na in X with 0 because Nan value are std diff. both pupil which no variance.
y = data['Score_extended']


# In[26]:


#Scale input
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# ### Split train and test set

# In[27]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=10, stratify=y)


# ### Feature importance

# #### find the best model for find feature importance

# In[28]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score as acc

dummy_clf = DummyClassifier(strategy='stratified')
dummy_clf.fit(X, y)
print("Dummy ACC: %.2f" % acc(y, dummy_clf.predict(X)))


# In[29]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


clf = ExtraTreesClassifier(n_estimators=1000, max_depth=4)
scores = cross_val_score(clf, X, y, cv=10, scoring=make_scorer(acc))  # cross-validation
print("ACC: %.2f +/- %.2f" % (scores.mean(), scores.std()))


# In[30]:


from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
model_dt=dt.fit(X,y)
dt_score = cross_val_score(model_dt, X, y, cv=10, scoring=make_scorer(acc))
print("Decition Tree ACC:%.2f +/- %.2f"%(dt_score.mean(),dt_score.std()))


# In[31]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
model_rf=rf.fit(X, y)
rf_score = cross_val_score(model_rf, X, y, cv=10, scoring=make_scorer(acc))
print("Random forest ACC:%.2f +/- %.2f"%(rf_score.mean(),rf_score.std()))


# I choose random forest for find feature importance because ACC is the highest.

# In[32]:


# Fit the model
rf.fit(X_train, y_train)

# Let's see the feature importances for our classifier
importances = rf.feature_importances_
std = np.std([rf.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. %s (%f)" % (f + 1, X_train.columns[indices[f]],  importances[indices[f]]))

# Plot the feature importances of the forest
fig = plt.figure(figsize=(15,5))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices], color="green", yerr=std[indices])
plt.xticks(range(X_train.shape[1]), np.array(X_train.columns)[indices])
plt.xlim([-1, X.shape[1]])
plt.ylim([0, None])
plt.xticks(rotation=90)
plt.savefig('Feature ranking')


# In[33]:


num_features = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
RMSE = []

for i in num_features:
    # Select the top i most important features
    top_idx = indices[:i]
    X_train_i = X_train.iloc[:, top_idx]
    X_test_i = X_test.iloc[:, top_idx]

    # Train the model
    rf.fit(X_train_i, y_train)

    # Evaluate the performance
    y_pred = rf.predict(X_test_i.fillna(0))

    # Calculate RMSE
    rmse = np.sqrt(mse(y_test, y_pred))
    RMSE.append(rmse)

# Plot the results
plt.plot(num_features, RMSE, marker='o')
plt.xlabel('Number of features')
plt.ylabel('RMSE')
plt.title('RMSE vs Number of features')
plt.show()

# Find the optimal number of features
optimal_idx = np.argmin(RMSE)
optimal_num = num_features[optimal_idx]
plt.savefig('Optimal number of features')
print('Optimal number of features:', optimal_num)


# In[34]:


#Choose features follow the optimal number
features=[]
importance=[]
for f in range(X_train.shape[1]):
    features.append(X_train.columns[indices[f]])
    importance.append(importances[indices[f]])

importance_dict = dict(zip(features, importance))
importance = pd.DataFrame(list(importance_dict.items()),columns=['keys', 'values'])
importance.set_index('keys', inplace=True)

top_idx=importance.sort_values(by='values',ascending=False)[:optimal_num].index

X_final =X[top_idx]
print('The optimal features are',X_final.columns.values)


# ### Train model

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3,random_state=10, stratify=y)


# In[36]:


# Define the models
models = [
    ('LinearRegression', LinearRegression()),
    ('DecisionTreeRegressor', DecisionTreeRegressor()),
    ('RandomForest',RandomForestRegressor()), 
    ('GradientBoost',GradientBoostingRegressor()), 
    ('Bayes',BayesianRidge())]

# Evaluate each model using cross-validation
results = []
names = []
for name, model in models:
    cv = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_cv = np.sqrt(-cv)
    results.append(rmse_cv)
    names.append(name)
    print(f"{name}: RMSE = {round(rmse_cv.mean(),2)}± {round(np.std(rmse_cv), 3)}")

    
# plot the mean accuracy for each model
plt.figure(figsize=(10, 5))
plt.boxplot([result for result in results],labels=names)

# add labels and title
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Comparison of model performance')
plt.savefig('Comparison of model performance')
plt.show()



best_idx = np.argmin(np.mean(results, axis=1))
print(f"The best model is {names[best_idx]} with Mean RMSE = {round(np.mean(results[best_idx]), 3)} ± {round(np.std(results[best_idx]), 3)}")


# In[37]:


model = BayesianRidge()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
print("RMSE on training set:", round(np.sqrt(mse(y_train, y_train_pred)),3))


# In[38]:


y_test_pred = model.predict(X_test)
print("RMSE on training set:", round(np.sqrt(mse(y_test, y_test_pred)),3))


# ### Hyperparameter fine-tuning

# In[39]:


from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid to search over
param_grid = {
    'n_iter':[100, 200, 300],
    'alpha_1': [1e-7, 1e-6, 1e-5],
    'alpha_2': [1e-7, 1e-6, 1e-5],
    'lambda_1': [1e-7, 1e-6, 1e-5],
    'lambda_2': [1e-7, 1e-6, 1e-5],
    'alpha_init':[1e-7, 1e-6, 1e-5,None],
    'lambda_init':[1e-7, 1e-6, 1e-5,None]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
random_search.fit(X, y)

# Print the best parameters and score
print("Best score: ", random_search.best_score_)
print("Best estimators: ",random_search.best_estimator_)


# In[40]:


final_model=random_search.best_estimator_


# In[41]:


final_model.fit(X_train, y_train)
final_y_train_pred = final_model.predict(X_train)
print("final RMSE on training set:", round(np.sqrt(mse(y_train, final_y_train_pred)),3))


# In[42]:


final_y_test_pred = final_model.predict(X_test)
print("final RMSE on training set:", round(np.sqrt(mse(y_test, final_y_test_pred)),3))


# In[ ]:





# In[ ]:




