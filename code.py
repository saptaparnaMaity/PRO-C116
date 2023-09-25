import pandas as pd
import plotly.express as px

df = pd.read_csv("Admission_Predict.csv")

TOEFL_Score = df["TOEFL Score"].tolist()
GRE_Score = df["GRE Score"].tolist()

fig = px.scatter(x=TOEFL_Score, y=GRE_Score)
fig.show()

import plotly.graph_objects as go

TOEFL_Score = df["TOEFL Score"].tolist()
GRE_Score = df["GRE Score"].tolist()

results = df["Chance of admit"].tolist()
colors=[]
for data in results:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")

fig = go.Figure(data=go.Scatter(
    x=TOEFL_Score,
    y=GRE_Score,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()

score = df[["TOEFL Score", "GRE Score"]]

results = df["Chance of admit"]
from sklearn.model_selection import train_test_split 

score_train, score_test, results_train, results_test = train_test_split(score, results, test_size = 0.25, random_state = 0)
print(score_train)

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(score_train, results_train)

results_pred = classifier.predict(score_test)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(results_test, results_pred))