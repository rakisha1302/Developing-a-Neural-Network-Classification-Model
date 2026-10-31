# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

## PROGRAM
```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from torch.utils.data import TensorDataset,dataloader
import pandas as pd
df = pd.read_csv(r"C:\Users\acer\Downloads\customers.csv")
df
df=df.drop(columns=["ID"])
df
df.fillna({"work_Experience": 0, "Family_Size": df["Family_Size"].median()}, inplace=True)
df.columns
cat_cols=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Segmentation"]
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
df
x = df.drop(columns=["Segmentation"])
y = df[["Segmentation"]].values
x = pd.get_dummies(x, dtype=int)
xt, xst, yt, yst = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler1 = StandardScaler()
xt = scaler1.fit_transform(xt)
xst = scaler1.transform(xst)
xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)
tr = TensorDataset(xt, yt.squeeze(1).long())
tst = TensorDataset(xst, yst.squeeze(1).long())

trl = DataLoader(tr, batch_size=64, shuffle=True)
tstl = DataLoader(tst, batch_size=64)

class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,128)
        self.l2=nn.Linear(128,64)
        self.l3=nn.Linear(64,32)
        self.l4=nn.Linear(32,4)
        self.dropout=nn.Dropout(0.3)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.dropout(x)
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
yt = yt.squeeze(1).long()
yst = yst.squeeze(1).long()
model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.0001)
model.eval()

with torch.no_grad():
    output = model(xt[:10])
    loss = criterion(output, yt[:10])

print("Output contains NaN:", torch.isnan(output).any())
print("Initial loss:", loss.item())
xt = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)
xst = torch.nan_to_num(xst, nan=0.0, posinf=0.0, neginf=0.0)

print(torch.isfinite(xt).all())
print(torch.isfinite(xst).all())
xt = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)
xst = torch.nan_to_num(xst, nan=0.0, posinf=0.0, neginf=0.0)

print(torch.isfinite(xt).all())
print(torch.isfinite(xst).all())
epochs=200
for i in range(epochs):
    model.train()
    total_loss = 0

    for a, b in trl:
        op.zero_grad()

        pred = model(a)


        loss = criterion(pred, b)

        loss.backward()
        op.step()

        total_loss += loss.item()

    if i % 10 == 0:
        print("Epoch:", i, "Loss:", total_loss / len(trl))
pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    
    act.extend(yst.numpy())
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(act, pre)

conf_matrix = confusion_matrix(act, pre)

cl_report = classification_report(act, pre)

print("Accuracy:", accuracy * 100, "%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", cl_report)
import seaborn as sns
import matplotlib.pyplot as plt
xl=['A','B','C','D']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=xl, yticklabels=xl,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

### Name: RAKISHA R

### Register Number: 212225230223

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here



    def forward(self, x):
        #Include your code here
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here

```

### Dataset Information

<img width="1302" height="442" alt="image" src="https://github.com/user-attachments/assets/2861b0fc-563e-43d1-9457-9ad996dbaa39" />


### OUTPUT

<img width="680" height="567" alt="image" src="https://github.com/user-attachments/assets/0639323d-889b-4571-a2ec-387c08ef4df0" />


<img width="577" height="410" alt="image" src="https://github.com/user-attachments/assets/1f0e044b-2dab-4353-994f-9215d4aac57e" />


## RESULT
Thus, the neural network classification model was successfully developed and trained using the customer segmentation dataset. The model classified customers into four segments (A, B, C, and D) using a PyTorch neural network. The performance was evaluated using accuracy, confusion matrix, and classification report.

The model successfully performed multiclass classification of customers into the four segments.
