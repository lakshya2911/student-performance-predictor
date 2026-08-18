
import pandas as pd
from sklearn.linear_model import LinearRegression
df=pd.read_csv("student performance.csv")
x=df[["Hours","Attendance","Previous Marks","Assignments"]]
y=df["Final Marks"]
model=LinearRegression()
model.fit(x,y)
prediction=model.predict([[5,80,65,7]])
print("predicted final marks are==",prediction[0])
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv("student_performance_2000.csv")
x=df[["Hours","Attendance","Previous Marks","Assignments"]]
y=df["Final Marks"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model=LinearRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
r2=r2_score(y_test,prediction)
mea=mean_absolute_error(y_test,prediction)
print("the variation is=",r2)
print("the average mean error is=",mea)
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
previous_marks = float(input("Enter previous marks: "))
assignments = float(input("Enter assignments completed: "))
new_student = [[hours, attendance, previous_marks, assignments]]
new_prediction = model.predict(new_student)
print("predicted final marks are==",round(new_prediction[0],2))
num=new_prediction[0]
#print(df.describe())
if num>=40:
    print("the student is pass!!")
else:
    print("student has failed")    
import matplotlib.pyplot as plt
plt.scatter(y_test,prediction)
plt.title("marks")
plt.xlabel("actual final marks") 
plt.ylabel("predicted final marks")   
plt.show()
