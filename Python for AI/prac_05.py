from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x= df[['age', 'education_years']]
y = df['income']

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(Y_test, predictions)
print(mse)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(X_train, Y_train)

accuracy = accuracy_score(Y_test,classifier.predict(X_test))
print(f"একিউরেসি: {accuracy:.2f}")