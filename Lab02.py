from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


# TASK 1
pd.set_option('display.precision', 2)
nyc = pd.read_csv('nyc-temp-1895-2022.csv')
nyc.columns = ["Date", "Temperature", "Anomally"]
nyc.Date = nyc.Date.floordiv(100)
nyc.index = nyc.Date
print(nyc.head())

train_set = nyc.loc[1895: 2019]
test_set = nyc.loc[2019:]

x_train = train_set.Date
y_train = train_set.Temperature

x_test = test_set.Date
y_test = test_set.Temperature


linear_regression = stats.linregress(x=x_train, y=y_train)
print(linear_regression.slope, linear_regression.intercept)
print('===================================================================')

# TASK 2
sbn.lmplot(x='Date', y='Temperature', data=train_set, order=1, line_kws={'color' : 'green'})
plt.show()
print('===================================================================')


# TASK 3
for value in x_test:
    test_temp = linear_regression.slope * value + linear_regression.intercept
    print(f"Predicted temperature for {value} is {test_temp}")
print('===================================================================')

# TASK 4
temp_for_1888 = linear_regression.slope * 1888 + linear_regression.intercept
print(f"Predicted temperature for 1888 {temp_for_1888}")
temp_for_1791 = linear_regression.slope * 1791 + linear_regression.intercept
print(f"Predicted temperature for 1791 {temp_for_1791}")
print('===================================================================')

# TASK 5
sbn.set_style('whitegrid')
axes = sbn.regplot(x=nyc.Date, y=nyc.Temperature)
print('===================================================================')

# TASK 6
axes.set_ylim(10, 70)
plt.show()
print('===================================================================')

# TASK 7
for value in x_test:
    test_temp = linear_regression.slope * value + linear_regression.intercept
    print(f"""Real temperature in {value} is {y_test[value]}, predicted value is {test_temp}. 
          \n Difference between them is {abs(y_test[value] - test_temp)} degrees""")
print("""As we can see, there is a difference between predicted and real values but that is not crucial. 
      So, our model did managed pretty good with the task""")
      