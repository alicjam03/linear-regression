Linear Regression on C02 Dataset
This project explores building a linear regression model in three different ways:
1. Using only Python built-ins
2. Using NumPy and Pandas
3. Standard machine learning libraries (scikit-learn)

Dataset
The data used was CO2 Emissions_Canada.csv which contains features of vehicles the project focuses on these two features:
1. Engine Size(L) as input x
2. CO2 Emissions (g/km) as output y

Objectives:
1. To predict CO2 emissions from engine size using linear regression
2. Understand the math and logic behind the model

Implementation:
Linear Regression with Libraries
File: libraries_linear_regression.py
Output: plot_with_libraries.png 

- Uses pandas, numpy, scikit-learn
- StandardScaler handles feature scaling
- LinearRegression model trains and predict with no need to specify epochs
- Easiest and fastest to implement

Linear Regression - NumPy and Pandas
File: numpy_linear_regression.py
Output: plot_only_numpy.png

- Uses NumPy and Pandas for data handling
- Manually implements gradient descent
- Uses 3000 epochs with a training loop
- Need manual experimentation or manual early stopping to find optimal epochs
- Great for understanding and learning

Linear Regression with No Libraries
File: no_numpy_linear_regression.py
Output: plot_no_numpy.png

- Pure Python
- CSV file read with csv module
- Manual mean/standard deviation calculation
- Loop based prediction and gradient decent with 3000 epochs
- Reinforced deep understanding

Standardization Explained
Before training the model z-score normalisation was implemented. This ensures all values x and y are on the same scale. The average becomes 0 and the spread becomes 1. Firstly standard deviation needs to be calculated using this formula:
![alt text](image-3.png)

To calculate is:
1. Find the mean of all the values
2. Subtract the mean from each value
3. Square each result
4. Sum all the squared results
5. Divide by the number of values
6. Take the square root

This tells us how much the data varies. We use this value to perform z-score normalisation. The formula for this is:
![alt text](image-4.png)

To calculate this:
1. Subtract the original value from the mean
2. Divide by the standard deviation

Model Training and Loss
The models aim to minise the Mean Squared Error, the formula is:
![alt text](image-1.png)
- w is the weight (slope)
- b is the bias (intecrept)
- To update the gradients a learning rate of 0.001 is used



