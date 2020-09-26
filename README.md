# ML-Algorithms-Practice-Examples

REGRESSION 

Regression is one of the most important concept in Machine Learning.
There are various types of Regression.
What exactly Regression stands for?
•	Regression is a statistical method.
•	It is used to determine the strength and character of the relationship between one dependent variable (usually denoted by Y) and a series of other independent variables.
Mathematical representation  : Y = βo + β1X + ∈
where, Y - Dependent variable
X - Independent variable
βo - Intercept
β1 - Slope
∈ - Error
1.	Here βo and β1 are 2 coefficients. 
2.	The variable we predict : Y
3.	The variable we use to make a prediction : X
4.	βo - This is the intercept term. It is the prediction value you get when X = 0
5.	β1 - This is the slope term. It explains the change in Y when X changes by 1 unit. ∈ - This represents the residual value, i.e. the difference between actual and predicted values.
The above equation is of simple linear regression. It is called linear as there is only one independent variable, if there are many independent variables present it is called as Multiple regression.
So. Formula to calculate these coefficients?
β1 = Σ(xi - xmean)(yi-ymean)/ Σ (xi - xmean)² where i= 1 to n (no. of obs.)

                       βo = ymean - β1(xmean)
                       
error = y(real)-y(predicted) = y(real)-(a+bx)



Here, I tried to build a linear regression model for a height weight dataset referring articles available on towardsdatascience.
Females correlation coefficient: 0.849608
Males correlation coefficient: 0.8629788

The plot shows a positive linear relation between height and weight for males and females.

- Residual plots show the difference between actual and predicted values. If the points in a residual plot are randomly dispersed around the horizontal axis, a linear regression model is appropriate for the data; otherwise, a non-linear model is more appropriate.
In residual.py file, what I observed is : the points are randomly distributed around 0, so linear regression is an appropriate model to predict the data.
