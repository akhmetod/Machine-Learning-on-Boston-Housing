# Machine-Learning-on-Boston-Housing

For the Boston housing data the objective is to apply a varied set of data mining techniques to best predict the continous variable- median value of occupied homes. The data contained 14 columns and 506 observations. On conducting EDA it was observed that there were no missing values in the data. The density plot of fields showed that black, chas, crim, zn show right skewness while age and ptratio show left skewness in data. Only age and tax showed a high correlation and there were a few outliers in the black and crim variables but since specific knowledge regarding outlier definition was not available, the outliers were ignored.

The various techniques applied for the prediction of median home value were- Generalized Linear Regression, Regression Tree, Generalized Additive Model and Neural Networks. The linear regression was fitted using a stepwise model selection procedure and the model performance was tested on MSE for both in-sample and out of sample data. The second model fitted was regression tree, where the tree size was picked through selecting the optimal tree complexity. Lastly for the nerual network model, which generally needs parameter tuning to arrive at the best results, two parameters were mainly tuned by looking for the least SSE- decay and hidden units. 
Below is a summary of the results of fitting various models for the prediction of median home value:

Model	In sample- MSE 
(80% training data)	Out of sample-MSE 
(20% testing data)	AIC	BIC
GLM-Linear	20.9	26.9	2400.9	2452.9
CART	16.7	21.6	-	-
GAM	7.9	13.5	2099.1	2336.5
Neural Net	4.5	9.8	-	-

Overall performances for the in-sample and out-of-sample were consistent across the models. Therefore, if neural networks had the best performance over in-sample, it typically gave the best results out-of-sample as well. This ensures there was no overfitting to a great degree that can lead to very poor out-of-sample performances. Ranking the performances, linear regression came out to be the worst in this exercise, with CART being slightly better. GAM was observed to nearly 100% better than linear regression. This could be down to smoothing of all the independent variables, as non-linearity was observed. The effective degrees of freedom for the GAM model came out to be 9.
Also as evidenced, Neural networks performed the best over both in sample and out of sample measures. The results were not always consistent on further re-runs, since convergence was not always at the global minima. Also, parameter tuning in this case was performed through cross validation, over a range of numbers for both decay (0 to 11) and units (0-20), with iterations kept at a maximum of 5000. When running with iterations at a 1000, convergence was not always observed. The observed best model had 8 hidden units and a decay of 0.008 that provided the least averaged SSE.
