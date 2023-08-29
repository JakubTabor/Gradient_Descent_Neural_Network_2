**BATCH GRADIENT DESCENT**

# First I load my simple dataset which include three columns : area, bedrooms and	price
* Now I need to scale my data, so I import "MinMaxScaler" from "preprocessing", and put inside area and bedroom columns I save it as X_scaled 
* Then I scale price column and I also reshape it into 2D array, I save it as x_scaled

# Now I gonna build from scratch batch gradient descent, I gonna need here X, y_true, epochs and learning_rate which I set to (0.01)
* First I pass "number_of_features", so "area and bedroom", shape of it will be in rows and columns
* Then I initialize my weights with ones and set shape as previous saved "number_of_features", also initialize bias with zero
* And I pass shape of "total_samples" for later which is " X.shape[0]"
* I create two empty list to record cost and epochs and later plot them 

# Then I make for loop in length of my epochs, inside I define "y_predicted", which is dot product of w1, w2 and X transpose, plus bias
* Now I define first derivative, so I pass X transpose dot product of difference between true and predicted value into minus average of total samples 
* Then my second derivative, which is sum of difference between true and predicted value into minus average of total samples
* Next I can calculate my weights, so my w minus learning rate into previous defined w_grad, and this same with bias 
* Then I define cost of my function, which is mean squared error between true and predicted value

# Now I define recording progress, and it will be every tenth iteration
* In my lists will be filling with values recorded values, cost and epochs
* And my function will return weights, bias, cost, cost list and epoch list, so I can plot it later

# Then I call my function and put inside X_scaled, y_scaled_reshaped, pass number of epochs (500)
* And get weights, bias and cost, I can compare it later whith "stochastic_gradient_descent"

# I plot history how my cost change during the epochs, I can see that close to 100th epoch my cost stop decreasing rapidly
* But I can see that "batch_gradient_descent" have smoth curve 

# Then I create prediction function, I gonna use in it "inverse_transform" from my "MinMaxScaler"
* So it return me inverse values, np. when I put 1 into it it will return me max. argument in my column, when i put 0.5, it will return me mean value of column
* I gonna use in my predict function : area, bedrooms, weights and bias 
* Now I must scale my features, so I use transform method from my scaler and put inside area and bedrooms and I supply [0] to get 1D array, all this I save as X_scaled
* Then I fill patter to get "scaled_price", so (w1 into X_scaled of [0] which is area  variable, plus w2 into X_scaled of [1] which is bedrooms variable, plus some bias
* Next I use "inverse_transform" method on "scaled_price" variable to get actual price, and supply [0][0] to get single value

# First prediction is for second house in my data, which is quite well predicted, next I check 5th house which is worst than previous
* My function not perform excellent, but is able to get close real values

**STOCHASTIX GRADIENT DESCENT**

# Now It's time to build stochastic gradient descent, I need to import random for it
* I gonna build the function that same as previous, so X, y_true, epochs and learning rate, which I set to (0.01)
* Now I pass the variables, so I define "number_of_features", weights, bias and "total_samples"   
* Then I create two lists to record epochs and cost, to plot them later

# I start for loop in length of my epochs, and now I gonna use previous imported "random" to choose random samples
* I need "randint" to pick random samples between (zero and total_samples - 1) indexes in python are -1, I save it as "random_index "
* Now I set it to pick samples from X and y_true, so I create two variables for it
* First will be (X of "random_index") and (y_true of "random_index"), I save it as "sample_x" and "sample_y" 

# Next I define "y_predicted", which is dot product between (w and transpose "sample_x") plus some bias 

# Then I calculate first derivatives, which is random sample of X transpose, dot product of difference between true sample and predicted
* And I take mean of it multiply by minus two, it will be variable "w_grad"
* Next my second derivatives for bias, it is the same, but in this case I take mean of y sample and predicted y, without dot product

# Now I can calculate weights, which is my "w" variable minus (learning rate) into (w_grad variable), which I calculate above
* And I do this same with bias, now I have defined bias and weights
* Then I calculate cost, which is squared difference between sample of y and predicted y

# Now I specify that cost and epoch will be recorded every 100th iteration, they will append in my lists
* I return from my function weights, bias cost, cost_list, epoch_list

# I can create object of my function "stochastic_gradient_descent", and put inside X_scaled, y_scaled and reshaped to "y_scaled.shape[0]", I pass (10000) epochs
* And I get weight, bias and cost of my "stochastic_gradient_descent", they are the same as form "batch_gradient_descent"
* But for "stochastic_gradient_descent" I need to set more epochs, because every epoch it pick only one sample

# Then I plot of graph previous saved in my lists (cost and epochs), we can see that it's not that smooth as on previous graph
* But later after 4000 epoch it is more stable

# I also check some predictions and they look pretty realistic, when I put (this same number of square feets but with one more bedroom) the price is higher

# Conclusions are that (Batch Gradient Descent) have good performance on small datasets, he use all training samples in one forward pass and then adjust weights
* In comparison (Stochastic Gradient Descent) use only one sample in forward pass and after adjust weights
* It is good when you have big dataset, because it saves a lot of computation, When you have a lot of computation power you can use also (Stochastic Gradient Descent)
