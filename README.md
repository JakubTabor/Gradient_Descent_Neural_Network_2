# I am gonna write from scratch two Gradient Descent (Batch and Stochastic), I choose this to to show how different look their (cost while epochs graph)
* In case of Batch Gradient Descent curve look very smooth, because it take all training samples for one forward pass
* It has a good performance on small datasets

![](https://github.com/JakubTabor/Portfolio_Deep_Learning/blob/main/Images/Batch_GD_plot.png)

# The reason why Gradient Descent is so important in Machine Learning is that it helps to reduce cost function, by changing a model's parameters. It provide   us to achieve the best accuracy


* When we compare Stochastic Gradient Descent we can see that the curve looks very irregular, thats because it use only one randomly picked sample for a forward pass.
* It is very eficiente on a large datasets

![](https://github.com/JakubTabor/Portfolio_Deep_Learning/blob/main/Images/Stochastic_GD_plot.png)

# The Third type of Gradient Descent is Mini Batch Gradient Descent 
* It could be used also on a big datasets, in this case it is very similar to SGD, but its faster on learning process
* On graph appears not perfect smooth line because my Gradient Descent (use batch of randomly picked samples)
* We set batch size at some number e.g 5 and after forward pass it find error for 5 randomly picked samples, so it updates weights more often than SGD


![](https://github.com/JakubTabor/Portfolio_Deep_Learning/blob/main/Images/Mini_Batch_Gradient_Descent.png)

