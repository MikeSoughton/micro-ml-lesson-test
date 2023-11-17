---
title: "Logistic Regression"
teaching: 15
exercises: 45
objectives:
- "When to perform logistic regression"
- "Setup training and testing data for logistic regression"
- "Understand the binary cross-entropy loss"
- "How to tune your model and implement regularisation"
- "Understand the performance metrics that can be used to evaluate classification models"
keypoints:
- "Binary classification"
- "Multiclass classification"
- "Likelihood"
- "Cross-entropy loss"
- "Performance metrics"
---

## Key concepts and lesson objectives
We have learnt the following concepts in the last lesson on linear regression which we shall use again here:
- Train-test splitting
- Gradient descent
- The loss/cost function
- Overfitting
- The bias-variance tradeoff

Continuing on from this we shall learn how logistic regression can be used for classification tasks in a similar vein to linear regression, but with a few important changes. By the end of this lesson you should know:
- When to perform logistic regression
- Setup training and testing data for logistic regression
- Understand the binary cross-entropy loss
- How to tune your model and implement regularisation
- Understand the performance metrics that can be used to evaluate classification models

## Logistic regression introduction

We saw in the last lesson on linear regression how we can train a model $$g(\boldsymbol{x} \mid \boldsymbol{\theta})$$ on training data $$\boldsymbol{X}_\mathrm{train}$$ with a continuous label $$Y_\mathrm{train}$$. But what now if we have a task where we want to train a model to distinguish between data which may belong to one of two classes. For example we may want to identify whether an image contains an image or either a cat or a dog? Or maybe we would like diagnose whether a patient has a medical condition based on their test results or perhaps we want to build an email filter which will filter out spam emails but keep legitimate emails. 
<img src="{{ page.root }}/fig/cats_and_dogs.png" alt="drawing" width="500"/>
<img src="{{ page.root }}/fig/Patient-data.png" alt="drawing" width="500"/>

> ## Discuss
>
> All of these tasks should have a binary output - what would the final output for each case be in human readable terms? \
> What would the final output for each case be in machine readable format? \
> Bonus: how many input dimensions are there for the patient example? How many would there be with image inputs?
{: .callout}

A classic example of a classification task within Machine Learning is predicting handwritten digits from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) containing 70000 images. Optical Character Recognition software is important to many industries - today close to 99 % of letters sent via post are sorted automatically! We will revist this task later with more powerful Deep Neural Networks and Convolutional Neural Networks but it is a good idea to first work with it in the scope of logistic regression. 

MNIST IMAGE HERE

We shall first use logistic regression for a *binary classification* task before then using its generalised version softmax regression for *multiclass classification*.

## Mathematical foundations
We can understand the setup for logistic regression in terms having input data $$\boldsymbol{x}^{(i)}$$ of some form and labelled response $$y_i = \{0,1\}$$. We therefore want a model that will take our input data and yield a result between 0 and 1, ideally with data with $$y_i = 0$$ as close to 0 as possible and with data with $$y_i = 1$$ as close to 1 as possible. In reality we expect that our outputs will be spread out between somewhat - this actually gives our model the ability to give probabilities which is no bad thing.

There are many functions which can give an output ranging from 0 to 1 but the classic example and most commonly used is the logistic or sigmoid or logistic function
\begin{equation}
\sigma(s) = \frac{1}{1 + \mathrm{e}^{-s}}.
\end{equation}
Importantly this goes to $$0$$ for $$s \to -\infty$$ and to $$1$$ for $$s \to \infty$$. 

<img src="{{ page.root }}/fig/Logistic-curve.png" alt="drawing" width="500"/>

<!-- ![FCN diagram]({{ page.root }}/fig/Logistic-curve.png)> -->

> ## Historical note
>
> Earlier binary classification models often returned an output of either 0 or 1 - known as *hard classification*.\
> Now almost all binary classification models, such as logistic regression, return an output ranging between 0 and 1 - known as *soft classification*. We can always round up or down if we want a final answer but with soft classification you also get a probability.
{: .callout}

The logistic model actually can be used to return the estimated probabilities of any datapoint belonging to either class $$y_i$$ given the value of that datapoint $$ \boldsymbol{x}^{(i)}$$ and the parameters of the model $$\boldsymbol{\theta}$$. The formal name for this type of probability is a *conditional probability* but don't worry if that doesn't mean anything to you yet! The probability of the model predicting a "positive" result of $$y_i=1$$ given our model and inputs is
\begin{align}
P(y_i=1|\boldsymbol{x}^{(i)},\boldsymbol\theta) = \sigma(\boldsymbol{\theta}^\mathsf{T} \boldsymbol{x}^{(i)}) = \frac{1}{1+\mathrm{e}^{-\boldsymbol{\theta}^\mathsf{T} \boldsymbol{x}^{(i)}}}. 
\end{align}
We can also easily find the probability of the model predicting a "negative" result of $$y_i=0$$ from this
\begin{align}
P(y_i=0|\boldsymbol{x}^{(i)},\boldsymbol\theta) &= 1 - P(y_i=1|\boldsymbol{x}^{(i)},\boldsymbol\theta).
\end{align}

These probabilities tell us what our model will output but are also important for understanding what our cost function will be. Remember that the value for the parameter $$\boldsymbol\theta$$ will be found through gradient descent, for which we need the cost function. Before deriving the cost function let's first make sure that we understand what our model is predicting:

> ## Worked excercise
>
> If we have data from two variables $$x_1$$ and $$x_2$$ distributed as shown below. When will our model predict that the data belongs to either class?
> <img src="{{ page.root }}/fig/Decision-boundary.png" alt="drawing" width="250"/>
> Our model should predict that $$P(y_i=1|\boldsymbol{x}^{(i)},\boldsymbol\theta) = g(\boldsymbol\theta^\mathsf{T} \boldsymbol x ) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2)$$. Now suppose that our model after training learns parameters 
> <p style="text-align: center;">$$\begin{align}
    \boldsymbol \theta &= \begin{bmatrix}
           -3 \\
           1 \\
           1 \\
         \end{bmatrix}
  \end{align}$$</p>
> Substituting these values we have $$g(\theta_0 + \theta_1 x_1 + \theta_2 x_2) = g(-3 + x_1 + x_2)$$. From the definition of the sigmoid function we can see that $$\sigma(s) \geq 0.5$$ when $$s \geq 0$$ and $$\sigma(s) < 0.5$$ when $$s < 0$$. 
>
> For us this means that if we round the results of our model to 0 or 1, it will predict $$y=1$$ if $$-3 + x_1 + x_2 \geq 0$$ and $$y=0$$ if $$-3 + x_1 + x_2 < 0$$. That gives us 
\begin{equation}
\text{predict } y_i=1 \text{ if }  = x_1 + x_2 \geq 3
\end{equation}
\begin{equation}
\text{predict } y_i=0 \text{ if }  = x_1 + x_2 < 3
\end{equation}
We can draw such a *decision boundary* on the graph as is shown above.
{: .callout}

> ## Excercise
>
> Now what if we had data distributed non-linearly as shown below?
> <img src="{{ page.root }}/fig/Decision-boundary-nonlinear.png" alt="drawing" width="250"/>
> We can use a polynomial model which predicts $$P(y_i=1|\boldsymbol{x}^{(i)},\boldsymbol\theta) = g(\boldsymbol\theta^\mathsf{T} \boldsymbol x ) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2)$$. Suppose that our model has parameters
> <p style="text-align: center;">$$\begin{align}
    \boldsymbol \theta &= \begin{bmatrix}
           -1 \\
           0 \\
           0 \\
           1 \\
           1
         \end{bmatrix}
  \end{align}$$</p>
> Now find the decision boundary which our model defines.
{: .callout}

Now let's go ahead and find the cost function. We shall actually take a different approach to deriving it than we did for linear regression through considering *likelihoods*. Remember that the goal of the cost function is to find the optimal model parameters $$\boldsymbol{\theta}$$; we shall find a function that does just that.

First though, a bit of statistics background: the likelihood function gives the likelihood or probability of our model being correct given response data $$y_i$$. It is generally written $$L(\theta \mid y)$$ - or in our case $$L(\boldsymbol{x}^{(i)},\boldsymbol\theta \mid y_i)$$ - notice how this is different to our conditional probability $$P(y \mid \theta)$$ - or in our case $$P(y_i \mid \boldsymbol{x}^{(i)},\boldsymbol\theta)$$.  

We can say that the likelihood for our model being correct given $$y_i=1$$ is  
\begin{equation}
L(\boldsymbol{X},\boldsymbol\theta, \mid y_i = 1) = \prod_{i=1}^n P(y_i=1|\boldsymbol{x}^{(i)}.\boldsymbol\theta)
\end{equation}
Notice that we use take the product here over all datapoints $$i$$ since we want the likelihood of obtaining the entire input dataset $$\boldsymbol{X} = \{x^{(i)}\}$$. We also have that the likelihood for our model being correct given $$y_0=0$$ is
\begin{equation}
L(\boldsymbol{X},\boldsymbol\theta, \mid y_i = 0) = \prod_{i=1}^n P(y_i=0|\boldsymbol{x}^{(i)}.\boldsymbol\theta)
\end{equation}

We can actually use a clever mathematical trick to write the likelihoods in a compact notation:
\begin{equation}
L(\boldsymbol{X},\boldsymbol\theta \mid y_i) = \prod_{i=1}^n P(y_i=1|\boldsymbol{x}^{(i)},\boldsymbol\theta)^{y_i} P(y_i=0|\boldsymbol{x}^{(i)},\boldsymbol\theta)^{1 - y_i}.
\end{equation}
You should be able to plug in $$y_i=0$$ or $$y_i=1$$ to this equation to obtain the two equations above.

Statisticians often prefer to work with the log-likelihood instead of the likelihood. Ours will be
\begin{equation}
\ln L(\boldsymbol{\theta}) = \sum_{i=1}^n y_i \ln \left( \sigma ( \boldsymbol{\theta}^\mathsf{T} \boldsymbol{x} ) \right) + (1 - y_i) \ln \left(1 -  \sigma ( \boldsymbol{\theta}^\mathsf{T} \boldsymbol{x} ) \right).
\end{equation}
Note that $$L(\boldsymbol{\theta}) = L(\boldsymbol{X},\boldsymbol\theta \mid y_i)$$, we often just drop the other symbols for convenience. You may be wondering where this is going, but if you are familiar with statistics then you may have heard of *maximum likelihood estimation* (MLE). MLE is a popular technique for finding the best estimate the parameters of a model by maximising the likelihood function given some observed data. Now if this were a pure statistics course then we would go ahead and perform MLE here to find the optimal value for $$\boldsymbol{\theta}$$, however we know have a prescription to do so with gradient descent which will be much less computationally expensive for big datasets.

If we take our loss to be the negative log-likelihood, then by minimising the loss function we are essentially maximising the likelihood, hence finding the optimal value for $$\boldsymbol{\theta}$$. The loss/cost function for logistic regression is therefore
\begin{equation}
l(\boldsymbol{\theta}) = - \sum_{i=1}^n y_i \ln \left( \sigma ( \boldsymbol{\theta}^\mathsf{T} \boldsymbol{x} ) \right) + (1 - y_i) \ln \left(1 -  \sigma ( \boldsymbol{\theta}^\mathsf{T} \boldsymbol{x} ) \right) .
\end{equation}
Within Statistics this quantity is also known as the cross-entropy. Since we are performing a binary classification task we call this the binary cross-entropy loss.

> ## Additional tip
>
> The loss function here can be easily generalised to include $$m$$ labels whereupon the classification problem is called *softmax regression*. In problems with more than two labels, in practice one often converts the $$y_i$$ labels into *one-hot encoded* values so that the algorithm does not think higher numbers more important than lower ones. 
{: .callout}

## Python excercise: MNIST classification
Now we shall put into practice our own classifier, working to predict what digits images from the MNIST dataset represent. You can follow along in the Jupyter notebook or you can run the code locally on your machine if you wish. We'll work through this excercise using Keras but there are also examples which you can follow using PyTorch and sklearn. Shown below is an example of the code that you will find in the notebook, but using softmax regression instead of logistic regression. The difference is minor and you will have a chance to run the code using logistic regression instead in the notebook.

Loading the data is straightforward as most ML libraries now come with the MNIST dataset already available. To load it in Keras do
~~~
from keras.datasets import mnist 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
~~~
{: .language-python}
We have 60000 training images and 10000 testing images. 

The MNIST images are 2D images of 28 x 28 pixels. We therefore want to flatten these images into 28*28 = 784 dimensional inputs. The images are in grayscale so we don't have to worry about the RGB colour dimensions here. We'll also want to normalise the images by their minimum and maximum values. The minimum value is 0 (pure black) and as the images are 8 bit, the maximum value is 255 (pure white). We can therefore do
~~~
input_dim = 28*28 
X_train = X_train.reshape(60000, input_dim) 
X_test = X_test.reshape(10000, input_dim) 
X_train /= 255 
X_test /= 255
~~~
{: .language-python}
Note that in future projects, it is generally better to use a more sophisticated normalisation method such as sklearns' `MinMaxScaler`, however as we know precisely the min and max values, dividing by 255 is good here. We also need to convert our labels into one-hot encoded values so that so that the algorithm does not think higher numbers more important than lower ones:
~~~
from keras.utils import np_utils 
Y_train = np_utils.to_categorical(y_train, nb_classes) 
Y_test = np_utils.to_categorical(y_test, nb_classes)
~~~
{: .language-python}

We can setup a model in Keras as such:
~~~
from keras.models import Sequential 
from keras.layers import Dense, Activation 
nb_classes = 10
output_dim = nb_classes
model = Sequential() 
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax')) 
~~~
{: .language-python}

Finally we compile and test the model:
~~~
batch_size = 128 
epochs = 20
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(X_test, Y_test)) 
score = model.evaluate(X_test, Y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])
~~~
{: .language-python}

In the notebook we will then go on to plot the losses and accuracies during training and discuss various performance metrics including ROC curves. You will also have a chance to try out different regularisations and see what effect they have on the performance.

## Final notes
Logistic and softmax regression are fundamental topics for understanding more complex machine learning methods, and you will see the sigmoid and softmax functions time and time again. Furthermore if you can get to grips with understanding the setup for a classification task and the performance metrics now, it will help a lot of the later topics click into place. 

Next lesson we shall look at Deep Neural Networks, for which you should now be well equipped!

{% include links.md %}
