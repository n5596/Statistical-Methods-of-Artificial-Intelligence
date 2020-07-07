## Assignment 2 Q1

### Part 1: Kernel Trick
	In this question, we are given a input data having 2 features and a non-linear separating boundary. Our goal is to apply a kernel in
such a manner that the data is mapped from 2 dimensions to 3 dimensions and becomes linearly separable in 3 dimensions. The kernels that I
have used are able to achieve this. 
	The first kernel that I have used takes in the 2 dimensional data X containing two features (x1, x2). It is mapped to 3 dimensions
by using the relation (x1, x2, x1^2 + x2^2). We can see that the new dimension x3 = x1^2 + x2^2 is a function of the first two dimensions. 
I have used this kernel with a perceptron of all three types of penalty functions - no penalty, L1 and L2. For the no penalty case, we can 
see that it makes the data linearly separable. For the L1 case, I noticed that the data becomes linearly separable as long as alpha is below
0.075. As long as this holds true, the performance metrics for training will all be 1.000. For the L2 case, I noticed that the data becomes
linearly separable as long as alpha is below 0.01. 
	The second kernel that I have used converts a 2 dimensional data X containing features (x1, x2) to three dimensions by using the 
relation (1.3*x1, x1+x2, sqrt(x1^2 + x2^2)). I have used this kernel with a perceptron of all three types of penalty functions- no penalty, 
L1 and L2. For the no penalty case, we can see that the data becomes linearly separable. For the L1 case, I noticed that the data becomes 
linearly separable as long as alpha is less than 0.012. For the L2 case, I noticed that the data becomes linearly separable as long as alpha 
is less than 0.0008. As long as data becomes linearly separable, the performance metrics will all be 1.000.
	The third kernel that I have used converts a 2 dimensional data X containing features (x1, x2) to three dimensions by using the 
relation (x1-x2, x2, sqrt(x1^2 + x2^2)). I have used this kernel with a perceptron of all three types of penalty functions- no penalty, L1
and L2. For the no penalty case, we can see that the kernel makes the data linearly separable. For the L1 case, I noticed that the data
becomes linearly separable as long as alpha is below 0.07. For the L2 case, I noticed that the data becomes linearly separable as long as
alpha is below 0.00005. As long as the conditions hold true, the performance metrics for training will all be 1.000.

### Part 2: Letter Classification
	In this question, we are supposed to do the english letter classification by using a SVM. Since the data is already divided into 5
train/validation splits, we have to output the average accuracy, precision, recall and f1 score. The kernels that I have used are 'Linear', 
'RBF' and 'Sigmoid'. We were expected to experiment with at least 3 kernels and to test out how the hyperparameter affect the performance.
	The first kernel that I have used is the Linear kernel. For this kernel, I got the best performance when I set C (penalty
parameter) to 1.07. The only hyperparameter that significantly affects the performance of the SVM when using the linear kernel is C (since
other hyperparameters like gamma and coef0 are for other kernels). I noticed that increasing values of C tend to perform better.
	The second kernel that I have used is the RBF kernel. For this kernel I got the best case at C = 13 and gamma = 0.1. The
hyperparameters that I experimented with are C and gamma. Gamma refers to the kernel coefficient which is primarily used for RBF, polynomial
and sigmoid kernels. The default value of gamma is 'auto' and in such a case 1/(number of features) will be used instead. I have observed
that as we increase the value of C from 1.0, the performance rises until a peak is reached at C = 13. Also, gamma set at 0.1 tends to give
better performance than gamma = 'auto' which is the default value.
	The third kernel that I have used is the Sigmoid kernel. For this kernel, I got the best performance when I kept C at 0.05, gamma = 
'auto' and coef0 = 0.000001. The hyperparameters which were used include C, gamma and coef0. Note that the default values of gamma and coef0
are 'auto' and 0.0, respectively. On experimenting with various values, I observed that the performance tends to improve on reducing C until
it reaches a peak at C = 0.05. Also, the performance tends to decrease on setting gamma to any numerical value other than its default. I also
observed that for smaller positive values of coef0, the performance tends to improve slightly.
