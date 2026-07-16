## Simple experiment on neural networks

At first I created my own network in numpy with formulas for gradient and selfwritten back and forward propagation.

My results:
95,63% accuracy on training set 
96% accuracy on test set

For just 3 epochs

You can see visualised certain errors and mistakes that my network did on test set, I chose 16 examples to visualize. You can see that in visualize_errors.py not really suggestive name.

Then I recreated my network in pytorch and by choosing effective learning rate which was lr = 1,1 suprisingly I trained my network firstly with 10 thousands epochs which caused accuracy equal to 100% and then I changed number of epochs to just 1 thousand.

for 1000 epochs I got results:
98,54% accuracy on training set
97,69% accuracy on test set

I was curious if my network was overfitting so here are results for 10 000 epochs. I think it is not hard to notice that my learning rate was pretty big.
100% accuracy on training set
97,64% accuracy on test set

So i got less accuracy with much much more epochs, just look at differences in percentage points. 
1000 epochs:
0,85 percantage points difference
10 000 epochs:
2,36 percantage points difference