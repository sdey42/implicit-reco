# implicit-reco

DATA:  
https://www.instacart.com/datasets/grocery-shopping-2017

DATA SCHEMA:  
https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b

PROBLEM DESCRIPTION:  
The dataset contains transactional information about what products users bought in the past. There are about 206k users and 50k products. 
A user may have bought a product multiple times, so in this case the pair <user_x, product_y> will appear multiple times in your dataset.  

**Goal is to predict which products users will likely buy.**

APPROACH:  
Built an Implicit Feedback recommender system that uses 70% training data, 30% test.

Furthermore, care was taken to generate the train/test sets as follows:
* every user + every product appears both in the _train_ and in the _test_ set,
* if a pair <user_x, product_y> appears in the _train_ing set at least once, it must not appear in the _test_ set,
* in the _test_ set, for each user, inject additonal 1000 random products that the particular user has not bought yet
 
EVALUATION:  
Accuracy of the recommender model is measured on the test set using Mean Percentile Ranking (MPR).
