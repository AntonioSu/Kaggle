### Contest requirements
In this contest, you need complete the analysis of what sorts of people were likely to survive. In particular, this is **Binary classification task**.
this contest lay on **how to process missing value**.
For details, please check [kaggle](https://www.kaggle.com/c/titanic)
### introduce file
- DealMiss.py preprocess the feather,including missing value and some other feather engineering.
- Excel2Csv.py transform excel to csv 
- text2vec.py transform feature to one-hot 
- main.py the entrance of project, it will call 'FeatureEngineering' function 

### data folder
- train.csv train file 
- test.csv test file
- test.xlsx predict the result of test.csv
- gender_submission.scv submit the result to Kaggle

### How to train?
run main.py and then print accuracy.

