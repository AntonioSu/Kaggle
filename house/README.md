### Contest requirements
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home. In particular, this is 
**Regression task**.
this contest lay on **how to select key feature**.
For details, please check [kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### introduce file
- clear_data.py preprocess the feather,including missing value and some other feather engineering.
- excel2csv.py transform excel to csv 
- text2vec.py transform feature to one-hot 
- main.py the entrance of project, it will call 'FeatureEngineering' function 

### data folder
- data_description.txt describe feature of house 
- train.csv train file 
- test.csv test file
- test.xlsx predict the result of test.csv
- submission.csv submit the result to Kaggle

### How to train?
run main.py and then print accuracy.

