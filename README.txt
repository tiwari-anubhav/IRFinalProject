Following are the steps to Run the scripts available:

1.) Install required libraries using pip install command.
The required libraries are: Pandas, nltk, sklearn, wordcloud, matplotlib.

2.) From the python terminal enter the following command to run the first script (LabelClasses.py) to label the classes. or use python IDE like pycharm.
>>> python LabelClass.py

3.) After that run the script named PreProcessNaiveBayes.py to run the naive bayes classification code.
>>> python PreProcessNaiveBayes.py
It will print the accuracy and classification report and return a wordcloud.
** The feature terms can be changed on lines 58-60.
** The wordcloud generation can be changed to "Positive" or "Negative" on line 152-153

4.) For testing the SVM classifier, Run the script SVMClassification.py
>>> python SVMClassification.py
** It outputs the accuracy/classification results of different configurations of SVM to a file called Results.txt
** For making the wordcloud uncomment lines 131-154
