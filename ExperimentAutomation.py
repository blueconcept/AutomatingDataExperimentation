
# coding: utf-8

# In[1]:

import pandas as pd
import json
from collections import Counter

class ExperimentTest():
    
    def __init__(self, test_file_path, actual_file_paths):
        self.raw_test_data = self.get_test_data(test_file_path)
        self.source_names = self.get_source_names()
        self.actual_data = self.get_actual_data(actual_file_paths)
        self.sourced_data = self.get_reformated_test()
        self.confusion_scores = self.get_source_scores()
        self.metrics = self.get_metrics()
        
    def get_test_data(self, test_file_path):
        return json.load(open(test_file_path, 'r'))
    
    def get_actual_data(self, csv_names):
        df_list = map(pd.read_csv, csv_names)
        df = pd.concat(df_list,axis=0)
        df = df.set_index("unique_id").drop(["Name", "Address", "City", "State"], axis=1)
        dictionary = df.to_dict()
        return dictionary["is_closed"]
    
    def get_source_names(self):
        return list({ item['source'] for key,item in self.raw_test_data.iteritems() })
        
    def get_reformated_test(self):
        return { name:{ key:item["source_label_is_closed"] for key,item in self.raw_test_data.iteritems() if item["source"] == name} for name in self.source_names } 
    
    def get_source_scores(self):
        return { name:self.get_scores(name) for name in self.source_names }
    
    def get_scores(self, name):
        counter = {'tp':0, 'fp':0,'tn':0, 'fn':0}
        for index,pred in self.sourced_data[name].iteritems():
            if pred == 'Y':
                if self.actual_data[index] == 'Y':
                    counter['tp'] += 1
                else:
                    counter['fp'] += 1
            else:
                if self.actual_data[index] == 'N':
                    counter['tn'] += 1
                else:
                    counter['fn'] += 1
        return counter
    
    def get_metrics(self):
        return { name:self.get_source_metric(name) for name in self.source_names}
        
    def get_source_metric(self, name):
        mm = MathMetric(self.confusion_scores[name])
        scores = {"precision":mm.precision(), "recall":mm.recall(), "f1_score":mm.f1_score(), "accuracy":mm.accuracy(), "mathews_correlation":mm.matthews_correlation()}
        return scores
    
class MathMetric():
    
    def __init__(self, score_dict):
        self.tp = float(score_dict['tp'])
        self.fp = float(score_dict['fp'])
        self.tn = float(score_dict['tn'])
        self.fn = float(score_dict['fn'])
        
    def total(self):
        return self.tp+self.fp+self.tn+self.fn
        
    def precision(self):
        return self.tp / (self.tp+self.fp)
        
    def recall(self):
        return self.tp / (self.tp+self.fn)

    def f1_score(self):
        return 2 * self.recall() * self.precision() / (self.recall() + self.precision())

    def accuracy(self):
        return (self.tn + self.tp) / (self.total())

    def matthews_correlation(self):
        return ((self.tp * self.tn) - (self.fp * self.fn)) / pow((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn), 0.5)


# The ExperimentTest class is a class for taking in a file all of data sources to be tested and a list of ground truth data to be used as actual values.  This ExperimentTest class should work for more then 2 data sources to be experimented upon.
# 
# The MetricMath class is a class for calculating the metrics with the 4 primitive measures (tp, fp,...).  I coded this up myself because I wanted to add a matthews correlation function.  

# In[2]:

hidden = "hidden_info.json"
andres = "is_closed_classification_Andres.csv"
betty = "is_closed_classification_Betty.csv"
craig = "is_closed_classification_Craig.csv"
dana = "is_closed_classification_Dana.csv"
elena = "is_closed_classification_Elena.csv"
csv_names = [andres, betty, craig, dana, elena]
et = ExperimentTest(hidden, csv_names)
pd.DataFrame(et.confusion_scores)


# So this is the confusion matrix in pandas form.  From the results, it seems that the models have a trade off of false positives for false negatives.  Accutronix has more false positives then Verifidelity and the opposite is the case for false negative.  

# In[3]:

pd.DataFrame(et.metrics)


# Given the confusion matrix, it's no surprise that models trade precision and recall and have similiar performances otherwise.  Accuracy here isn't very useful because the classes are not balanced at all.  In fact, they are about 30% and 70%.
# 
# While the performances of the data sources are closely even, I would say that Accutronix is slightly better because it's slightly better with all metrics aside from precision where if precision were compared to the other data source's recall, Accutronix would be out winning.
# 
# The important metric here is Matthew's correlation metric which is a metric that is can be used even if the classes are inbalanced as they are in these cases.  For this metric 1 is a perfect predictor, 0 is random performance, and -1 is with no correct predictions and is related to chi-squared test.    
