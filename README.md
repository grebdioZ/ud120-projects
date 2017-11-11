ud120-projects
==============

Starter project code for students taking Udacity ud120, with modifications applied during the course.

Forked this mainly for versioning of the final project which I enjoyed a lot. 


=============================================
BEST RESULTS INCLUDING EMAIL ARCHIVE ANALYSIS
=============================================
Using the provided data dictionary plus subject and address features from the provided email archive, the best results I could achieve (as of Nov 2017) using automatic feature selection and a DecisionTreeClassifier with  were:

Best result scores ( kFolds, averages for 6 folds ):
====================================================
* Accuracy  : 0.88988
* F1        : 0.78704
* Precision : 0.80000
* Recall    : 0.83333

External 'tester' module results:
=================================
*        Accuracy_ext: 0.87700
*              F1_ext: 0.73548
*       Precision_ext: 0.64528
*          Recall_ext: 0.85500

OPTIMIZATION_CRIT: F1_ext

FEATURES INITIALLY USED for evaluation: Num = 16374: ['emails_RCVD_From (x2990)', 'emails_SENT_Subject (x7275)', 'emails_SENT_To (x6109)']

FEATURES IN BEST CLASSIFIER: Num = 3: [u'emails_RCVD_From_lavorato@enron.com', u'emails_SENT_To_bob.butts@enron.com', u'emails_SENT_Subject_new']

FEATURES IN BEST CLASSIFIER, with importances: [(u'emails_SENT_To_bob.butts@enron.com', 0.435), (u'emails_RCVD_From_lavorato@enron.com', 0.382), (u'emails_SENT_Subject_new', 0.183)]

DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=4,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='best')
	Accuracy: 0.87700	Precision: 0.64528	Recall: 0.85500	F1: 0.73548	F2: 0.80282
	Total predictions: 5000	True positives:  855	False positives:  470	False negatives:  145	True negatives: 3530
	
	
===================================================
BEST RESULTS USING ONLY THE PROVIDED data_dict INFO
===================================================
		
Best result scores ( kFolds, averages for 6 folds ):
====================================================
* Accuracy  : 0.86190
* F1        : 0.57778
* Precision : 0.63889
* Recall    : 0.62500

External 'tester' module results:
================================
*        Accuracy_ext: 0.89122
*              F1_ext: 0.52545
*       Precision_ext: 0.50988
*          Recall_ext: 0.54200

OPTIMIZATION_CRIT: F1_ext

FEATURES INITIALLY USED for evaluation: Num = 22: ['shared_receipt_with_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'from_messages', 'exchange_with_poi', 'ratio_to_poi', 'ratio_from_poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees', 'other']*

FEATURES IN BEST CLASSIFIER: Num = 2: ['shared_receipt_with_poi', 'exchange_with_poi']*

FEATURES IN BEST CLASSIFIER, with importances: [('exchange_with_poi', 0.559), ('shared_receipt_with_poi', 0.441)]*

DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=4,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=42, splitter='best')
	Accuracy: 0.89122	Precision: 0.50988	Recall: 0.54200	F1: 0.52545	F2: 0.53526
	Total predictions: 9000	True positives:  542	False positives:  521	False negatives:  458	True negatives: 7479

*exchange_with_poi is a feature derived from the features that is high if a person's sent AND received emails both contain a high ratio of POI addresses/senders. It's basically sqrt( from_this_person_to_poi/to_messages * from_poi_to_this_person / from_messages )