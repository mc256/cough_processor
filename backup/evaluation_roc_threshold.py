import pickle

RUN = 'Dec19-balanced-30dB'
RUN = 'Dec19-30dB-ratio9'
MODEL = 49

with open('./results/%s-%d-roc-threshold.pkl' % (RUN, MODEL), 'rb') as handle:
    thresholds_roc = pickle.load(handle)
    print(thresholds_roc.shape)
    print(thresholds_roc[thresholds_roc.shape[0]//5])

'''
Old Data only trained on fine-grained
window 5
True positive rate/ Recall/ Sensitivity/ Probability of Detection: 0.7591240875912408
Precision: 0.014835948644793153
Fall-out/ Probability of false alarm: 0.3180778383357642
f1: 0.02910312019028963
Accuracy: 0.6824062490464712
True positive rate/ Recall/ Sensitivity/ Probability of Detection: 0.7591240875912408
Precision: 0.014835948644793153
'''

'''
segmented_train_set_dB30_balanced.csv
window 5
acc: 0.9657974592398293
f1: 0.07510431154381085
Precision: 0.11020408163265306
Recall: 0.056962025316455694
'''

'''
segmented_train_set_dB30_balanced.csv
window 3
acc: 0.7658283186751016
f1: 0.06259007617871114
Precision: 0.6204081632653061
Recall: 0.03295750216825672
'''

'''
segmented_train_set_dB30_ratio9.csv
window 3
acc: 0.5688936892454868
f1: 0.0470668485675307
Precision: 0.8448979591836735
Recall: 0.024207695006431995
'''