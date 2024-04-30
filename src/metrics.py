import json
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
def calculate_metrics(true_sentiments, predicted_labels, response_times):
    metrics_ = precision_recall_fscore_support(true_sentiments, predicted_labels, average='weighted')
    F1_score_ = 2/(1/metrics_[0] + 1/metrics_[1])

    # Print metrics
    print(" Precision:", metrics_[0])
    print(" Recall:", metrics_[1])
    print(" F1-Score:", F1_score_)


    # Print response time stats
    response_times = np.array(response_times)
    print("Average response time: ", response_times.mean())
    print("Max response time: ", np.max(response_times))
    print("Min response time: ", np.min(response_times))
    print("Median response time: ", np.median(response_times))
