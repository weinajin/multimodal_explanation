import torch
from sklearn import metrics
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)



def precision(output, target):
    output, target = get_numpy(output, target)
    return metrics.precision_score(target, output, average=None)


def recall(output, target):
    output, target = get_numpy(output, target)
    return metrics.recall_score(target, output, average=None)

def f1(output, target):
    output, target = get_numpy(output, target)
    return metrics.f1_score(target, output, labels = [0,1])#, average = None)

def prauc(output, target):
    # precision_recall_auroc
    output, target = get_numpy(output, target)
    precision, recall, thresholds = metrics.precision_recall_curve(target, output, pos_label= 0) # for BRATS_HGG task, LGG =0 is the minority class
    # calculate precision-recall AUC
    prauc = metrics.auc(recall, precision)
    return prauc

def auroc(output, target):
    output, target = get_numpy(output, target)
    return metrics.roc_auc_score(target, output, labels = [0,1], average = None)

def cm(output, target):
    output, target = get_numpy(output, target)
    return metrics.confusion_matrix( target, output, labels=[0,1])

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def get_numpy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        output = pred.cpu().numpy()
        target = target.cpu().numpy()
    return output, target


#### SUMMARY METRICS FOR VAL #####

def get_output_target(test_csv, result_file, label, label_dict):
    # groundtruth
    test_df = pd.read_csv(test_csv)

    test_df = test_df[test_df[label].notna()]  # get rid of data without gt gt
    gt_dict = label_dict
    gt = [gt_dict[test_df.loc[idx, label]] for idx in range(len(test_df))]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.argmax(prob))
    return pred, gt

def avg_precision(test_csv, result_file):
    pred, gt = get_output_target(test_csv, result_file)
    # record muli class precision
    precision = metrics.precision_score(gt, pred, average=None)
    return precision

def avg_recall(test_csv, result_file):
    pred, gt = get_output_target(test_csv, result_file)
    # record muli class recall
    recall = metrics.recall_score(gt, pred, average=None)
    return recall

def confusion_matrix(test_csv, result_file):
    pred, gt = get_output_target(test_csv, result_file)
    return metrics.confusion_matrix(gt, pred, labels=[0,1])

def metrics_all(test_csv, result_file, label, label_dict):
    labels = [0,1]
    pred, gt = get_output_target(test_csv, result_file, label, label_dict)
    # metrics
    recall = metrics.recall_score(gt, pred, average=None)
    precision = metrics.precision_score(gt, pred, average=None)
    cm = metrics.confusion_matrix(gt, pred, labels=labels)
    f1 =  metrics.f1_score(gt, pred, labels = labels)
    auroc = metrics.roc_auc_score(gt, pred, average = None)
    return precision, recall, f1, auroc, cm