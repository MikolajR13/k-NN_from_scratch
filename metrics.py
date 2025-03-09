import numpy as np


class Metrics:

    def __init__(self):

        self.confusion_matrix = None
        self.labels = None
        self.accuracy = None
        self.precision = None
        self.recall = None

    def prepare_confusion_matrix(self, y_true, y_pred):

        if len(y_true) != len(y_pred):
            raise ValueError(f"Needed same length of y_pred and y_true")

        self.labels = np.unique(np.concatenate((y_true, y_pred)))
        # to cover not continuous labels numbers or different from int type of labels
        ids = {}
        for id, label in enumerate(self.labels):
            ids[label] = id

        dimensions = len(self.labels)
        matrix = np.zeros((dimensions, dimensions), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):

            true_id = ids[true_label]
            pred_id = ids[pred_label]
            matrix[true_id, pred_id] += 1

        self.confusion_matrix = matrix

        return matrix


    #https://home.agh.edu.pl/~pszwed/wiki/lib/exe/fetch.php?media=med:med-w04.pdf


    def _accuracy(self, y_true, y_pred):

        self.accuracy = np.sum(y_true == y_pred) / len(y_pred)
        return self.accuracy

    def _precision(self, y_true, y_pred, average='macro'):

        # precision per every class
        #self.prepare_confusion_matrix(y_true, y_pred) : if we want to follow the specification
        matrix = self.confusion_matrix

        # true_p is when pred and true are the same
        true_p = np.diag(matrix)
        # false_p are in columns per class
        false_p = np.sum(matrix, axis=0) - true_p
        # to avoid dividing by 0
        precisions = np.zeros_like(true_p, dtype=float)
        non_zero = (true_p + false_p) > 0
        precisions[non_zero] = true_p[non_zero] / (true_p[non_zero] + false_p[non_zero])

        if average == 'macro':
            self.precision = np.mean(precisions)
            return self.precision
        elif average == 'micro':
            # doesn't need to check if true_p + false_p are != 0 because confusion matrix cannot be empty
            self.precision = np.mean(sum(true_p)/(sum(true_p)+sum(false_p)))
            return self.precision
        else:
            self.precision = precisions
            return self.precision

    def _recall(self, y_true, y_pred, average='macro'):

        # the same as precision but we need to use FN instead of FP ( rows instead of columns )
        # recall for every class
        #self.prepare_confusion_matrix(y_true, y_pred) # if we want to follow the specification
        matrix = self.confusion_matrix

        # true_p is when pred and true are the same
        true_p = np.diag(matrix)
        # false_p are in columns per class
        false_n = np.sum(matrix, axis=1) - true_p
        # to avoid dividing by 0
        recalls = np.zeros_like(true_p, dtype=float)
        non_zero = (true_p + false_n) > 0
        recalls[non_zero] = true_p[non_zero] / (true_p[non_zero] + false_n[non_zero])

        if average == 'macro':
            self.recall = np.mean(recalls)
            return self.recall
        elif average == 'micro':
            # doesn't need to check if true_p + false_p are != 0 because confusion matrix cannot be empty
            self.recall = np.mean(sum(true_p) / (sum(true_p) + sum(false_n)))
            return self.recall
        else:
            self.recall = recalls
            return recalls


    def _f1_score(self, y_true, y_pred, average='macro'):
        # depends on average type and because we use precision and recall
        # used precision and recall values are already computed for chosen average type
        # harmonic mean of precision and recall
        if self.precision == 0 and self.recall == 0:
            return 0
        else:
            f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
            return f1

    def __call__(self, y_true, y_pred, average='macro'):

        self.prepare_confusion_matrix(y_true, y_pred)
        accuracy = self._accuracy(y_true, y_pred)
        precision = self._precision(None, None, average)
        recall = self._recall(None, None, average)
        # it is already computed with chosen average
        f1_score = self._f1_score(None, None, average)

        return {
            'Average type': average,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_score': f1_score,
            'Confusion_matrix': self.confusion_matrix
        }


