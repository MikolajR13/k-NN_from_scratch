import numpy as np
import pandas as pd

class Metrics:

    def accuracy(y_true, y_pred):
        """Oblicza dokładność klasyfikacji."""
        pass

    def precision(y_true, y_pred, average='macro'):
        """Oblicza precyzję. Obsługuje micro/macro averaging."""
        pass

    def recall(y_true, y_pred, average='macro'):
        """Oblicza recall. Obsługuje micro/macro averaging."""
        pass

    def f1_score(y_true, y_pred, average='macro'):
        """Oblicza F1-score. Obsługuje micro/macro averaging."""
        pass

    def __call__(self, y_true, y_pred, average='macro'):
        pass