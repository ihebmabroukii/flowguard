"""Our own implementations, written with NumPy only (no ML library).

- SoftmaxRegression: multi-class logistic regression trained by mini-batch gradient descent
- MLPClassifier: fully connected network with ReLU, dropout and hand-written backpropagation
- IsolationForest: unsupervised anomaly detector (Liu, Ting and Zhou, 2008)
- TrivialClassifier: majority-class and random-guess baselines (the floor to beat)

The scikit-learn / XGBoost models elsewhere in flowguard.models are benchmarks: if our
implementation scores close to the library one on the same data, that is evidence it is correct.
"""

from flowguard.models.scratch.isolation_forest import IsolationForest
from flowguard.models.scratch.mlp import MLPClassifier
from flowguard.models.scratch.softmax import SoftmaxRegression
from flowguard.models.scratch.trivial import TrivialClassifier

__all__ = ["IsolationForest", "MLPClassifier", "SoftmaxRegression", "TrivialClassifier"]
