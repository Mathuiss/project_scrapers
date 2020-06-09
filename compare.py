import numpy as np
import pandas as pd
import preprocessor
from predict import predict


name = "Amanda Nunes"
preprocessor.preprocess(name, 0.99)
predict(name)
