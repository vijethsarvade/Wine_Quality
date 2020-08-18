import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dia=pd.read_csv("Data/datasets_228_482_diabetes.csv")
dia.shape
dia.columns                
dia.info()             
pd.options.display.max_columns= None
dia.describe()