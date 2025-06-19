from metaflow import step, parameters,IncludeFile
from io import StringIO
import pandas as pd
import numpy as np




class DatasetMixin():

    dataset = IncludeFile('data',
                         help ='The path for the input file',
                         default = '/Users/kshitiztiwari/my_ml_school/data/penguins.csv')
    
    def load_dataset(self):
        data = pd.read_csv(StringIO(self.dataset))
        data['sex'] = data['sex'].replace('.',np.nan)

        generator = np.random.default_rng(seed = 42)
        data = data.sample(frac=1, random_state = generator)

        return data
    

def build_feature_transformer():
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numerical_transformer = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )


    feature_transformer = ColumnTransformer(
        transformers= [
            ('numerical', numerical_transformer, make_column_selector(dtype_exclude ='object')),
            ('categorical', categorical_transformer, ['sex','island'])

        ]
    )

    return feature_transformer

def build_target_transformer():
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer

    return ColumnTransformer(
        [('target', OrdinalEncoder(), ['species'])]
    )

def build_model():
    from xgboost import XGBClassifier
    
    model = XGBClassifier(learning_rate = 0.1,
                          n_estimators = 2,
                          max_depth = 3)

    return model
    


    
    

