import numpy as np
import pandas as pd
import os
import logging
import sys

logger = logging.getLogger(__name__)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

CFG = {
    'feature_names': [
        'product_code', 'product_type_no', 'graphical_appearance_no',
        'perceived_colour_value_id', 'index_code', 'section_no',
    ],
    'svd__n_components': 50,
    'svd__random_state': 17,
    'output_file': 'articles_vectors'
}


def preprocess_articles(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    return df.loc[:, feature_names]

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

def get_model(cfg: dict,) -> Pipeline:

    return Pipeline([
        ('oh', OneHotEncoder(handle_unknown='error')),
        ('pca', TruncatedSVD(random_state=cfg['svd__random_state'], n_components=cfg['svd__n_components']))
    ])

import matplotlib.pyplot as plt

def pairplot(data: np.array, first_n: int=4, bins: int=20, **kwargs) -> tuple:

    f, ax = plt.subplots(first_n, first_n,)
    
    for i in range(first_n):
        for j in range(first_n):
            if i == j:
                ax[i][j].hist(X_svd[:, i], bins=bins, **kwargs)
            else:
                ax[i][j].scatter(X_svd[:, i], X_svd[:, j], **kwargs)
    
    plt.tight_layout()
    return f, ax

# https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
from time import perf_counter
from contextlib import contextmanager

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    
import datetime
import joblib

if __name__ == "__main__":
    
    articles = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/articles.csv')

    with catchtime() as time_to_preprocess:
        X = preprocess_articles(articles, feature_names=CFG['feature_names'])

    logger.info('Preprocess input, time taken {:.2f} s'.format(time_to_preprocess()))
    svd = get_model(CFG)
    
    with catchtime() as time_to_fit:
        svd.fit(X)
    logger.info('Fit performed, time taken {:.2f} s'.format(time_to_fit()))
    
    X_svd = svd.transform(X)
    vr = svd.named_steps.pca.explained_variance_ratio_
    ev = vr.sum()
    logger.info('Explained variance: {:.1f}%'.format(ev * 100))
    
    if ev < .5: logger.warning('Explained variance is less than 50%, try increase svd__n_components')
    
    plt.plot(vr.cumsum())
    plt.savefig('variance_ratio.png')
    
    f, ax = pairplot(X_svd, 4,)
    f.set_figheight(20)
    f.set_figwidth(20)
    plt.savefig('pairplot.png')
    
    today = str(datetime.date.today())
    suffix = 'n_components{}_{}'.format(CFG['svd__n_components'], today)
    
    logger.info('Saving vectors')
    
    vectors = pd.DataFrame(X_svd, index=articles.article_id)
    vectors.to_csv(f'vectors_{suffix}.csv')

    logger.info('Saving model')
    joblib.dump(svd, f'model_{suffix}.pkl')
               