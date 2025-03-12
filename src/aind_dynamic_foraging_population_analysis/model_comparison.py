"""Population model comparison"""

import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['font.family'] = 'Arial'

import numpy as np
import pandas as pd
import seaborn as sns

from aind_analysis_arch_result_access.han_pipeline import get_session_table, get_mle_model_fitting
from aind_analysis_arch_result_access.util.s3 import get_s3_pkl, get_s3_json

import logging
logger = logging.getLogger(__name__)


def get_all_model_metrics(use_cache=True, cache_path="~/capsule/data/df_model_fitting_all.pkl"):
    """Get all model metrics from either cache or result access API.

    Parameters
    ----------
    use_cache : bool, optional
        Whether to use cached data, by default True
        If true, it will load data from the cache. If cache does not exist 
           or is invalid, it will fetch data from the API.
        If False, it will fetch data from the API and update the cache_path.
    cache_path : str, optional
        Cache path, by default "~/capsule/data/df_model_fitting_all.pkl"
    """
    
    if use_cache:
        try:
            logger.info(f"Trying to load data from cache: {cache_path}...")
            df_model_fitting = pd.read_pickle(cache_path)
            logger.info("Done!")
            return df_model_fitting
        except Exception as e:
            logger.warning(f"Cache not found or invalid: {e}. Fetching from API.")
    
    # Fetch from result access API
    logger.info("Fetching data from result access API...")
    df_model_fitting = get_mle_model_fitting(
        from_custom_query={"status": "success"},
        if_include_latent_variables=False,
        paginate_settings={"paginate": True, "paginate_batch_size": 5000},
    )
    df_model_fitting.to_pickle(cache_path)
    logger.info(f"Data fetched and cached successfully to {cache_path}.")
    return df_model_fitting



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(filename)s:%(lineno)d - %(levelname)s - %(message)s')


    df_model_fitting = get_all_model_metrics()
    