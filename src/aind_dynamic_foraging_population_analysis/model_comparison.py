"""Population model comparison"""

import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
# plt.rcParams['font.family'] = 'Arial'

import numpy as np
import pandas as pd
import seaborn as sns

from aind_analysis_arch_result_access.han_pipeline import (
    get_session_table,
    get_mle_model_fitting,
)
from aind_analysis_arch_result_access.util.s3 import get_s3_pkl, get_s3_json

import logging

logger = logging.getLogger(__name__)

SESSION_KEYS = ["subject_id", "session_date"]


def get_all_model_metrics(
    use_cache=True, cache_path="~/capsule/data/df_model_fitting_all.pkl"
):
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
            logger.info(f"{len(df_model_fitting)} rows loaded from cache.")
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
    logger.info(f"{len(df_model_fitting)} rows fetched from API and saved to cache.")
    return df_model_fitting


def enrich_with_df_session(df, selected_fields):
    """Enrich any df with session information from get_session_table.

    Parameters
    ----------
    df: pd.DataFrame
        Any dataFrame containing SESSION_KEYS (["subject_id", "session_date"])
    selected_fields: list of str
        Fields to merge from session table.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with selected session information.
    """
    logger.info("Fetching session table...")
    df_session = get_session_table()

    logger.info("Merging model fitting data with session data...")
    # Merge in session metadata
    df_session["session_date"] = df_session["session_date"].astype("str")
    df_enriched = df.merge(
        df_session[SESSION_KEYS + selected_fields],
        on=SESSION_KEYS,
        how="left",
    )
    return df_enriched


def plot_best_model_counts(df, metrics, ax):
    """
    Plot the counts of best model variants based on a specified metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model metrics and agent_alias.
    metrics : str
        The metric to determine the best model (e.g., 'AIC', 'BIC').
    ax : matplotlib.axes.Axes
        The axes to plot on.
    
    """
    # Count the number of best model for each session
    session_keys = ["subject_id", "session_date"]
    best_models = df.loc[df.groupby(session_keys)[metrics].idxmin()]

    # Ensure all required models are present, even if never best
    best_models_count = best_models['agent_alias'].value_counts()
    all_aliases = df['agent_alias'].unique()
    best_models_count = best_models_count.reindex(all_aliases, fill_value=0)
    best_models_count.plot(kind='bar', ax=ax)
    # Sort bars by count (descending)
    best_models_count = best_models_count.sort_values(ascending=False)
    ax.clear()
    best_models_count.plot(kind='bar', ax=ax)
    ax.set_xlabel('Model variant')
    plt.xticks(rotation=45, ha='right')
    sns.despine(trim=True)
    ax.set(
        ylabel=f"Number of sessions where\n the model is the best \n(based on {metrics})",
        title=f"n = {best_models_count.sum()} sessions"
    )
    return ax


def compare_models_across_curriculum(df, metrics):
    """
    Compare best model counts across different curriculums and stages.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing model metrics and agent_alias.
    metrics : str
        The metric to determine the best model (e.g., 'AIC', 'BIC').
    """
    unique_curriculums = df['curriculum_name'].dropna().unique()
    unique_stages = df['current_stage_actual'].dropna().unique()
    stage_order = [None, 'STAGE_1_WARMUP', 'STAGE_1', 'STAGE_2', 'STAGE_3', 'STAGE_4', 'STAGE_FINAL', 'GRADUATED']

    # Convert all to string for comparison, handle None as 'None'
    unique_stages = sorted(unique_stages, key=lambda x: stage_order.index(x) if x in stage_order else len(stage_order))
    n_rows = len(unique_curriculums)
    n_cols = len(unique_stages)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for i, curriculum in enumerate(unique_curriculums):
        for j, stage in enumerate(unique_stages):
            df_sub = df[
                (df['curriculum_name'] == curriculum) &
                (df['current_stage_actual'] == stage)
            ]
            ax = axes[i, j]
            if not df_sub.empty:
                plot_best_model_counts(df_sub, metrics, ax)
            ax.set_title(f"{curriculum}\n{stage}")
            ax.set_xlabel('Model variant')
            n_sessions = df_sub[["subject_id", "session_date"]].drop_duplicates().shape[0]
            ax.set_title(f"{curriculum}\n{stage}\nn={n_sessions} sessions")
            ax.set_ylabel('Best model count')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    sns.despine(trim=True)
    return fig, axes


def _subtract_baseline(group, baseline_models=["LossCounting"], metric="AIC"):
    """Subtract baseline metric from each model in the group.

    Parameters
    ----------
    group : group of pd.DataFrame
        DataFrame containing model metrics and agent_alias.
    baseline_models : list of str, optional
        The mean of these models will be used as baseline, by default ['LossCounting']
    metric : str, optional
        The metric to subtract, by default 'AIC'
    """
    baseline = group.loc[group["agent_alias"].isin(baseline_models), metric].mean()
    group[f"delta_{metric}"] = group[metric] - baseline
    return group


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    )

    df_model_fitting = get_all_model_metrics(use_cache=True)
    df_model_fitting = enrich_with_df_session(df_model_fitting)

    n_models = df_model_fitting.value_counts("nwb_name")

    # Filter sessions that have all 30 models fitted
    df_filtered = df_model_fitting[
        df_model_fitting["nwb_name"].isin(n_models[n_models == 30].index)
    ]

    # Filtered on curriculum version and stages
    df_model_fitting_30_filtered = df_model_fitting_30_filtered.query(
        "curriculum_version_group == 'v3' &"
        "current_stage_actual in ['STAGE_FINAL', 'GRADUATED']"
    )
