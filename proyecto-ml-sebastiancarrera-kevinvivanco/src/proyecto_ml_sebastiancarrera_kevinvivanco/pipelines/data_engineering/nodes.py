"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 1.0.0
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_datasets(cleaned_dataset, customer_agg, RFM):
    # --- cleaned_dataset ---
    if "CustomerDOB" in cleaned_dataset.columns:
        cleaned_dataset = cleaned_dataset.drop(columns=["CustomerDOB"])
    
    # --- customer_agg ---
    if "Unnamed: 0" in customer_agg.columns:
        customer_agg = customer_agg.drop(columns=["Unnamed: 0"])
    
    # --- RFM ---
    drop_cols = [c for c in ["Unnamed: 0", "Segment"] if c in RFM.columns]
    RFM = RFM.drop(columns=drop_cols)
    
    return cleaned_dataset, customer_agg, RFM

