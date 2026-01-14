import argparse
from pathlib import Path

import pandas as pd

def load_raw(unzipped_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame,pd.DataFrame]:
    """
    Load raw M5 files.
    Supports both validation and evaluation variants.
    Returns: (sales, calendar, prices)
    """
    calendar_path = unzipped_dir/"calendar.csv"
    prices_path = unzipped_dir/"sell_prices.csv"

    sales_validation = unzipped_dir/"sales_train_validation.csv"
    sales_evaluation = unzipped_dir/"sales_train_evaluation.csv"

    if sales_validation.exists():
        sales_path = sales_validation
        variant = "validation"
    elif sales_evaluation.exists():
        sales_path = sales_evaluation
        variant = "evaluation"
    else:
        raise FileNotFoundError(
            "Could not find sales_train_validation.csv or sales_train_evaluation.csv"
        )
    
    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing {calendar_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing {prices_path}")

    print(f"[INFO] Using sales file: {sales_path.name} ({variant})")

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)

    return sales, calendar, prices