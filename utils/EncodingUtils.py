import numpy as np
import pandas as pd
from datetime import datetime
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant
)
from ConfigSpace import ConfigurationSpace

class EncodingUtils:
    """Shared encoding utilities to ensure consistency across all classes"""
    
    @staticmethod
    def try_parse_date(s):
        """Parse date string to numeric YYYYMMDD format"""
        if not isinstance(s, str):
            return None
        for fmt in ["%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y"]:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.year * 10000 + dt.month * 100 + dt.day
            except Exception:
                pass
        return None
    
    @staticmethod
    def infer_column_type(values):
        """
        Determine column type consistently:
        - 'date': parseable date strings
        - 'numeric': numeric values
        - 'categorical': everything else
        """
        values = list(set(values))
        
        # Check if all are dates
        parsed_dates = []
        all_dates = True
        for v in values:
            date_val = EncodingUtils.try_parse_date(v)
            if date_val is None:
                all_dates = False
                break
            parsed_dates.append(date_val)
        
        if all_dates:
            return 'date', parsed_dates
        
        # Check if all numeric
        if all(isinstance(v, (int, float, np.number)) for v in values):
            return 'numeric', [float(v) for v in values]
        
        # Otherwise categorical
        return 'categorical', values
    
    @staticmethod
    def encode_value(value, col_type):
        """Encode a value based on column type"""
        if value == "?" or pd.isna(value):
            return "?"
        
        if col_type == 'date':
            # Convert date string to numeric
            parsed = EncodingUtils.try_parse_date(value)
            return float(parsed) if parsed is not None else value
        elif col_type == 'numeric':
            return float(value)
        else:  # categorical
            return str(value)
    
    @staticmethod
    def encode_dataframe(df, column_types):
        """
        Transform a DataFrame to use consistent encoding.
        Returns a new DataFrame with encoded values.
        """
        df_encoded = df.copy()
        for col in df.columns:
            if col in column_types:
                col_type = column_types[col]
                df_encoded[col] = df[col].apply(
                    lambda x: EncodingUtils.encode_value(x, col_type)
                )
        return df_encoded