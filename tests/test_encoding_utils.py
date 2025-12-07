import unittest
import pandas as pd
from utils.EncodingUtils import EncodingUtils   # <-- adjust import if needed


class TestEncodingUtils(unittest.TestCase):

    def test_infer_numeric(self):
        vals = [1, 2, 3.5, 4]
        col_type, processed = EncodingUtils.infer_column_type(vals)
        self.assertEqual(col_type, "numeric")
        self.assertEqual(sorted(processed), [1.0, 2.0, 3.5, 4.0])

    def test_infer_categorical_dates(self):
        vals = ["12/25/2021", "2021-12-25"]
        col_type, processed = EncodingUtils.infer_column_type(vals)
        self.assertEqual(col_type, "date")
        self.assertEqual(sorted(processed), ["20211225","20211225"])

    def test_encode_numeric(self):
        encoded = EncodingUtils.encode_value("3.5", "numeric")
        self.assertEqual(encoded, 3.5)

    def test_encode_categorical(self):
        encoded = EncodingUtils.encode_value("dog", "categorical")
        self.assertEqual(encoded, "dog")

    def test_encode_dataframe(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["2021-12-25", "12/25/2021", "random"]
        })

        col_types = {
            "a": "numeric",
            "b": "date",
        }

        df_encoded = EncodingUtils.encode_dataframe(df, col_types)

        self.assertEqual(df_encoded["a"].iloc[0], 1.0)
        self.assertEqual(df_encoded["b"].iloc[0], "20211225")
        self.assertEqual(df_encoded["b"].iloc[2], "random")


if __name__ == "__main__":
    unittest.main()
