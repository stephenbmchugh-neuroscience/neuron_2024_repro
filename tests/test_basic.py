from src.preprocessing import basic_clean
import pandas as pd

def test_basic_clean_drops_duplicates_and_nans():
    df = pd.DataFrame({'a':[1,1,None], 'b':[2,2,3]})
    out = basic_clean(df)
    assert out.shape[0] == 1
