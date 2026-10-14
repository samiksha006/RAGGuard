import pandas as pd
from src.data.prepare import prepare_dataframe, normalize_label

def test_label_normalization():
    assert normalize_label(1)==1
    assert normalize_label('grounded')==0
    assert normalize_label('hallucinated')==1

def test_prepare_minimal():
    df=pd.DataFrame({'question':['q1','q2'],'context':['c1','c2'],'response':['a1','a2'],'label':[0,1],'source_id':['s1','s2']})
    out=prepare_dataframe(df)
    assert list(out.columns)==['row_id','question','context','answer','label','source_id']
    assert out.label.tolist()==[0,1]
