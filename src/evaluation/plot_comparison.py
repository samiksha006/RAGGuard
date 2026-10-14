"""Plot layer/model trade-offs from train_probes.py comparison.csv."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='artifacts/probes/comparison.csv'); p.add_argument('--output-dir',default='reports/results'); a=p.parse_args()
    df=pd.read_csv(a.input); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for metric,title,ylabel,name in [('f1','Layer-wise F1 comparison','F1 score','layer_f1.png'),('pr_auc','Layer-wise PR-AUC comparison','PR-AUC','layer_pr_auc.png'),('latency_ms_per_sample','Inference latency by layer and model','Milliseconds per sample','layer_latency.png')]:
        fig,ax=plt.subplots()
        for model,g in df.groupby('model'):
            g=g.sort_values('layer'); ax.plot(g.layer,g[metric],marker='o',label=model)
        ax.set(title=title,xlabel='Transformer layer index',ylabel=ylabel); ax.legend(); fig.tight_layout(); fig.savefig(out/name,dpi=160); plt.close(fig)
    print('Saved layer comparison plots to',out)
if __name__=='__main__': main()
