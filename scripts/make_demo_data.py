"""Generate synthetic data for software smoke tests only, never for research claims."""
import argparse, random
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--rows',type=int,default=600); p.add_argument('--output',default='data/demo/ragtruth_demo.csv'); a=p.parse_args()
    random.seed(42); rows=[]
    true_facts=[('The capital of France is Paris.','Paris is the capital of France.'),('Water freezes at 0 degrees Celsius under standard pressure.','At standard pressure, water freezes at 0°C.'),('The Earth orbits the Sun.','Earth revolves around the Sun.')]
    false_facts=[('The capital of France is Paris.','The capital of France is Lyon.'),('Water freezes at 0 degrees Celsius under standard pressure.','Water freezes at 40°C under standard pressure.'),('The Earth orbits the Sun.','The Sun orbits the Earth.')]
    for i in range(a.rows):
        y=i%2; q=f"Question {i}: state the relevant fact"; ctx,ans=random.choice(false_facts if y else true_facts)
        rows.append({'question':q,'context':ctx,'response':ans,'label':y,'source_id':f'source_{i%120}'})
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output,index=False); print(f'Wrote {len(rows)} synthetic smoke-test rows to {a.output}. Do not use these for research results.')
if __name__=='__main__': main()
