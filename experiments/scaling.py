"""Fit L(N)=E+A*N^(-alpha) using non-embedding parameter counts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,default=Path('results_table.csv'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=list(csv.DictReader(args.input.open()))
    fits={}
    for method in ['Vanilla','BLD']:
        selected=[r for r in rows if r['family']=='KDA' and r['method']==method]
        n=np.array([float(r['nonembedding'])/1e6 for r in selected])
        loss=np.array([float(r['loss']) for r in selected])
        params,_=curve_fit(lambda n,e,a,alpha:e+a*n**(-alpha),n,loss,
            p0=[2,10,.3],bounds=([0,0,0],[min(loss),np.inf,5]),maxfev=100000)
        fits[method]=dict(E=params[0],A=params[1],alpha=params[2],N_unit='million non-embedding parameters')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(fits,indent=2)+'\n')


if __name__=='__main__':main()
