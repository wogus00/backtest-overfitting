"""Paper-style reconstruction of all 13 figures in Bailey et al. (2015).

Figures 1 and 6-13 use the user's experiment outputs (Figure 1 schematic).
Figures 2-5 are clearly labeled PROXIES using the user's no-signal and
seasonal experiments, NOT reproductions of the paper's unavailable examples.
Figures 9/13 compare the IS-selected OOS Sharpe distribution with the
complete pooled OOS strategy-score distribution, recalculated from returns.

Run: python plotting_all_13.py
Requirements: numpy pandas matplotlib pyarrow
No strategy recalculation, no modification of existing files.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT_CANDIDATES = [
    HERE, HERE.parent,
    Path.home() / 'backtest-overfitting' / 'backtest-overfitting',
]

def locate_data():
    for root in ROOT_CANDIDATES:
        if (root/'results/no_signal/sharpe/cscv.parquet').exists() and (root/'results/seasonal/sharpe/cscv.parquet').exists():
            return root
    raise FileNotFoundError('Could not find experiment Parquet files. Edit ROOT_CANDIDATES near the top.')

ROOT = locate_data()
# Save all 13 plots inside the project's results/figures folder
OUT = ROOT / 'results' / 'figures'

# Create the folder if it doesn't already exist
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'figure.dpi':120})


def save(fig, number, slug):
    dest=OUT/f'Figure_{number:02d}_{slug}.png'
    fig.savefig(dest,dpi=250,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'Saved: {dest.name}')


def read_dataset(name):
    b=ROOT/'results'/name/'sharpe'
    return {
        'cscv':pd.read_parquet(b/'cscv.parquet'),
        'performance':pd.read_parquet(b/'performance.parquet'),
        'returns_path':ROOT/'results/returns'/f'{name}.parquet',
        'prices_path':ROOT/'data/prices/synthetic'/f'{name}.parquet',
    }


def figure1():
    pairs=list(combinations('ABCD',2))
    fig,ax=plt.subplots(figsize=(8.2,5.0))
    ax.axis('off')
    for j,letter in enumerate('ABCD'):
        ax.text(1.8+j*0.75,6.35,letter,ha='center',fontsize=11,weight='bold')
        ax.text(4.9+j*0.75,6.35,letter,ha='center',fontsize=11,weight='bold')
    ax.text(2.9,6.9,'IS',ha='center',fontsize=12)
    ax.text(6.0,6.9,'OOS',ha='center',fontsize=12)
    for i,pair in enumerate(pairs):
        comp=[v for v in 'ABCD' if v not in pair]
        for j,ch in enumerate(pair): ax.text(1.8+j*.75,5.6-i*.77,ch,ha='center',fontsize=11)
        for j,ch in enumerate(comp): ax.text(4.9+j*.75,5.6-i*.77,ch,ha='center',fontsize=11)
        ax.annotate('',xy=(4.45,5.57-i*.77),xytext=(3.25,5.57-i*.77),arrowprops={'arrowstyle':'->','color':'0.35','lw':.9})
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((1.35,.95),6.25,5.8,fill=False,lw=1))
    ax.plot([4.15,4.15],[.95,6.75],c='black',lw=.8)
    ax.set_xlim(.8,8);ax.set_ylim(.6,7.4)
    ax.set_title('CSCV: six symmetric IS/OOS combinations for S = 4',pad=12)
    save(fig,1,'CSCV_schematic')


def valid_cscv(c):
    return c.loc[np.isfinite(pd.to_numeric(c['performance_is'],errors='coerce')) & np.isfinite(pd.to_numeric(c['performance_oos'],errors='coerce'))]


def degradation(ax,c,title):
    v=valid_cscv(c)
    x=v.performance_is.to_numpy(float);y=v.performance_oos.to_numpy(float)
    ax.scatter(x,y,s=9,c='#702c29',alpha=.66,linewidth=0)
    if len(x)>1 and np.ptp(x)>1e-10:
        a,b=np.polyfit(x,y,1);z=np.linspace(x.min(),x.max(),120)
        ax.plot(z,a*z+b,color='black',lw=1.2)
        eq=f'OOS = {a:.2f} IS {b:+.2f}'
    else: eq='Regression unavailable'
    ax.axhline(0,color='grey',linewidth=.8,ls=':')
    ax.set(xlabel='SR IS',ylabel='SR OOS',title=title)
    ax.text(.03,.96,eq+'\n'+f'Prob(SR OOS < 0) = {(y<0).mean():.2f}',transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'.7','alpha':.85})
    return x,y


def logits(ax,c,title):
    l=pd.to_numeric(c['logit'],errors='coerce').to_numpy(float)
    l=l[np.isfinite(l)]
    if l.size==0: raise ValueError('No valid logit values')
    ax.hist(l,bins=min(35,max(10,int(np.sqrt(len(l))*2))),density=True,color='forestgreen',edgecolor='white',lw=.35)
    ax.axvline(0,c='red',ls='--',lw=1.1)
    ax.set(xlabel='Logits',ylabel='Frequency density',title=title)
    ax.text(.03,.95,f'Prob(overfit) = {(l<0).mean():.2f}\nCSCV splits = {len(l)}',transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'.7','alpha':.8})
    return l


def figure2or3(number,c,label):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,9),constrained_layout=True)
    degradation(ax1,c,'1 | OOS Performance Degradation')
    logits(ax2,c,'0 | Histogram of Rank Logits')
    fig.suptitle(f'Figure {number} — illustrative proxy using {label}\nNOT the original Figure {number} dataset',fontsize=12)
    save(fig,number,'illustrative_proxy')


def figure7or11(number,c,label):
    fig,ax=plt.subplots(figsize=(7.2,5.5));degradation(ax,c,f'{label}: OOS Performance Degradation')
    save(fig,number,'performance_degradation')


def figure8or12(number,c,label):
    fig,ax=plt.subplots(figsize=(7.2,5.5));logits(ax,c,f'{label}: Histogram of Rank Logits')
    save(fig,number,'logit_distribution')


def load_returns(path):
    df=pd.read_parquet(path)
    if df.empty: raise ValueError(f'Empty strategy returns: {path}')
    return df


def oos_distribution(returns,c,S=None):
    """Compute full OOS score distributions for the saved CSCV split order.

    The order matches itertools.combinations over contiguous blocks, as in
    the supplied cscv.py. No strategy backtests are rerun.
    """
    n_splits=len(c)
    choices=[s for s in range(2,34,2) if s<=len(returns) and comb(s,s//2)==n_splits]
    if S is None:
        if len(choices)!=1: raise ValueError(f'Cannot uniquely identify CSCV S from {n_splits} splits: {choices}')
        S=choices[0]
    blocks=np.array_split(np.arange(len(returns)),S)
    X=returns.to_numpy(dtype=np.float64,copy=False)
    pooled=[]
    for selected in combinations(range(S),S//2):
        selected=set(selected)
        idx=np.concatenate([blocks[k] for k in range(S) if k not in selected])
        # Vectorized annualized Sharpe, matching regular sample std with ddof=1.
        a=X[idx]
        mu=a.mean(axis=0)
        sd=a.std(axis=0,ddof=1)
        sr=np.divide(mu,sd,out=np.full_like(mu,np.nan),where=sd>0)*np.sqrt(252)
        pooled.append(sr[np.isfinite(sr)])
    return np.concatenate(pooled)


def dominance_axes(ax,c,benchmark,title):
    selected=pd.to_numeric(c['performance_oos'],errors='coerce').to_numpy(float)
    selected=selected[np.isfinite(selected)]
    benchmark=np.asarray(benchmark,dtype=float)
    benchmark=benchmark[np.isfinite(benchmark)]
    if not len(selected) or not len(benchmark): raise ValueError('No valid OOS scores for dominance graph')
    xx=np.linspace(min(selected.min(),benchmark.min()),max(selected.max(),benchmark.max()),500)
    fs=np.searchsorted(np.sort(selected),xx,side='right')/len(selected)
    fb=np.searchsorted(np.sort(benchmark),xx,side='right')/len(benchmark)
    # Positive area means selected distribution tends to dominate benchmark.
    area=np.r_[0,np.cumsum((fb[1:]-fs[1:]+fb[:-1]-fs[:-1])/2*np.diff(xx))]
    ax.plot(xx,fs,c='#c64b3c',lw=1.7,label='optimized (IS-selected)')
    ax.plot(xx,fb,c='#3d5f9d',lw=1.7,label='non-optimized (all OOS trials)')
    ax.set(xlabel='OOS Sharpe Ratio',ylabel='Cumulative probability',title=title,ylim=(0,1.02))
    ax2=ax.twinx();ax2.plot(xx,area,color='#438f67',lw=1.5,label='SD2')
    ax2.set_ylabel('Integrated CDF difference (SD2)')
    a,l=ax.get_legend_handles_labels();b,m=ax2.get_legend_handles_labels()
    ax.legend(a+b,l+m,loc='upper left',fontsize=8)


_OOS_CACHE = {}

def dominance_fig(number,c,returns,label,proxy=False):
    key = (id(returns), len(c))
    if key not in _OOS_CACHE:
        _OOS_CACHE[key] = oos_distribution(returns,c)
    benchmark = _OOS_CACHE[key]
    fig,ax=plt.subplots(figsize=(8,5.4))
    dominance_axes(ax,c,benchmark,f'{label}: OOS Cumulative Distribution')
    if proxy:
        fig.suptitle(f'Illustrative proxy for Figure {number}; original example data unavailable',fontsize=10)
    save(fig,number,'dominance_proxy' if proxy else 'stochastic_dominance')


def figure6or10(number,ds,label):
    p=pd.read_parquet(ds['prices_path'])
    prices=p['price'].to_numpy(dtype=float) if isinstance(p,pd.DataFrame) else np.asarray(p,dtype=float)
    returns=load_returns(ds['returns_path'])
    perf=ds['performance']
    best=perf.loc[pd.to_numeric(perf['sharpe'],errors='coerce').idxmax()]
    sid=best['strategy_id']
    if sid in returns: series=returns[sid]
    elif str(sid) in returns: series=returns[str(sid)]
    else: series=returns.iloc[:,int(sid)]
    r=series.to_numpy(dtype=float)
    wealth=np.r_[1,np.cumprod(1+r)]
    profit=wealth[1:]-1
    index=returns.index
    x=index if isinstance(index,pd.DatetimeIndex) else np.arange(len(profit))
    fig,ax=plt.subplots(figsize=(10,5.7))
    ax.plot(x,profit,c='#6d3fa1',lw=1.4,label='Accumulated strategy profit')
    ax.set(xlabel='Time',ylabel='Accumulated strategy profit (initial capital = 1)')
    ax2=ax.twinx();ax2.plot(x,prices[:len(profit)],c='#9a9a9a',alpha=.65,lw=1.15,label='Underlying price')
    ax2.set_ylabel('Underlying synthetic price')
    a,l=ax.get_legend_handles_labels();b,m=ax2.get_legend_handles_labels();ax.legend(a+b,l+m,loc='upper left',fontsize=8)
    ax.set_title(f'{label}: backtested performance of the selected strategy\nFull-sample best strategy ID {sid}; Sharpe {best["sharpe"]:.2f}')
    save(fig,number,'backtest_and_prices')


def main():
    print('Loading data from:',ROOT)
    print('Writing images to:',OUT)
    a=read_dataset('no_signal');b=read_dataset('seasonal')
    figure1()
    # Figures 2-5 rely on unavailable paper-specific examples. These
    # proxies are built from OUR experiments, and cannot be presented
    # as reproductions of the original paper's Figures 2-5.
    figure2or3(2,a['cscv'],'no-signal experiment')
    figure2or3(3,b['cscv'],'seasonal experiment')
    ra=load_returns(a['returns_path'])
    rb=load_returns(b['returns_path'])
    dominance_fig(4,a['cscv'],ra,'No-signal example',proxy=True)
    dominance_fig(5,b['cscv'],rb,'Seasonal example',proxy=True)
    figure6or10(6,a,'No signal')
    figure7or11(7,a['cscv'],'No signal')
    figure8or12(8,a['cscv'],'No signal')
    dominance_fig(9,a['cscv'],ra,'No signal')
    figure6or10(10,b,'Seasonal signal')
    figure7or11(11,b['cscv'],'Seasonal signal')
    figure8or12(12,b['cscv'],'Seasonal signal')
    dominance_fig(13,b['cscv'],rb,'Seasonal signal')
    print('\n13 numbered images saved. Figures 2-5 are explicitly marked illustrative proxies.')

if __name__=='__main__': main()
