"""Render the frozen research figure IDs; no adaptation or metric estimation."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
def read_rows(path):
    with (ROOT / path).open(newline='') as stream:
        return list(csv.DictReader(stream))
def load_values():
    values = json.loads((ROOT / 'figures/figure_values.json').read_text())
    for path, info in values['sources'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('Frozen source hash mismatch: ' + path)
        if len(read_rows(path)) != info['rows']:
            raise ValueError('Frozen source row count mismatch: ' + path)
    sources = {'2':'results/canonical/fixed_mu_metrics.csv', '3':'results/external/fixed_mu_metrics.csv', '5':'results/cross_dataset/cross_dataset_selector_per_domain.csv', 'S1':'results/cross_dataset/cross_dataset_mu_sensitivity.csv'}
    for figure, path in sources.items():
        rows = read_rows(path)
        # Match complete frozen records exactly, preserving strings and precision.
        for record in values['figures'][figure]['plotted_records']:
            if record not in rows:
                raise ValueError('Plotted record absent from CSV: ' + figure)
    # Figure 6 uses the already frozen per-cell count payload. Both its state-detail
    # and cohort-denominator CSV inputs are hash-validated above; no new summary.
    return values

def render(output, selected=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    O = Path(output)
    O.mkdir(parents=True, exist_ok=True)
    V = load_values()
    C = json.loads((ROOT/'figures/captions.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
    W=180/25.4
    def save(f,n):
        if selected is None or str(n) in selected:
            for ext in ['png','pdf']:f.savefig(O/f'figure{n}.{ext}',dpi=300)
        plt.close(f)
    def tidy(t):
        return t.replace('target-present macro-F1','Target-present Macro-F1').replace('same-support','same-label-support').replace('UCI primary same-support','UCI primary same-label-support')
    # Fixed physical page width. All source font sizes >=8pt; no tight-bbox shrink.
    f,ax=plt.subplots(figsize=(W,6.3));f.subplots_adjust(left=.02,right=.98,top=.93,bottom=.02);ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,text,color='#e7eff6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',facecolor=color,edgecolor='#34516a'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.3,'color':'#34516a'})
    box(.2,7.8,4.25,1.5,'Labeled source (Xₛ, yₛ)\nFit StandardScaler on Xₛ\nUnit-norm observations')
    box(5.55,7.8,4.25,1.5,'Unlabeled target block Xₜ\nWhole block available\nSource scaling + unit norm')
    box(1.1,5.45,7.8,1.55,'Canonical BDA candidates\nμ = 0.0, 0.1, …, 1.0\n1NN bootstrap before projection; 10 outputs')
    box(.2,2.9,4.25,1.7,'Label-free decisions\nFixed μ / mean or median margin /\nhard-onehot EnsV / multi-μ vote')
    box(5.55,2.9,4.25,1.7,'Freeze + hash\nDecisions and predictions fixed','#e6f1e9')
    box(.2,.2,3.5,1.55,'Held-out target labels yₜ\nOpened only for evaluation','#fff0db')
    box(4.65,.2,5.15,1.55,'Retrospective evaluation only\nTarget-present Macro-F1;\nsingle-μ oracle, support and gaps','#fff0db')
    for a,b in [((2.3,7.8),(3.3,7)),((7.7,7.8),(6.7,7)),((3.2,5.45),(2.4,4.6)),((4.45,3.7),(5.55,3.7)),((7.6,2.9),(7.6,1.75)),((3.7,.97),(4.65,.97))]:arrow(a,b)
    f.suptitle('Strict target-label-free protocol',fontsize=12,y=.98);save(f,'1')
    for n,key,title,h in [('2','target_batch','UCI same-label-support targets',8),('3','target','External reconstructed derived benchmark',10)]:
        records=V['figures'][n]['plotted_records'];targets=list(dict.fromkeys(r[key] for r in records))
        f,axs=plt.subplots((len(targets)+1)//2,2,figsize=(W,h),layout='constrained',sharex=True,sharey=True)
        for ax,t in zip(axs.flat,targets):
            rr=[r for r in records if r[key]==t]
            ax.plot([float(r['mu']) for r in rr],[float(r['target_present_macro_f1']) for r in rr],'-o',color='#2166ac',ms=3,lw=1.2)
            ax.set(title=('Batch '+t if n=='2' else t),xlim=(0,1),ylim=(0,1),xticks=np.arange(0,1.01,.2),yticks=np.arange(0,1.01,.2));ax.grid(alpha=.2);ax.tick_params(labelbottom=True,labelleft=True)
        for ax in list(axs.flat)[len(targets):]:ax.axis('off')
        f.suptitle(title,fontsize=11);f.supxlabel('Balance parameter μ',fontsize=9);f.supylabel('Target-present Macro-F1',fontsize=9);save(f,n)
    methods=['fixed_mu_0.0','fixed_mu_0.5','margin_mean','margin_median','EnsV-onehot','multi-mu-ensemble']
    labels=['Fixed\nμ = 0','Fixed\nμ = 0.5','Mean\nmargin','Median\nmargin','Hard-onehot\nEnsV','Multi-μ\nvote']
    colors=plt.get_cmap('tab10').colors;markers=['o','s','^','D','v','P','X']
    f,axs=plt.subplots(2,1,figsize=(W,8.5),sharey=True,layout='constrained')
    for ax,cohort,title in zip(axs,['UCI_PRIMARY','EXTERNAL_PRIMARY'],['UCI same-label-support (6 domains)','External derived (7 domains)']):
        records=[r for r in V['figures']['5']['plotted_records'] if r['cohort']==cohort];targets=list(dict.fromkeys(r['target'] for r in records))
        for j,t in enumerate(targets):
            ys=[float(next(r['oracle_gap'] for r in records if r['target']==t and r['method']==m)) for m in methods]
            ax.scatter(np.arange(6)+(j-(len(targets)-1)/2)*.07,ys,color=colors[j],marker=markers[j],s=34,label=('B'+t if cohort=='UCI_PRIMARY' else t),linewidths=.5,edgecolors='#222222',zorder=3)
        ax.axhline(0,color='black',lw=.8);ax.axhline(.15,color='#a33b20',ls='--',lw=1)
        ax.text(5.5,.157,'0.15 descriptive threshold',ha='right',fontsize=8)
        ax.set(title=title,xticks=range(6),xticklabels=labels,xlim=(-.5,5.5),ylim=(-.06,.35),yticks=[-.05,0,.1,.15,.2,.3],ylabel='Oracle gap (absolute F1)');ax.grid(axis='y',alpha=.2);ax.legend(ncol=4,loc='upper left',frameon=False,handletextpad=.4,columnspacing=.8)
    f.suptitle('Deployable strategies vs retrospective single-μ oracle',fontsize=11);save(f,'5')
    C['5']=C['5'].replace('Colors identify target domains and horizontal offsets separate points.','Distinct marker shapes and colors identify target domains; the same shape key remains readable in grayscale. Horizontal offsets separate points.').replace('UCI primary (6 domains)','UCI primary same-label-support (6 domains)')
    f,axs=plt.subplots(2,2,figsize=(W,8.2),gridspec_kw={'width_ratios':[11,1.6]},layout='constrained')
    for i,item in enumerate(V['figures']['6']['plotted_records']):
        grid=np.array(item['output_counts_by_target_mu']);boot=np.array(item['bootstrap_runs_by_target']);ss=item['summary'];bs=item['bootstrap_summary'];targets=item['targets']
        for ax,arr,high in [(axs[i,0],grid,10),(axs[i,1],boot,11)]:
            im=ax.imshow(arr,vmin=0,vmax=high,cmap='Blues',aspect='auto');ax.set_yticks(range(len(targets)),[('B'+t if i==0 else t) for t in targets]);ax.tick_params(length=0)
            for (y,x),v in np.ndenumerate(arr):ax.text(x,y,str(v),ha='center',va='center',fontsize=8,color='white' if v>=7 else '#122c43')
        axs[i,0].set(title=f"{item['dataset']}: outputs 1–10\n{ss['present_missing_run_count']}/{ss['run_count']} runs; {ss['present_missing_state_count']}/{ss['observed_state_count']} states",xticks=range(11),xticklabels=[f'{m/10:.1f}' for m in range(11)],xlabel='Balance parameter μ',ylabel='Target domain')
        axs[i,1].set(title=f"Bootstrap\nstate 0\n{bs['present_missing_run_count']}/{bs['run_count']}",xticks=[0],xticklabels=['/ 11 μ'])
    f.suptitle('Present-class pseudo-label collapse',fontsize=11)
    # Counts printed in every cell carry the interpretation independently of color.
    f.supxlabel('Output cells: affected states / 10; bootstrap cells: affected runs / 11',fontsize=8);save(f,'6')
    f,axs=plt.subplots(2,1,figsize=(W,7.5),sharey=True,layout='constrained')
    for ax,metric,title in zip(axs,['f1_range','f1_std'],['Range across 11 μ values','Population SD across 11 μ values']):
        for g,c,col in [(0,'UCI_PRIMARY','#2166ac'),(1,'EXTERNAL_PRIMARY','#b35806')]:
            rr=[r for r in V['figures']['S1']['plotted_records'] if r['cohort']==c]
            for j,r in enumerate(rr):
                x=g+(j-(len(rr)-1)/2)*.09;y=float(r[metric]);ax.scatter(x,y,s=28,color=col);ax.annotate(('B' if g==0 else '')+r['target'],(x,y),xytext=(0,6),textcoords='offset points',fontsize=8,ha='center')
        ax.set(title=title,xticks=[0,1],xticklabels=['UCI primary\n6 target domains','External primary\n7 target domains'],xlim=(-.35,1.4),ylim=(0,1),yticks=np.arange(0,1.01,.2),ylabel='Absolute F1 variation');ax.grid(axis='y',alpha=.2)
    f.suptitle('Supplementary Figure S1 · Descriptive sensitivity',fontsize=11);save(f,'S1')
    (O/'figure_values.json').write_text(json.dumps(V, indent=2)+'\n')
    (O/'captions.json').write_text(json.dumps(C, indent=2)+'\n')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['1','2','3','5','6','S1','all'], default='all')
    parser.add_argument('--output', '--outputdir', type=Path, default=Path('reproduced_figures'))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output must be a new directory')
    render(args.output, None if args.figure == 'all' else {args.figure})
if __name__ == '__main__':
    main()
