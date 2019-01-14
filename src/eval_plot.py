## suppress warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import argparse
import json
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd

## custom rcParams settings for matplotlib
sys.path.append('config')
import mpl_style
colors = rcParams['axes.prop_cycle'].by_key()['color']

## mode ID is only command-line argument
## store as variable model_id and assert
## that path models/model_id exists,
## then change to that directory
parser = argparse.ArgumentParser()
parser.add_argument('model_id', help='model ID/directory name')
args = parser.parse_args()
model_id = args.model_id 

assert os.path.exists('./models/{}'.format(model_id))
os.chdir('models/{}'.format(model_id))

## load model and plots configuration files
model_dict = json.load(open('model.json','r'))
plots_dict = json.load(open('plots.json','r'))

def compute_bins(plots_dict, df, nbins, bin_type):
    '''given a pandas DF of scores and labels,
    compute nbins using either percentiles or 
    histogram style bins'''
    assert bin_type in ['Bin','Percentile'] 
    label_map = plots_dict['label_map']
    ## Bin into nbins uniform bins
    if bin_type == 'Bin':
        df[bin_type] = df['score'].apply(
            lambda x: int(np.round(x*nbins, 0))
        )
    ## Bin by percentile
    elif bin_type == 'Percentile':
        df = df.sort_values(by='score')
        df['rk'] = np.arange(0, df.shape[0], 1)
        df['rk'] /= float(df.shape[0])
        df[bin_type] = df['rk'].apply(
            lambda x: int(np.round(x*nbins, 0))
        )

    scores_to_plot = df[[bin_type,'label']]
    scores_to_plot['label'] = scores_to_plot['label'].astype(str)
    ## reindex 0,1,2,...,100
    new_idx = np.arange(0, nbins+1, 1)

    scores_to_plot['count'] = 1
    # scores_to_plot.pivot(index=bin_type, columns='label').fillna(0)
    return scores_to_plot\
                .groupby([bin_type,'label'])\
                .count()\
                .reset_index(drop=False)\
                .pivot(index=bin_type, columns='label', values='count')\
                .rename(columns=label_map)\
                .reindex(new_idx)\
                .fillna(0)\
                .sort_index()

def compute_plot_bins(plots_dict, df, nbins, bin_type, colors, plots_dir):
    '''given a pandas DF of bins, plot the bins
    overlaid on each other'''
    
    label_map = plots_dict['label_map']
    scores_colors = (colors[0], colors[3], colors[1], colors[2])

    for i, c in enumerate(df.columns.tolist()[::-1]):
        df[c].plot(kind='bar', rot=0, color=scores_colors[i], width=1, 
                   alpha = 0.75 - 0.25*i)
        
    plt.title('Distribution of Scores vs. Labels: Score {}s'
                .format(bin_type))
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel('Score {}s'.format(bin_type))
    skipticks = int(np.ceil(nbins / 20.))
    a, b = plt.xticks()
    plt.xticks(a[::skipticks], b[::skipticks])
    plt.savefig('{}/distributions__{}_{}bins.png'
                    .format(plots_dir, bin_type, nbins))
    plt.cla()
    
def plot_trend(plots_dict, df, bin_type, nbins, plots_dir):
    import statsmodels.api as sm
    '''given binned pandas DF, and dict defining
    success, return a bar chart of success by score bin
    and trendlines to indicate whether higher scores correlate
    with greater success'''
    
    success_name = plots_dict['success_name']
    success_col = plots_dict['label_map']['1']
    failure_col = plots_dict['label_map']['0']
    
    df['success'] = df[success_col].astype(float) \
                    / ( df[success_col] + df[failure_col] )

    ax = plt.figure().add_subplot(111)
    ## plot bar chart of win rate by bin
    df['success'].plot(
        kind='bar', color=colors[2], rot=0, alpha = 0.75, ax=ax,
        title='{} By {}'.format(success_name, bin_type), 
        label='Actual {}'.format(success_name)
    )
    
    ## ols method doesn't take columns with sapces
    bin_type_nospace = bin_type.replace(' ','_')
    scores_trend = df.reset_index(drop=False)\
                     .loc[:, [bin_type,'success']]\
                     .rename(columns={ bin_type:bin_type_nospace })
    
    ## Fit OLS model    
    linreg = sm.formula.ols(
        formula='success ~ {}'.format(bin_type_nospace), 
        data=scores_trend
    )
    
    res = linreg.fit()
    scores_trend = scores_trend.assign(OLS=res.fittedvalues)

   ## LOWESS Trends (wider window and tigher window)
    lowess = sm.nonparametric.lowess

    lowess_tight = lowess(
            scores_trend['success'], 
            scores_trend[bin_type_nospace], 
            frac=1/np.sqrt(nbins)
        )
    lowess_wide = lowess(
            scores_trend['success'], 
            scores_trend[bin_type_nospace], 
            frac=2./3
        )

    ## assign method takes **kwargs in the form of 
    ## new column name: column definition
    scores_trend = scores_trend.assign(
        **{'LOWESS Wide Window': lowess_wide[:,1],
           'LOWESS Tight Window': lowess_tight[:,1]}
    )

    ## plot trendlines
    scores_trend['OLS'].plot(kind='line', ax=ax, color=colors[1], 
                             label='OLS (slope={:.4f})'
                                      .format(res.params[bin_type]))
    scores_trend['LOWESS Wide Window'].plot(
                kind='line', ax=ax, color=colors[0]
            )
    scores_trend['LOWESS Tight Window'].plot(
                kind='line', ax=ax, color=colors[4]
            )
    
    skipticks = int(np.ceil(nbins / 20.))
    plt.legend()
    plt.ylabel(success_name)
    plt.xlabel('Score {}s'.format(bin_type))    
    a, b = plt.xticks()
    plt.xticks(a[::skipticks], b[::skipticks])
    plt.savefig('{}/score_trend__{}_{}bins.png'
                  .format(plots_dir, bin_type, nbins))
    plt.cla()

def plot_single_roc_curve(df, tprs, aucs, mean_fpr, i, set_name, make_plot):
    '''given pandas DF of label and score,
    compute ROC values and add to current plot.
    append appropriate values to trps and aucs'''
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    
    fpr, tpr, thresholds = roc_curve(df['label'], df['score'])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    if make_plot is True:
        _ = plt.plot(fpr, tpr, lw=0.5, alpha=0.5, color=colors[i+1],
                     label='ROC {} (AUC = {:.2f})'
                              .format(set_name, roc_auc))

def roc_plot_kfold_errband(roc_sets, plots_dir):
    '''plot ROC curves, with curves and AUCs 
    for each of the K folds, the mean curve/AUC, 
    and error band'''
    from sklearn.metrics import auc
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (set_nbr, df) in enumerate(roc_sets.iteritems()):
        ## plot a single ROC curve
        ## and append to tprs and aucs
        try:
            ## fold scores
            int(set_nbr)
            set_nbr_str = 'Fold {}'.format(i)
            make_plot = True
        except:
            ## non-fold scores
            set_nbr_str = ''
            make_plot = False

        plot_single_roc_curve(df, tprs, aucs, mean_fpr, 
                              i, set_nbr_str, make_plot)

    ## random (as good as random guessing)
    _ = plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color=colors[-1],
             label='Always Predict Majority Class', alpha=0.8)

    ## compute mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    if len(roc_sets.keys()) > 1:
        _ = plt.plot(mean_fpr, mean_tpr, color=colors[0],
                 label=r'Mean ROC (AUC = {:.2f} $\pm$ {:.2f})'
                          .format(mean_auc, std_auc),
                 lw=2, alpha=.8)
    else:
        _ = plt.plot(mean_fpr, mean_tpr, color=colors[0],
                 label=r'Overall ROC', lw=2, alpha=.8)

    ## add error bands, +/- 1 stddev
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    _ = plt.fill_between(
        mean_fpr, tprs_lower, tprs_upper, color=colors[-1], 
        alpha=0.2, label=r'$\pm$ 1 std. dev.'
    )

    ## modify plot attributes
    _ = plt.xlim([-0.05, 1.05])
    _ = plt.ylim([-0.05, 1.05])
    _ = plt.xlabel('False Positive Rate')
    _ = plt.ylabel('True Positive Rate')
    _ = plt.title('ROC Curve')
    _ = plt.legend(loc="lower right")
    plt.savefig('{}/roc.png'.format(plots_dir))
    plt.cla()

def plot_by_threshold(df, metric, plots_dir):
    '''given a pandas DF of scores, 
    compute and plot the accuracy, 
    given score thresholds
    0.00,0.01,0.02,...,1.00'''
    from sklearn.metrics import f1_score, accuracy_score
    
    acc_curve = {}
    for i in np.arange(0, 1.01, 0.01):
        df['pred'] = (df['score'] >= i).astype(int)
        
        if metric == 'Accuracy':
            acc_curve[i] = accuracy_score(df['label'], df['pred'])
        elif metric == 'F1':
            if min(df['label'].value_counts().shape[0], 
                   df['pred'].value_counts().shape[0]) <= 1:
                acc_curve[i] = 0
            else:
                acc_curve[i] = f1_score(df['label'], df['pred'])

    acc_df = pd.DataFrame\
               .from_dict(acc_curve, orient='index')\
               .sort_index()\
               .rename(columns={0:metric})
    acc_df.index.name = 'Model Score Threshold'

    ## peak of curve
    peak = acc_df.loc[acc_df.idxmax()]
    peak_score = peak.index[0]
    peak_acc = peak.iloc[0,0]
    ## y-axis value at x=0.5
    x0_50_acc = acc_df.loc[0.50, metric]

    _ = acc_df.plot(
        kind='line', 
        title='{} at Different Classification Thresholds'.format(metric)
    )
    _ = plt.axhline(
        y=peak_acc, xmin=0, xmax=1, 
        label='Score {:.3f} | {} {:.3f} (Peak)'
                  .format(peak_score, metric, peak_acc),
        linewidth=2, color=colors[2]
    )
    _ = plt.axhline(
        y=acc_df.loc[0.50, metric], xmin=0, xmax=1, 
        label='Score 0.50 | {} {:.3f}'.format(metric, x0_50_acc),
        linewidth=2, color=colors[3]
    )
    _ = plt.legend()
    _ = plt.ylabel(metric)
    plt.savefig('{}/{}_by_thresholds'.format(plots_dir, metric))
    return acc_df

## START EXECUTION
if not os.path.exists('plots'): 
    os.mkdir('plots')
if not os.path.exists('stats'): 
    os.mkdir('stats')

    
for scores_csv in os.listdir('scores'):
    dirname = scores_csv.replace('_scores.csv','')
    stats_dir = 'stats/{}'.format(dirname)
    plots_dir = 'plots/{}'.format(dirname)
    if not os.path.exists(stats_dir): 
        os.mkdir(stats_dir)
    if not os.path.exists(plots_dir): 
        os.mkdir(plots_dir)
    
    scores_df = pd.read_csv(
        'scores/{}'.format(scores_csv), 
        index_col=model_dict['index']
    )

    ## Plot metrics by score threshold
    for metric in plots_dict['threshold_metrics']:
        metric_by_threshold = plot_by_threshold(scores_df, metric, plots_dir)
        metric_by_threshold.to_csv('{}/metric_by_threshold__{}.csv'
                                      .format(stats_dir, metric))

    ## Plot distributions
    for bin_type in plots_dict['bin_types']:
        for nbins in plots_dict['plot_bins']:
            ## plot bins
            curr_bins = compute_bins(plots_dict, scores_df, nbins, bin_type)
            compute_plot_bins(plots_dict, curr_bins, nbins, 
                              bin_type, colors, plots_dir)
            ## plot trend
            if bin_type == 'Percentile':
                plot_trend(plots_dict, curr_bins, bin_type, nbins, plots_dir)
            ## store data
            curr_bins.to_csv('{}/distributions__{}_{}.csv'
                                .format(stats_dir, nbins, bin_type))

    #### ROC
    if 'fold' in scores_df.columns:
        roc_sets = {
            k: scores_df[scores_df['fold'] == k][['label','score']]
            for k in scores_df['fold'].unique()
        }
    else:
        roc_sets = {'Full': scores_df}
    roc_plot_kfold_errband(roc_sets, plots_dir)

print 'successfully completed evaluation and plotting.'