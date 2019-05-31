import sys
import warnings
from typing import Any

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import json
from matplotlib import rcParams, pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import interp
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve
import statsmodels.api as sm

sys.path.append('../../config')
from mpl_style import *
colors = rcParams['axes.prop_cycle'].by_key()['color']

rcParams['figure.dpi'] = 96
rcParams['figure.figsize'] = (12,8)

class EvaluationData:
    def __init__(self, plots_dict, scores_df):
        self.plots_dict = plots_dict
        self.scores_df = scores_df

        save_loc = os.path.join(plots_dict['models_dir'], plots_dict['model_id'])
        self.stats_dir = '{}/stats'.format(save_loc)
        if not os.path.exists(self.stats_dir):
            os.mkdir(self.stats_dir)

    def get_ridge_data(self):
        ridge_df = self.scores_df[['label', 'score']].rename(
            columns={'score': 'Score'}
        ).assign(
            **{'label': self.scores_df['label']
                .astype(str)
                .map(self.plots_dict['label_map'])
               }
        )

        if self.plots_dict['save_data'] is True:
            ridge_df.to_csv('{}/ridge_data.csv'.format(self.stats_dir))
        return ridge_df

    def get_threshold_data(self, metric):
        df = self.scores_df
        acc_curve = {}
        for i in np.arange(0, 1.01, 0.01):
            df['pred'] = (df['score'] >= i).astype(int)

            if metric == 'Accuracy':
                acc_curve[i] = accuracy_score(df['label'],
                                              df['pred'])
            elif metric == 'F1':
                if min(df['label'].value_counts().shape[0],
                       df['pred'].value_counts().shape[0]) <= 1:
                    acc_curve[i] = None
                else:
                    acc_curve[i] = f1_score(df['label'], df['pred'])

        acc_df = pd.DataFrame \
            .from_dict(acc_curve, orient='index') \
            .sort_index() \
            .rename(columns={0: metric})
        acc_df.index.name = 'Model Score Threshold'

        if self.plots_dict['save_data'] is True:
            stats_dir = self.stats_dir
            acc_df.to_csv(f'{stats_dir}/metric_by_threshold__{metric}.csv')
        return acc_df

    def get_bins_data(self, bin_type, nbins):
        '''given a pandas DF of scores and labels,
        compute nbins using either percentiles or
        histogram style bins'''
        assert bin_type in ['Bin', 'Percentile']
        # Bin into 'nbins' uniform bins
        df = self.scores_df
        if bin_type == 'Bin':
            df[bin_type] = df['score'].apply(
                lambda x: int(np.round(x * nbins, 0))
            )
        ## Bin by percentile
        elif bin_type == 'Percentile':
            df = df.sort_values(by='score')
            df['rk'] = np.arange(0, df.shape[0], 1)
            df['rk'] /= float(df.shape[0])
            df[bin_type] = df['rk'].apply(
                lambda x: int(np.round(x * nbins, 0))
            )

        scores_to_plot = df[[bin_type, 'label']]
        scores_to_plot['label'] = scores_to_plot['label'].astype(str)
        ## reindex 0,1,2,...,100
        new_idx = np.arange(0, nbins + 1, 1)

        scores_to_plot['count'] = 1
        # scores_to_plot.pivot(index=bin_type, columns='label').fillna(0)
        bins_out = scores_to_plot \
                    .groupby([bin_type, 'label']) \
                    .count() \
                    .reset_index(drop=False) \
                    .pivot(index=bin_type, columns='label', values='count') \
                    .rename(columns=self.plots_dict['label_map']) \
                    .reindex(new_idx) \
                    .fillna(0) \
                    .sort_index()

        if self.plots_dict['save_data'] is True:
            stats_dir = self.stats_dir
            bins_out.to_csv(f'{stats_dir}/distributions__{nbins}_{bin_type}.csv')
        return bins_out


    def get_roc_sets(self):
        if 'fold' in self.scores_df.columns:
            return {
                k: self.scores_df[self.scores_df['fold'] == k][['label', 'score']]
                for k in self.scores_df['fold'].unique()
            }
        else:
            return {'Full': self.scores_df}

    def get_mean_auc(self):
        """called separately from roc_plot_kfold_errband"""
        roc_sets = self.get_roc_sets()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i, (set_nbr, df) in enumerate(roc_sets.items()):
            fpr, tpr, thresholds = roc_curve(df['label'], df['score'])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        # compute mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        if self.plots_dict['save_data'] is True:
            with open('{}/auc.txt'.format(self.stats_dir),'w') as w:
                w.write('\n'.join(
                    ['Fold: {}'.format(a) for a in aucs]
                ))
                w.write('\nMean: {}'.format(mean_auc))
        return mean_auc

    def get_accuracy_at_topn(self, groupby_col, col_range):
        """
        :param groupby_col: will be split by __, so
            a tuple like week_id__season will be handled
            this way
        :param col_range: inputs into np.arange, i.e. 
            low, high, skip
        :return: pd.DataFrame of accuracies over col_range
        """
        groupby_cols = groupby_col.split('__')
        scores_grouped = self.scores_df[['score', 'label'] + groupby_cols] \
                            .sort_values(by='score', ascending=False) \
                            .groupby(groupby_cols)
        max_dim = self.scores_df.groupby(groupby_cols).size().max()
        col_range[1] = max_dim

        acc_by_n_games = pd.DataFrame.from_dict({
            n: {'accuracy': scores_grouped.head(n)['label'].mean()}
            for n in np.arange(*col_range)
            if n <= max_dim
        }, orient='index')

        if self.plots_dict['save_data'] is True:
            acc_by_n_games.to_csv(
                '{}/accuracy_at_topn__{}.csv'
                    .format(self.stats_dir, '__'.join(groupby_cols))
            )
        return acc_by_n_games


class EvaluateAndPlot(EvaluationData):
    def __init__(self, plots_dict, scores_df):
        self.plots_dict = plots_dict
        self.scores_df = scores_df

        save_loc = os.path.join(plots_dict['models_dir'], plots_dict['model_id'])
        self.plots_dir = '{}/plots'.format(save_loc)
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)
        self.stats_dir = '{}/stats'.format(save_loc)
        if not os.path.exists(self.stats_dir):
            os.mkdir(self.stats_dir)

    def plot_all(self):
        self.plot_ridge()
        self.plot_thresholds()
        self.plot_bins()
        self.plot_roc()
        self.plot_accuracy_by_topn()

    def plot_ridge(self):
        self.ridge_viz(self.get_ridge_data())

    def ridge_viz(self, ridge_df):
        """took the following from the seaborn documentation
        https://seaborn.pydata.org/examples/kde_ridgeplot.html
        and applied it to visualizing score distributions
        """
        # backup of matplotlib parameters
        # to reset at the bottom of this function
        import matplotlib as mpl
        params_backup = rcParams.copy()

        title = 'Score Distributions of {} vs. {}' \
            .format(self.plots_dict['label_map']['1'],
                    self.plots_dict['label_map']['0'])
        sns.set(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})

        # Create the data
        g = sns.FacetGrid(
            ridge_df, row='label', hue='label',
            aspect=3, height=2.5, palette=['#35978f', '#9970ab']
        )

        # first the curve, then a white line, then axis
        g.map(sns.kdeplot, 'Score', clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        g.map(sns.kdeplot, 'Score', clip_on=False, color='#f0f0f0', lw=2, bw=.2)

        # add nice distribution label on the left
        def label(x, color, label):
            ax = plt.gca()
            ax.text(-0.2, .2, label, fontweight='bold', color=color,
                    ha='left', va='center', transform=ax.transAxes)

        g.map(label, 'Score')

        g.fig.suptitle(title)
        # add the overlap effect
        g.fig.subplots_adjust(hspace=-.25)
        # get rid of the facet labels
        g.set_titles("")
        g.set(yticks=[])
        g.set(xlim=(0, 1))
        g.despine(bottom=True, left=True)
        if self.plots_dict['save_plots']:
            g.savefig('{}/score_densities.png'.format(self.plots_dir))
        else:
            plt.show()
        plt.clf()

        ## reset matplotlib parameters
        mpl.rcParams.update(params_backup)

    def plot_thresholds(self):
        """Plot metrics by score threshold"""
        for metric in self.plots_dict['threshold_metrics']:
            self.plot_by_threshold(self.get_threshold_data(metric), metric)

    def plot_by_threshold(self, acc_df, metric):
        """given a pandas DF of scores,
        compute and plot the accuracy,
        given score thresholds
        0.00,0.01,0.02,...,1.00"""

        # peak of curve
        peak = acc_df.loc[acc_df.idxmax()]
        peak_score = peak.index[0]
        peak_acc = peak.iloc[0, 0]
        # y-axis value at x=0.5
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
        _ = plt.tight_layout()
        if self.plots_dict['save_plots']:
            plots_dir = self.plots_dir
            plt.savefig(f'{plots_dir}/thresholds_by_{metric}.png')
        else:
            plt.show()
        plt.clf()

    def plot_bins(self):
        for bin_type in self.plots_dict['bin_types']:
            for nbins in self.plots_dict['plot_bins']:
                bins_df = self.get_bins_data(bin_type, nbins)
                # plot bins
                self.bin_viz(bins_df, bin_type, nbins)
                # plot trend
                if bin_type == 'Percentile':
                    self.bin_trend_viz(bins_df, bin_type, nbins)


    def bin_viz(self, df, bin_type, nbins):
        '''given a pandas DF of bins, plot the bins
        overlaid on each other'''
        scores_colors = (colors[0], colors[1], colors[3], colors[2])

        for i, c in enumerate(self.plots_dict['label_map'].values()):
            df[c].plot(kind='bar', rot=0, color=scores_colors[i], width=1,
                       alpha=0.75 - 0.25 * i)

        plt.title('Distribution of Scores vs. Labels: Score {}s'
                  .format(bin_type))
        plt.legend()
        plt.ylabel('Count')
        plt.xlabel('Score {}'.format(bin_type))
        skipticks = int(np.ceil(nbins / 20.))
        a, b = plt.xticks()
        plt.xticks(a[::skipticks], b[::skipticks])
        plt.tight_layout()
        if self.plots_dict['save_plots']:
            plots_dir = self.plots_dir
            plt.savefig(f'{plots_dir}/distributions__{bin_type}_{nbins}bins.png')
        else:
            plt.show()
        plt.clf()

    def bin_trend_viz(self, df, bin_type, nbins):
        """given binned pandas DF, and dict defining
        success, return a bar chart of success by score bin
        and trendlines to indicate whether higher scores correlate
        with greater success"""
        success_name = self.plots_dict['success_name']
        success_col = self.plots_dict['label_map']['1']
        failure_col = self.plots_dict['label_map']['0']

        df['success'] = df[success_col].astype(float) \
                        / (df[success_col] + df[failure_col])

        ax = plt.figure().add_subplot(111)
        ## plot bar chart of win rate by bin
        df['success'].plot(
            kind='bar', color=colors[2], rot=0, alpha=0.75, ax=ax,
            title='{} By {}'.format(success_name, bin_type),
            label='Actual {}'.format(success_name)
        )

        ## ols method doesn't take columns with sapces
        bin_type_nospace = bin_type.replace(' ', '_')
        scores_trend = df.reset_index(drop=False) \
                           .loc[:, [bin_type, 'success']] \
            .rename(columns={bin_type: bin_type_nospace})

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
            frac=1 / np.sqrt(nbins)
        )
        lowess_wide = lowess(
            scores_trend['success'],
            scores_trend[bin_type_nospace],
            frac=2. / 3
        )

        ## assign method takes **kwargs in the form of
        ## new column name: column definition
        scores_trend = scores_trend.assign(
            **{'LOWESS Wide Window': lowess_wide[:, 1],
               'LOWESS Tight Window': lowess_tight[:, 1]}
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
        plt.tight_layout()

        if self.plots_dict['save_plots'] is True:
            plots_dir = self.plots_dir
            plt.savefig(f'{plots_dir}/score_trend__{bin_type}_{nbins}bins.png')
        else:
            plt.show()
        plt.clf()

    def plot_roc(self):
        self.plot_roc_kfold_errband(self.get_roc_sets())

    def plot_roc_kfold_errband(self, roc_sets):
        """plot ROC curves, with curves and AUCs
        for each of the K folds, the mean curve/AUC,
        and error band"""
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i, (set_nbr, df) in enumerate(roc_sets.items()):
            # plot a single ROC curve
            # and append to tprs and aucs
            try:
                # fold scores
                int(set_nbr)
                set_nbr_str = 'Fold {}'.format(i)
                make_plot = True
            except:
                # non-fold scores
                set_nbr_str = ''
                make_plot = False

            self.plot_single_roc_curve(df, tprs, aucs, mean_fpr,
                                        i, set_nbr_str)

        # random (as good as random guessing)
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
                         lw=2, alpha=.8,
                         label=r'Overall ROC (AUC = {:.2f} $\pm$ {:.2f})'
                         .format(mean_auc, std_auc))

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
        plt.tight_layout()
        if self.plots_dict['save_plots']:
            plt.savefig('{}/roc.png'.format(self.plots_dir))
        else:
            plt.show()
        plt.clf()

    def plot_single_roc_curve(self, df, tprs, aucs, mean_fpr, i, set_name):
        '''given pandas DF of label and score,
        compute ROC values and add to current plot.
        append appropriate values to trps and aucs'''
        fpr, tpr, thresholds = roc_curve(df['label'], df['score'])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        _ = plt.plot(fpr, tpr, lw=0.5, alpha=0.5, color=colors[i + 1],
                     label='ROC {} (AUC = {:.2f})'.format(set_name, roc_auc))

    def plot_accuracy_by_topn(self):
        for col_range in self.plots_dict['accuracy_at_topn'].items():
            self.accuracy_by_topn_viz(
                self.get_accuracy_at_topn(*col_range),
                col_range[0]
            )

    def accuracy_by_topn_viz(self, df, col_range):
        if df.shape[0] > 25:
            plot_type = 'line'
        else:
            plot_type = 'bar'
        col_title = col_range.replace('__', ', ').upper()
        df.plot(
            kind=plot_type, rot=0, legend=None,
            title='Accuracy if Top N Games Are Predicted Each: {}'
                .format(col_title)
        )
        plt.xlabel('# of Games Predicted')
        plt.ylabel('Accuracy')

        if self.plots_dict['save_plots'] is True:
            plt.savefig(
                '{}/accuracy_at_topn__{}.png'
                    .format(self.stats_dir, '__'.join('__'.join(col_range)))
            )
        else:
            plt.show()
        plt.clf()