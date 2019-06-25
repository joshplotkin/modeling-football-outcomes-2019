# TODO: colorbar legends on residuals plots
# TODO: residual by spread
# TODO: fix bars getting cutoff
# TODO: differences between jupyter display and savefig
# TODO: (above) https://stackoverflow.com/questions/7906365/matplotlib-savefig-plots-different-from-show

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from matplotlib import rcParams, pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import interp
import shap
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve, confusion_matrix
import statsmodels.api as sm
import xgboost

sys.path.append('../../config')
from mpl_style import *
colors = rcParams['axes.prop_cycle'].by_key()['color']

rcParams['figure.dpi'] = 96
rcParams['figure.figsize'] = (12,8)

class EvaluationData:
    def __init__(self, plots_dict, scores_df):
        self.plots_dict = plots_dict
        self.scores_df = scores_df

        if scores_df['label'].unique().shape[0] > 2:
            self.is_regression = True
        else:
            self.is_regression = False

        save_loc = os.path.join(plots_dict['models_dir'], plots_dict['model_id'])
        self.stats_dir = '{}/stats'.format(save_loc)
        if not os.path.exists(self.stats_dir):
            os.mkdir(self.stats_dir)

    def add_regression_to_classification_data(self):
        eval_dict = self.plots_dict['regression_evaluation']
        raw_comparison = eval_dict['comparison']
        label_col = eval_dict['label']
        round_score = eval_dict['round_score']

        if type(raw_comparison) is int:
            comparison = raw_comparison
        else:
            assert raw_comparison in self.scores_df.columns
            comparison = self.scores_df[raw_comparison]

        self.scores_df['label'] = self.scores_df[label_col].apply(lambda x: max(x, 0))

        if round_score:
            self.scores_df['comparison_score'] = self.scores_df['regression_score'].round() - comparison
        else:
            self.scores_df['comparison_score'] = self.scores_df['regression_score'] - comparison

        self.scores_df['score'] = self.scores_df \
                                     .sort_values(by='comparison_score', ascending=False) \
                                     .loc[:, 'comparison_score'] \
                                     .rank(pct=True)
        self.scores_df['binary_pred'] = (self.scores_df['comparison_score'] > 0).astype(int)
        self.scores_df.drop('comparison_score', axis=1, inplace=True)

    def get_ridge_data(self):
        ridge_df = self.scores_df[['label', 'score']].rename(
            columns={'score': 'Score'}
        ).assign(
            **{'label': self.scores_df['label']
                .astype(str)
                .map(self.plots_dict['label_map'])
               }
        )

        if self.plots_dict['save']['data'] is True:
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

        if self.plots_dict['save']['data'] is True:
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

        if self.plots_dict['save']['data'] is True:
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

        if self.plots_dict['save']['data'] is True:
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

        if self.plots_dict['save']['data'] is True:
            acc_by_n_games.to_csv(
                '{}/accuracy_at_topn__{}.csv'
                    .format(self.stats_dir, '__'.join(groupby_cols))
            )
        return acc_by_n_games

    def get_distributions_data(self):
        act = self.scores_df['regression_label'] \
            .round() \
            .astype(int) \
            .value_counts()
        act /= act.sum()
        act.name = 'Actual'

        pred = self.scores_df['regression_score'] \
            .round() \
            .astype(int) \
            .value_counts()
        pred /= pred.sum()
        pred.name = 'Predicted'

        dist = act.to_frame().merge(
                pred.to_frame(), left_index=True,
                right_index=True, how='outer'
            ).sort_index().fillna(0)

        if self.plots_dict['save']['data'] is True:
            stats_dir = self.stats_dir
            dist.to_csv(f'{stats_dir}/regression_distributions.csv')

        return dist

    def get_scatter_data(self):
        scatter = self.scores_df[['regression_label','regression_score']]\
                    .sort_values(by=['regression_label','regression_score'])\
                    .reset_index(drop=True)

        val_range = scatter['regression_label'].max() - scatter['regression_label'].min()
        lowess = sm.nonparametric.lowess
        lowess_tight = lowess(
            scatter['regression_label'],
            scatter['regression_score'],
            frac=1 / np.sqrt(val_range)
        )
        lowess_wide = lowess(
            scatter['regression_label'],
            scatter['regression_score'],
            frac=2. / 3
        )
        ## Fit OLS model
        linreg = sm.formula.ols(
            formula='regression_score ~ regression_label',
            data=scatter
        )
        res = linreg.fit()

        scatter = scatter.assign(
            **{'LOWESS Wide Window': lowess_wide[:, 1],
               'LOWESS Tight Window': lowess_tight[:, 1],
               'OLS': res.fittedvalues}
        )

        if self.plots_dict['save']['data'] is True:
            stats_dir = self.stats_dir
            scatter.to_csv(f'{stats_dir}/scatter_regression.csv')

        return scatter

    def get_residuals_season_week_mean(self):
        residuals = self.scores_df \
            .assign(sq_error=
                    (self.scores_df['regression_score']
                     - self.scores_df['regression_label']) ** 2
                    )[['season', 'week_id', 'sq_error']]
        residuals.columns = ['Season', 'Week #', 'RMSE']
        mean_residuals = residuals.groupby(['Season', 'Week #']) \
            .agg(lambda x: np.sqrt(x.mean())) \
            .reset_index()

        pivot_residuals = mean_residuals.pivot(index='Season', columns='Week #', values='RMSE')
        pivot_residuals.sort_index(ascending=False, inplace=True)

        if self.plots_dict['save']['data'] is True:
            stats_dir = self.stats_dir
            mean_residuals.to_csv(f'{stats_dir}/residuals_mean_by_season_week.csv')

        return residuals, mean_residuals, pivot_residuals

    def get_confusion_matrix_data(self, label_col, binary_pred_col):

        confus_matrix = confusion_matrix(self.scores_df[label_col],
                                         self.scores_df[binary_pred_col])
        class_names = [self.plots_dict['label_map']['0'], self.plots_dict['label_map']['1']]

        confus_df = pd.DataFrame(confus_matrix)
        confus_df.index = list(map(lambda x: f'Actual: {x}', class_names))
        confus_df.columns = list(map(lambda x: f'Predicted: {x}', class_names))

        if self.plots_dict['save']['data'] is True:
            stats_dir = self.stats_dir
            confus_df.to_csv(f'{stats_dir}/confusion_matrix_regression.csv')

        return confus_df

    def get_feature_importances(self, model):
        """xgboost has score, fscore, and _feature importances.
        deprioritizing sklearn for now. will need to be different
        for each model.
        """
        import inspect

        if 'xgboost' in str(inspect.getmodule(model)):
            # Other options:
            # importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
            # mdl._Booster.get_fscore()
            # mdl.feature_importances_
            importances = model._Booster.get_score(importance_type='weight')
            importances_df = pd.DataFrame.from_dict(importances, orient='index')
            importances_df.index.name = 'feature'
            importances_df.columns = ['importance']
            importances_df.sort_values(by='importance', ascending=False, inplace=True)

        else:
            print('Feature importance not supported for this model type')
            return

        if self.plots_dict['save']['data'] is True:
            stats_dir = self.stats_dir
            importances_df.to_csv(f'{stats_dir}/feature_importances.csv')

    def get_shap_vals(self, model_dict, model_objects):
        for fold in np.arange(model_dict['kfolds']):
            explainer = shap.TreeExplainer(model_objects[fold])
            features = self.scores_df[self.scores_df['fold'] == fold] \
                           .loc[:, model_dict['features_list']]
            shap_values = explainer.shap_values(features)
            shap_fold_df = pd.DataFrame.from_dict(
                {features.index[i]: dict(zip(model_dict['features_list'], val))
                 for i, val in enumerate(shap_values)},
                orient='index'
            )
            shap_fold_df['bias'] = explainer.expected_value

            if fold == 0:
                shap_df = shap_fold_df
            else:
                shap_df = shap_df.append(shap_fold_df)

        # check that the sum of shap values + bias equals model prediction
        tmp = features[[]].merge(shap_df, left_index=True, right_index=True).sum(axis=1).to_frame()
        tmp[1] = model_objects[fold].predict(features)
        assert ((tmp[0] - tmp[1]).abs() > 1e-4).sum() == 0

        before = shap_df.shape[0]
        shap_df = self.scores_df[[]].merge(shap_df, left_index=True, right_index=True)
        assert shap_df.shape[0] == before

        if self.plots_dict['save']['data'] is True:
            stats_dir = self.stats_dir
            shap_df.to_csv(f'{stats_dir}/shap_values.csv')

        return shap_df


class EvaluateAndPlot(EvaluationData):
    def __init__(self, plots_dict, scores_df, is_classification):
        self.plots_dict = plots_dict
        self.scores_df = scores_df
        self.is_classification = is_classification

        if self.is_classification is False:
            self.add_regression_to_classification_data()

        save_loc = os.path.join(plots_dict['models_dir'], plots_dict['model_id'])
        self.plots_dir = '{}/plots'.format(save_loc)
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)
        self.stats_dir = '{}/stats'.format(save_loc)
        if not os.path.exists(self.stats_dir):
            os.mkdir(self.stats_dir)

    def plot_all(self, model_dict=None, features_df=None, model_objects=None):
        self.plot_ridge()
        self.plot_thresholds()
        self.plot_bins()
        self.plot_roc()
        self.plot_accuracy_by_topn()

        if self.is_classification is False:
            self.plot_distributions()
            self.plot_scatter()
            self.plot_residuals_by_season_week_all()
            self.plot_confusion_matrix()

        # all 3 must be defined to run shap plots
        if type(None) not in [type(model_dict), type(model_objects), type(features_df)]:
            shap_df = self.get_shap_vals(model_dict, model_objects)
            self.plot_shap_feature_importance(model_dict, features_df, model_objects, shap_df)
            self.plot_shap_dependence(model_dict, features_df, model_objects, shap_df)

        if type(model_objects['full']) is not type(None):
            self.plot_feature_importances(model_objects['full'])

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

        g.fig.suptitle(title, fontsize=20)
        # add the overlap effect
        g.fig.subplots_adjust(hspace=-.25)
        # get rid of the facet labels
        g.set_titles("")
        g.set(yticks=[])
        g.set(xlim=(0, 1))
        g.despine(bottom=True, left=True)
        if self.plots_dict['save']['plots']:
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
        if self.plots_dict['save']['plots']:
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
        scores_colors = ('#1a9850', '#d73027', colors[3], colors[2])

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
        if self.plots_dict['save']['plots']:
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

        if self.plots_dict['save']['plots'] is True:
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
        if self.plots_dict['save']['plots']:
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
        _ = plt.plot(fpr, tpr, lw=0.5, alpha=0.5, color=colors[(i + 1) % len(colors)],
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
            title='Accuracy if Top N Games By Model Score Are Predicted Each: {}'
                .format(col_title)
        )
        plt.xlabel('# of Games Predicted')
        plt.ylabel('Accuracy')

        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/accuracy_at_topn__{}.png'
                    .format(self.plots_dir, col_range)
            )
        else:
            plt.show()
        plt.clf()

    def plot_distributions(self):
        self.distributions_viz(self.get_distributions_data())

    def distributions_viz(self, vals):
        ax = plt.figure().add_subplot(111)

        _ = (vals['Actual']).sort_index().plot(
            kind='bar', rot=0, width=1., ax=ax, alpha=0.75, legend=True,
        )
        _ = (vals['Predicted']).sort_index().plot(
            kind='bar', rot=0, width=1., ax=ax, alpha=0.75, legend=True, color=colors[2],
        )

        _ = plt.title('Comparison of Distributions: Actual vs. Predicted', fontsize = 20)
        a, b = plt.xticks()
        _ = plt.xticks(a[1::3], b[1::3])
        _ = plt.ylabel('Proportion of Games')
        _ = plt.xlabel('Final Score Margin (Home - Visitor)')

        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/distributions.png'
                    .format(self.plots_dir)
            )
        else:
            plt.show()
        plt.clf()

    def plot_scatter(self):
        self.scatter_viz(self.get_scatter_data())

    def scatter_viz(self, scatter):
        ax = plt.figure().add_subplot(111)

        _ = scatter[['regression_score', 'regression_label']].plot(
            kind='scatter', x='regression_label', y='regression_score',
            alpha=0.5, ax=ax
        )
        _ = scatter.set_index('regression_label')[['LOWESS Wide Window']].plot(
            kind='line', ax=ax, color=colors[1]
        )
        _ = scatter.set_index('regression_label')[['LOWESS Tight Window']].plot(
            kind='line', ax=ax, color=colors[2]
        )
        _ = scatter.set_index('regression_label')[['OLS']].plot(
            kind='line', ax=ax, color=colors[3]
        )

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, '#262626', alpha=0.25, zorder=0, label='x=y')
        _ = plt.legend()
        _ = plt.title('Scatter Plot of Actual vs. Predicted, with Trendlines', fontsize=20)
        _ = plt.xlabel('Actual')
        _ = plt.ylabel('Predicted')

        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/scatter_actual_vs_predicted.png'
                    .format(self.plots_dir)
            )
        else:
            plt.show()
        plt.clf()

    def plot_residuals_by_season_week_all(self):
        residuals, mean_residuals, pivot_residuals = self.get_residuals_season_week_mean()
        self.residuals_by_season_week_bars_viz(mean_residuals, plot_cols=2)
        self.residuals_by_season_week_heatmap_viz(pivot_residuals)
        self.residuals_by_season_week_bars_agg_viz(mean_residuals, residuals)

    def residuals_by_season_week_bars_viz(self, mean_residuals, plot_cols=2):
        fig = plt.figure(figsize=(20, 16))

        all_seasons = sorted(mean_residuals['Season'].unique())
        all_weeks = sorted(mean_residuals['Week #'].unique())
        plot_rows = np.ceil(len(all_seasons) / plot_cols) + 1

        cm = mpl.cm.PiYG_r
        cnorm = mpl.colors.Normalize(vmin=mean_residuals['Week #'].min() * 0.8,
                                     vmax=mean_residuals['Week #'].max() * 1.2)
        smap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cm)

        for s, season in enumerate(all_seasons):
            ax = fig.add_subplot(plot_rows, plot_cols, s + 1)
            data = mean_residuals[mean_residuals['Season'] == season] \
                .set_index('Week #') \
                [['RMSE']] \
                .reindex(fill_value=0)
            _ = ax.bar(
                x=data.index, height=data['RMSE'],
                color=data['RMSE'].apply(lambda x: smap.to_rgba(x))
            )
            _ = ax.set_title(f'Season: {season}')
            _ = ax.set_xticks(data.index)
            _ = ax.set_xticklabels(all_weeks, rotation=0, minor=all_weeks)

            if s == 0:
                _ = plt.ylabel('Residual')
                _ = plt.xlabel('Week #')
            else:
                _ = plt.ylabel('')
                _ = plt.xlabel('')

            _ = plt.ylim(0, mean_residuals['Week #'].max())

        _ = plt.tight_layout()
        _ = plt.subplots_adjust(top=0.88)

        _ = plt.suptitle('Residuals by Season-Week', fontsize=30,
                         color='#262626', y=0.95)
        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/resiuals_by_season_week_bars.png'
                    .format(self.plots_dir)
            )
        else:
            plt.show()
        plt.clf()

    def residuals_by_season_week_bars_agg_viz(self, mean_residuals, residuals):
        def rmse(x):
            return np.sqrt(x.mean())

        fig = plt.figure()
        all_weeks = sorted(mean_residuals['Week #'].unique())

        maxy = max(residuals.groupby(['Season'])['RMSE'].agg(rmse).max(),
                   residuals.groupby(['Week #'])['RMSE'].agg(rmse).max())

        miny = min(residuals.groupby(['Season'])['RMSE'].agg(rmse).min(),
                   residuals.groupby(['Week #'])['RMSE'].agg(rmse).min())

        cm = mpl.cm.PiYG_r
        cnorm = mpl.colors.Normalize(vmin=miny, vmax=maxy)
        smap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cm)

        all_vals = {'Season': sorted(mean_residuals['Season'].unique()),
                    'Week #': sorted(mean_residuals['Week #'].unique())}

        for i, dim in enumerate(['Season', 'Week #']):
            dim_residuals = residuals.groupby([dim])['RMSE'] \
                .agg(lambda x: np.sqrt(x.mean())) \
                .to_frame()

            ax = fig.add_subplot(2, 1, i + 1)
            _ = ax.bar(
                x=dim_residuals.index, height=dim_residuals['RMSE'],
                color=dim_residuals['RMSE'].apply(lambda x: smap.to_rgba(x))
            )

            _ = plt.ylim([0, maxy])
            _ = plt.ylabel('RMSE')
            _ = plt.xlabel(dim)
            _ = ax.set_xticks(dim_residuals.index)
            _ = ax.set_xticklabels(all_vals[dim], rotation=0, minor=all_weeks)

        _ = plt.tight_layout()
        _ = plt.subplots_adjust(top=0.9)
        _ = plt.suptitle('Residuals by Season and by Week', fontsize=20,
                         color='#262626', y=0.95)

        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/resiuals_by_season_week_bars_agg.png'
                    .format(self.plots_dir)
            )
        else:
            plt.show()
        plt.clf()

    def residuals_by_season_week_heatmap_viz(self, pivot_residuals):
        fig = plt.figure(figsize=(8,6))
        ax = sns.heatmap(pivot_residuals, annot=True, cmap=plt.cm.PiYG_r)
        _ = ax.set_yticklabels(ax.get_yticklabels(), weight='bold', fontsize=10)
        _ = ax.set_xticklabels(ax.get_xticklabels(), weight='bold', fontsize=10)
        _ = plt.title('RMSE by Season and Week', fontsize=12)

        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/resiuals_by_season_week_heatmap.png'
                    .format(self.plots_dir)
            )
        else:
            plt.show()
        plt.clf()

    def plot_confusion_matrix(self):
        self.confusion_matrix_viz(
            self.get_confusion_matrix_data('label', 'binary_pred')
        )

    def confusion_matrix_viz(self, confusion_df):
        def print_confusion_matrix(confusion_matrix, class_names):
            """Taken/modified from here:
            https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823"""
            df_cm = pd.DataFrame(
                confusion_matrix, index=class_names, columns=class_names,
            )
            tot = df_cm.sum().sum()
            annot = df_cm.astype(str).values

            for i in np.arange(annot.shape[0]):
                for j in np.arange(annot[i].shape[0]):
                    v = int(annot[i][j])
                    annot[i][j] = '{}\n({:.1f}%)'.format(v, v / tot * 100)

            label_col = self.plots_dict['regression_evaluation']['label'].upper()

            heatmap = sns.heatmap(df_cm, annot=annot, fmt='', cmap=plt.cm.BuGn)
            heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
            heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
            plt.title(f'Confusion Matrix: {label_col}', fontsize=20)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        class_names = [self.plots_dict['label_map']['0'],
                       self.plots_dict['label_map']['1']]
        print_confusion_matrix(confusion_df.values, class_names)

        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/confusion_matrix_{}.png'
                    .format(self.plots_dir,
                            self.plots_dict['regression_evaluation']['label'].lower())
            )
        else:
            plt.show()
        plt.clf()

    def plot_feature_importances(self, model):
        """TODO: round feature importances on bars"""
        import inspect
        if 'xgboost' in str(inspect.getmodule(model)):
            if self.plots_dict['save']['data'] is True:
                self.get_feature_importances(model)

            plt.figure(figsize=(24, 24))
            # Available importance_types = ['weight', 'gain',
            # 'cover', 'total_gain', 'total_cove']
            importance_type = 'gain'
            ax = xgboost.plot_importance(
                model, ax=None, height=0.2, xlim=None, ylim=None,
                title=f'Feature Importance Using {importance_type}',
                xlabel='XGBoost Feature Importance',
                ylabel='Features', importance_type=importance_type,
                max_num_features=20, grid=True, show_values=True,
            )
            _ = plt.subplots_adjust(left=0.4)
            _ = plt.subplots_adjust(right=1.0)

            if self.plots_dict['save']['plots'] is True:
                plt.savefig(
                    '{}/feature_importance.png'.format(self.plots_dir)
                )
            else:
                plt.show()
            plt.clf()
        else:
            print('Feature importance not supported for this model type')

    def plot_shap_force_plot(self, games, model, shap_df=None):
        model_dict = model.model_dict
        features_df = model.cv_scores

        if type(shap_df) is type(None):
            shap_df = self.get_shap_vals(model_dict, model.model_objects)
        # check if games is a single element
        if not np.array(games).shape:
            games = np.array(games).tolist()
            bias = shap_df.loc[games, 'bias']
            multi = False
        else:
            bias = shap_df.loc[games, 'bias'].mean()
            multi = True

        force_plot = shap.force_plot(
            bias,
            shap_df.loc[games, model_dict['features_list']].values,
            features_df.loc[games, model_dict['features_list']]
        )

        if self.plots_dict['save']['plots'] is True:
            if multi is True:
                img_path =  '{}/force_plot_multi.html'\
                                .format(self.plots_dir)
            else:
                img_path = '{}/force_plot_{}.html' \
                                .format(self.plots_dir, games)

            shap.save_html(img_path, force_plot)
        else:
            return force_plot

    def plot_shap_dependence(self, model_dict, features_df, model_objects, shap_df=None):
        if type(shap_df) is type(None):
            shap_df = self.get_shap_vals(model_dict, model_objects)

        for i, name in enumerate(model_dict['features_list']):
            self.shap_dependence_viz(shap_df, features_df, model_dict, i, name)

    def shap_dependence_viz(self, shap_df, features_df, model_dict, i, name):
        shap.dependence_plot(
            i,
            shap_df.loc[:, model_dict['features_list']].values,
            features_df.loc[:, model_dict['features_list']],
            show=(not self.plots_dict['save']['plots'])
        )
        if self.plots_dict['save']['plots'] is True:
            dependence_path = '{}/dependence_plots'.format(self.plots_dir)
            if not os.path.exists(dependence_path):
                os.mkdir(dependence_path)
            plt.savefig(f'{dependence_path}/dependence_plot_{name}_{i}.png')
            plt.clf()

    def plot_shap_feature_importance(self, model_dict, features_df, model_objects, shap_df=None):
        """TODO: y axis labels get cutoff. fix that."""
        if type(shap_df) is type(None):
            shap_df = self.get_shap_vals(model_dict, model_objects)

        for plot_type in ['bar','dot','violin']:
            self.shap_feature_importance_viz(model_dict, shap_df, features_df, plot_type)

    def shap_feature_importance_viz(self, model_dict, shap_df, features_df, plot_type):
        shap.summary_plot(
            shap_df.loc[:, model_dict['features_list']].values,
            features=features_df.loc[:, model_dict['features_list']].values,
            feature_names=model_dict['features_list'],
            plot_type=plot_type,
            show=(not self.plots_dict['save']['plots'])
        )
        if self.plots_dict['save']['plots'] is True:
            plt.savefig(
                '{}/shap_importance_{}.png'
                    .format(self.plots_dir, plot_type)
            )
        plt.clf()
