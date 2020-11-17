from IPython.display import display
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_flavor as pf
import seaborn as sns
import scikit_posthocs as sp
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.stats.api as sms
import statsmodels.stats.stattools as st
from statsmodels.stats.power import  tt_ind_solve_power
#plt.style.use('ggplot')

def bootstrap(A, B):
    combined = A + B
    resampled = np.random.choice(combined, size=len(combined), replace=True)
    resampled_A = resampled[:len(A)]
    resampled_B = resampled[len(A):]
    return resampled_A, resampled_B

def bootstrap_sim(dataframe, feature, target, control_groups=None, num_trials=20000, alternate='both', param='mean', p_adjust=False, show_hist=False):
    text_color = plt.rcParams.get('ytick.color')
    controls = []
    groups = dataframe.groupby(feature)[target]
    if control_groups is None:
        controls = [x[0] for x in groups]
    else:
        if type(control_groups) == str:
            controls = [control_groups]
        else:
            try:
                it = iter(control_groups)
            except:
                controls = [control_groups]
            else:
                for cont in control_groups:
                    controls.append(cont)
    results = pd.DataFrame()
    #k = len(groups) - 1
    #if len(controls) > 1:
    js = range(1, len(controls)+1)
    lst = [(len(groups) - j) for j in js]
    k = sum(lst)
    cols = len(groups) - 1
    row = 0
    string = ''
    
    if show_hist:
        nrows = (k-cols)//cols + 1
        if nrows == 1:
            vsize = 6
        else:
            vsize = 2
        fig, axes = plt.subplots(nrows=nrows, 
                                 ncols=(cols),
                                 figsize=(12, vsize*nrows)
                                )
        fig.tight_layout(h_pad=2)
        if nrows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
    
    if param == 'mean':
        param_function = np.mean
    elif param == 'median':
        param_function = np.median
    elif param == 'mode':
        param_function = pd.Series.mode
    elif param == 'var':
        param_function = np.var
    elif param == 'std':
        param_function = np.std
    else:
        return ("Error: invalid parameter passed.")
    
    print("Performing bootstrap simulation for parameter: {}".format(param))
    
    control_lists = []
    group_diffs = []
    combos = []
    measured_diffs = []
    prev_controls = []
    for control in controls:
        p_vals = {}
        #group_diffs = []
        print("Testing control group: {}".format(control))
        for name, group in groups:
            if name == control:
                control_group = group.copy()
                if param == 'mode':
                    control_param = np.mean(param_function(control_group))
                elif param == 'var' or param == 'std':
                    control_param = param_function(control_group, ddof=1)
                else:
                    control_param = param_function(control_group)
                
        for name, group in groups:
            group_p = {}
            diffs_list = []
            if name == control or name in prev_controls:
                continue
            if param == 'mode':
                exp_param = np.mean(param_function(group))
            elif param == 'var' or param == 'std':
                exp_param = param_function(group, ddof=1)
            else:
                exp_param = param_function(group)
            param_diff = exp_param - control_param
            further_diffs = 0

            if alternate == 'both':
                param_diff = np.abs(param_diff)            

            for i in range(num_trials):
                bA, bB = bootstrap(list(control_group), list(group))
                if param == 'mode':
                    diff = np.mean(param_function(bB)) - np.mean(param_function(bA))
                    diffs_list.append(diff)
                else:
                    diff = param_function(bB) - param_function(bA)
                    diffs_list.append(diff)

                if alternate == 'both':
                    diff = np.abs(diff)

                if alternate == 'lower':
                    if diff <= param_diff:
                        further_diffs += 1
                elif alternate == 'both' or alternate == 'higher':
                    if diff >= param_diff:
                        further_diffs += 1
                else:
                    print("Error: invalid alternative hypothesis. Options are 'both', 'higher', or 'lower'")
                    break

            p = further_diffs / num_trials
            if p_adjust:
                string = "p-values adjusted for {} group comparisons".format(k)
                p *= k
                if p > 1:
                    p = 1
            group_p['p_val to {}'.format(control)] = p
            p_vals[name] = group_p
            measured_diffs.append(param_diff)
            group_diffs.append(diffs_list)
            combos.append((control, name))
            prev_controls.append(control)
            
        result = pd.DataFrame.from_dict(p_vals)
        results = pd.concat([results, result], axis=0, sort=False)
            
    if show_hist:
        list_num = 0
        for ax in axes:
            if list_num > len(group_diffs):
                break
            group_list = group_diffs[list_num]
            diff_mean = round(np.mean(group_list), 2)
            diff_std = np.std(group_list, ddof=1)
            xs = np.linspace(min(group_list), max(group_list), 1000)
            ys = stats.norm.pdf(xs, loc=diff_mean, scale=diff_std)
            ax.plot(xs, ys, color='gray')
            ax.hist(group_list, alpha=0.6, density=True)
            ax.axvline(x=measured_diffs[list_num], 
                       ls=':', 
                       label='Mean: {}'.format(diff_mean),
                       color='black')
            ax.set_title('{} vs {}'.format(combos[list_num][0], combos[list_num][1]),
                        color=text_color)
            #ax.legend()
            list_num += 1

        plt.show()
    print(string)
    
    return results             

def check_normality(data, cols, display_results=True, drop_na=True):
    info = {}
    ad_results = {}
    jb_results = {}
    
    for col in cols:
        if type(data) == pd.core.frame.DataFrame:
            x = data[col].dropna() if drop_na else data[col]
        else:
            try:
                x = data.dropna() if drop_na else data
            except:
                x = data
        
         # Perform Anderson-Darling test on data
        stat, crit, p = stats.anderson(x, 'norm')
        ad_results[col] = {}
        ad_results[col]['statistic'] = stat
        ad_results[col]['critical'] = crit[2]
        
        info[col] = {}
        info[col]['K-S'] = {}
        info[col]['Shapiro-Wilk'] = {}
        info[col]['K-S']['Statistic'], info[col]['K-S']['p-value'] = stats.kstest(x, 'norm')
        info[col]['Shapiro-Wilk']['Statistic'], info[col]['Shapiro-Wilk']['p-value'] = stats.shapiro(x)
        
        jbstat, jbp, jbskew, jbkurt = st.jarque_bera(x)
        jb_results[col] = {}
        jb_results[col]['Statistic'] = jbstat
        jb_results[col]['p-value'] = jbp
        jb_results[col]['Skew'] = jbskew
        jb_results[col]['Kurtosis'] = jbkurt

    dict_of_df = {k: pd.DataFrame(v) for k,v in info.items()}
    test_results = pd.concat(dict_of_df, axis=0)
    #mux = pd.MultiIndex.from_tuples(ad_results.keys())
    ad_results = pd.DataFrame.from_dict(ad_results, orient='index')
    jb_results = pd.DataFrame.from_dict(jb_results, orient='index')
    #ad_results = pd.DataFrame(ad_results, index=mux)
    if display_results == True:
        print("Normality Test Results for {}:".format(cols))
        print("-------------------------------------------------------------------------------------------")
        names = ["K-S and Shapiro-Wilk:", "Anderson-Darling:", "Jarque-Bera:"]
        display_side_by_side(test_results, ad_results, jb_results.T, names=names)
        
    return test_results, ad_results

def cohen_d(A, B):
    n1, n2 = len(A), len(B)
    std1, std2 = np.std(A, ddof=1), np.std(B, ddof=1)
    mean1, mean2 = np.mean(A), np.mean(B)
    numerator = mean1-mean2
    pooled_sd = np.sqrt(((n1-1)*(std1**2) + (n2-1)*(std2**2)) / (n1+n2-2))
    d = numerator / pooled_sd
    return d

def combT(a,b):
    universal_set = sorted(a + b)
    combinations = set(itertools.combinations(universal_set, len(a)))
    groupings = []
    for combination in combinations:
        temp_list = universal_set.copy()
        for element in combination:
            temp_list.remove(element)
        groupings.append((list(combination), temp_list))
    return groupings

def compare_groups(dataframe, feature, targets, control_group=None, alpha=0.05, p_adjust=False, show_groups=True, **kwargs):
    figsize = (12,8)
    edgecolor = None
    # Deal with keyword arguments
    for k, v in kwargs.items():
        if k not in ['figsize','edgecolor']:
            raise TypeError("compare_groups got an unexpected keyword argument {}".format(k))
        else:
            if k == 'figsize':
                figsize = v
            elif k == 'edgecolor':
                edgecolor = v
    text_color = plt.rcParams.get('ytick.color')
    # Deal with targets input
    if type(targets) == str:
        targets = [targets]
    for target in targets:
        control = None
        info = {}
        grouped = dataframe.groupby([feature])[target]
        if control_group is None:
            control_group = grouped.iloc[0][0]
        k = len(grouped) - 1
        for group in grouped:
            temp = {}
            if group[0] == control_group:
                control = np.array(group[1])
                continue
            else:
                test_group = np.array(group[1])
                size = len(test_group)
                if size == 1:
                    mu, std = control.mean(), control.std(ddof=1)
                    effect_size = np.abs((test_group[0] - mu) / std)
                    p = 2 * stats.norm.sf(effect_size)                    
                else:
                    stat, p = stats.ttest_ind(test_group, control, equal_var=False)
                    effect_size = cohen_d(test_group, control)
                if p_adjust:
                    p *= k
                    if p > 1:
                        p = 1
                
                temp['p-val'] = p
                temp['effect size'] = effect_size
                temp['size'] = size
                temp['power'] = tt_ind_solve_power(effect_size = effect_size,
                                                  nobs1 = size,
                                                  alpha = alpha,
                                                  ratio = len(control) / size)
                info[group[0]] = temp

        info = pd.DataFrame.from_dict(info)
        print('Testing {} groups for statistically significant effects on {}'.format(feature, target))
        display(info.round(6))
        
        # Plot test results
        X = list([str(x) for x in info.columns])
        if not show_groups:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=figsize, 
                                           gridspec_kw={"hspace": 0.05})
        else:
            fig = plt.figure(constrained_layout=True, figsize=figsize)
            gs = fig.add_gridspec(2, 2)
            ax = fig.add_subplot(gs[:, 0])
            if edgecolor is not None:
                ax = sns.boxplot(x=dataframe[feature], y=dataframe[target],
                                 whiskerprops={'color': edgecolor},
                                 capprops={'color': edgecolor},
                                 flierprops={'markerfacecolor': edgecolor,
                                             'markeredgecolor': edgecolor}
                                )
            else:
                ax = sns.boxplot(x=dataframe[feature], y=dataframe[target])
            # fix edgecolors if needed:
            #if edgecolor is not None:
            #    for i, artist in enumerate(ax.artists):
            #        #artist.set_edgecolor(edgecolor)
            #        for j in range(i*6, i*6+6):
            #            if j in range(i*6+4, i*6+6):
            #                continue
            #            line = ax.lines[j]
            #            line.set_color(edgecolor)
            #            line.set_mfc(edgecolor)
            #            line.set_mec(edgecolor)
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 1])
        ax1.set_title('Target: {}'.format(target), color=text_color)
        if len(grouped) - 1 == 1:
            ax1.scatter(X,
                        info.loc['p-val'], 
                        color='#3572C6', 
                        label='p-value', 
                        marker='x',
                        linewidth=4,
                        s=50,
                       )
        else:
            ax1.plot(X, info.loc['p-val'], color='#3572C6', label='p-value')
        ax1.axhline(y=alpha, ls='-.', label='alpha: {}'.format(alpha), alpha=0.7)
        ax1.set(xlabel='')
        ax1.legend()
        if len(grouped) - 1 == 1:
            ax2.scatter(X,
                        info.loc['effect size'], 
                        color='g', 
                        label='effect size', 
                        marker='x',
                        linewidth=4,
                        s=50,
                       )
        else:
            ax2.plot(X, info.loc['effect size'], color='g', label='effect size')
        ax2.set_xlabel('{}'.format(feature), color=text_color)
        ax2.legend()
        plt.show()

from IPython.display import display_html
def display_side_by_side(*args, names=None):
    html_str=''
    html_str+='<table>'
    for i, df in enumerate(args):
        html_str+='<td>'
        if names:
            name_str = names[i]+'<br/>'
            html_str+=name_str
        html_str+=df.to_html()
        html_str+='</td>'
    html_str+='</table></body>'
    display_html(html_str.replace('table','table style="display:inline" cellpadding=100'),raw=True)
    
def do_a_linreg(dataframe, features, target):
    linreg_type = None
    text_color = plt.rcParams.get('ytick.color')
    if type(features) == str:
        try:
            np.issubdtype(dataframe[features].dtype, np.number)
        except:
            linreg_type = 'multi'
        else:
            linreg_type = 'simple'
        predictors = features
        feature = features
    else:
        try:
            it = iter(features)
        except:
            targets = [targets]
        if len(features) == 1:
            try:
                np.issubdtype(dataframe[features[0]].dtype, np.number)
            except:
                linreg_type = 'multi'
            else:
                linreg_type = 'simple'
                feature = features[0]
            predictors = features[0]
        else:
            linreg_type = 'multi'
            predictors = '+'.join(features)
    formula = target + '~' + predictors
    print("Linear Regression for {}".format(formula))
    
    # Use scipy to generate a graph of regression over data
    if linreg_type == 'simple':
        linreg = LinearRegression().fit(np.reshape([dataframe[feature]],(-1,1)), dataframe[target])
        X = np.linspace(dataframe[feature].min(), dataframe[feature].max(), 500).reshape(-1,1)
        y = linreg.predict(X)
        plt.figure(figsize=(12,6))
        plt.scatter(dataframe[feature], dataframe[target], 
            #color='green'
            )
        plt.xlabel(feature, color=text_color)
        plt.ylabel(target, color=text_color)
        plt.title("Linear Regression for {} ~ {}".format(target,feature), color=text_color)
        #plt.xlim(dataframe[feature].min() - np.abs(dataframe[feature].min()*0.01),
        #	dataframe[feature].max() + np.abs(dataframe[feature].min()*0.01)
        #	)
        #print('HI')
        plt.plot(X,y, color='orange', label='predictions')
        plt.legend()
        plt.show()
    
    # Build statsmodels mode
    model = ols(formula, dataframe).fit()
    display(model.summary())
    
    # Plot Q-Q plot of residuals
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    ax1.set_title("Q-Q plot for model residuals", color=text_color)
    sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True, ax=ax1);
    
    # Perform Goldfeld-Quandt test of homoscedasticity
    temp = pd.concat([dataframe[features], dataframe[target]], axis=1)
    GQ_results = goldfeld_quandt(temp, target, model, ax2)
    
    print("Test for homoscedasticity of residuals:")
    display(GQ_results)
    # Perform normality check on residuals
    check_normality(model.resid, cols=['residuals']);
    print("Skew:", skew(model.resid))
    print("Kurtosis:", kurtosis(model.resid))
    
    plt.show()
    
    return model

def ecdf(data, group_by=None, targets=None, ax=None, **kwargs):
    """Produces ECDF graphs for input data. Inputs can be 1d array-like, pandas Series, or
    pandas DataFrame. If a DataFrame is passed, group_by and targets may be set for group 
    comparisons. If no target is set for a DataFrame, all columns will be graphed."""
    text_color = plt.rcParams.get('ytick.color')
    linewidth = 2
    # Handle keyword arguments
    for k, v in kwargs.items():
        if k not in ['linewidth']:
            raise TypeError('ecdf got an unexpeted keyword argument: {}'.format(k))
        else:
            if k == 'linewidth':
                linewidth = v
    # Deal with input data
    if group_by is not None:
        if type(data) == pd.core.frame.DataFrame:
            print("Grouping DataFrame by {}".format(group_by))
            print("Target Features:", targets)
            if type(targets) == str:
                targets = [targets]
            else:
                try:
                    it = iter(targets)
                except:
                    targets = [targets]
            cols = targets + [group_by]
            data = data[cols]
            variables = data.columns[:-1]
            data = data.groupby(group_by)
        else:
            return("Error: only DataFrame input works with group_by functionality")
    else:      
        if type(data) == pd.core.series.Series:
            variables = [data.name]
        elif type(data) == pd.core.frame.DataFrame:
            if targets is None:
                variables = list(data.columns)
            else:
                if type(targets) == str:
                    targets = [targets]
                else:    
                    try:
                        it = iter(targets)
                    except:
                        targets = [targets]
                print("Target Features:", targets)
                variables = targets
        elif type(data) == pd.core.groupby.generic.DataFrameGroupBy:
            variables = list(data.obj.columns)
        else:
            data = pd.Series(data, name='data')
            variables = [data.name]
    
    
    if type(data) == pd.core.groupby.generic.DataFrameGroupBy:
        for variable in variables:
            if not ax:
                fig, ax = plt.subplots(figsize=(12,8))
            max_x = 0
            for name, group in data:
                x = np.sort(group[variable])
                n = len(group)
                y = np.arange(1, n+1) / n
                ax.plot(x, y, marker='.', label=name, alpha=0.7, linewidth=linewidth)
                if max(x) > max_x:
                    max_x = max(x)
                    #max_x = 0
            ax.axhline(y=0.5, ls=':', color='gray')
            ax.axhline(y=0.05, ls=':', color='gray')
            ax.axhline(y=0.95, ls=':', color='gray')
            ax.annotate('0.5', xy=(max_x, 0.47))
            ax.annotate('0.95', xy=(max_x, 0.92))
            ax.annotate('0.05', xy=(max_x, 0.02))
            ax.legend()
            plt.title("ECDF for feature: {}".format(variable), color=text_color)
            plt.show()
                
    else:
        n = len(data)
        y = np.arange(1, n+1) / n
        if not ax:
            fig, ax = plt.subplots(figsize=(12,8))
        max_x = 0
        for variable in variables:
            if type(data) == pd.core.series.Series:
                x = np.sort(data)
                string = variable
            else:
                x = np.sort(data[variable])
                string = 'Data'
            ax.plot(x, y, marker='.', label=variable)
            if max(x) > max_x:
                max_x = max(x)
        ax.axhline(y=0.5, ls=':', color='gray')
        ax.axhline(y=0.05, ls=':', color='gray')
        ax.axhline(y=0.95, ls=':', color='gray')
        ax.annotate('0.5', xy=(max_x, 0.47))
        ax.annotate('0.95', xy=(max_x, 0.92))
        ax.annotate('0.05', xy=(max_x, 0.02))
        plt.title("ECDF for {}".format(string), color=text_color)
        plt.legend()
        plt.show()

def explore_groups(dataframe, feature, target):
    percentages = {}
    group_stats, groups = group_hist(dataframe, feature, target, show_hist=False, return_groups=True)
    for name, group in groups:
        percentages[name] = group.sum() / dataframe[target].sum()
    percent_df = pd.DataFrame.from_dict(percentages, orient='index')
    column_name = 'Percent of Total {}'.format(target)
    percent_df.columns = [column_name]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(12,12), 
                                       gridspec_kw={"height_ratios": (.40, .20, .40),
                                                   "hspace": 0.05})
    percent_df[column_name].plot(kind='bar', ax=ax1, alpha=0.7)
    ax1.legend()
    ax1.set_title('Percent of Total {}, Order Frequency, and Avg {} per {}'.format(target,
                                                                                   target,
                                                                                   feature),
                 c='gray', size=16)
    ax2.plot(list(range(len(group_stats))), group_stats.Size, label='Order Frequency', 
             c='green', alpha=0.7)
    ax2.legend()
    ax3 = group_stats.Mean.plot(kind='bar', label='Avg {}'.format(target), alpha=0.6, color='blue')
    ax3.legend()
    plt.show()
    
    return pd.concat([group_stats, percent_df], axis=1)

def f_test(var1, var2, df1, df2, alternate='both'):
    F = var1/var2
                
    if alternate == 'both':
        if F > 1:
            p = stats.f.sf(F, df1, df2) * 2
        elif F <= 1:
            p = stats.f.cdf(F, df1, df2) * 2
    elif alternate == 'lower':
        p = stats.f.cdf(F, df1, df2)
    elif alternate == 'higher':
        p = stats.f.sf(F, df1, df2)
    else:
        return ("Error: invalid alternate hypothesis. Choices: 'both', 'lower', 'higher'")
    
    return p

def f_test_groups(data, group_var, target, alternate='both'):
    groups = data.groupby(group_var)[target]
    scores = {}
    
    for name1, group1 in groups:
        group1_scores = {}
        var1 = np.var(group1, ddof=1)
        df1 = len(group1) - 1
        
        for name2, group2 in groups:
            
            if name2 != name1:
                var2 = np.var(group2, ddof=1)
                df2 = len(group2) - 1
                p = f_test(var1, var2, df1, df2)
                group1_scores[name2] = p
                
        scores[name1] = group1_scores
    
    scores = pd.DataFrame(scores).sort_index()
    
    return scores         

def goldfeld_quandt(dataframe, target, model, ax=None, alternative='two-sided'):
    text_color = plt.rcParams.get('ytick.color')
    exog = pd.DataFrame(model.model.exog)
    endog = pd.DataFrame(model.model.endog, columns=[target])
    temp = pd.concat([endog, exog], axis=1)
    temp = temp.sort_values(target).reset_index(drop=True)
    #display(temp.head())
    #dataframe = dataframe.reset_index()
    #temp = dataframe.sort_values(by=target).reset_index()
    #temp = temp.rename(columns={'index':'old_index'})
    #display(temp)
    lwr_thresh = dataframe[target].quantile(q=.45)
    upr_thresh = dataframe[target].quantile(q=.55)
    #lower_indices = temp[temp[target] <= lwr_thresh].index
    #upper_indices = temp[temp[target] >= upr_thresh].index
    middle_10percent_indices = dataframe[(dataframe[target] >= lwr_thresh) & (dataframe[target]<=upr_thresh)].index
    indices = [x for x in dataframe.index if x not in middle_10percent_indices]
    #print(indices)
    #return indices
    if not ax:
        fig, ax = plt.subplots(figsize=(6,6))
    #ax.scatter(temp[target].iloc[indices], model.resid.iloc[indices])
    features = [x for x in dataframe.columns if x not in [target]]
    #predictions = model.predict(dataframe.loc[indices][features])
    #ax.scatter(predictions, model.resid.loc[indices])
    predictions = model.predict(dataframe[features])
    #predictions = model.predict(model.model.exog)
    ax.scatter(predictions, model.resid)
    ax.set_xlabel(target+' predictions', color=text_color)
    ax.set_ylabel('Model Residuals', color=text_color)
    ax.set_title("Residuals versus {} predictions".format(target), color=text_color)
    #ax.axvline(x=lwr_thresh, ls=':',linewidth=2, color='gray')
    #ax.axvline(x=upr_thresh, ls=':',linewidth=2, color='gray')
    ax.axhline(y=0, c='r')
    if not ax:
        plt.show()
    #test = sms.het_goldfeldquandt(model.resid.iloc[indices], model.model.exog[indices])
    test = sms.het_goldfeldquandt(#model.resid.iloc[indices], 
                                  temp[target],
                                  #model.model.endog[indices],
                                  temp[[x for x in temp.columns if x not in [target]]],
                                  split=0.45,
                                  drop=0.10,
                                  alternative=alternative
                                  )
    #print(test)

    #var1 = np.var(temp.iloc[upper_indices][target])
    #var2 = np.var(temp.iloc[lower_indices][target])
    #df1 = len(temp.iloc[upper_indices]) - 1
    #df2 = len(temp.iloc[lower_indices]) - 1
    #p = f_test(var1, var2, df1, df2)
    results = pd.DataFrame(index=['Goldfeld-Quandt'], columns=['F_statistic', 'p_value'])
    results.loc['Goldfeld-Quandt','F_statistic'] = test[0]
    results.loc['Goldfeld-Quandt','p_value'] = test[1]
    #results.loc['Goldfeld-Quandt','F_statistic'] = var1/var2
    #results.loc['Goldfeld-Quandt','p_value'] = p
    return results 

def group_hist(data, feature, target, show_hist=True, return_groups=False):
    text_color = plt.rcParams.get('ytick.color')
    print('Showing stats for {} grouped by {}'.format(target, feature))
    grouped = data.groupby([feature])[target]
    stats = {}
    for group in grouped:
        temp = pd.Series(group[1])
        stats[group[0]] = {}
        stats[group[0]]['Mean'] = temp.mean()
        stats[group[0]]['Median'] = temp.median()
        stats[group[0]]['Std'] = temp.std(ddof=1)
        stats[group[0]]['Size'] = len(temp)
    
    stats = pd.DataFrame.from_dict(stats, orient='index')
    display(stats)
    
    if show_hist:
        plt.figure(figsize=(12,7))
        grouped.hist(density=True, histtype='step', alpha=1, stacked=True, lw=2)
        plt.legend([x[0] for x in grouped])
        plt.title('Grouped Histogram for {}'.format(target), color=text_color)
        plt.xlabel(feature)
        plt.show()
    
    if return_groups:
        return stats, grouped
    else:
        return stats

def make_boxplot(dataframe, x, y):
    text_color = plt.rcParams.get('ytick.color')
    fig, ax = plt.subplots(figsize=(12,6))
    plt.title("Side by side comparison of all group distributions:", color=text_color)
    sns.boxplot(x=dataframe[x], y=dataframe[y])
    plt.show()

# A function for metropolis MCMC algorithm:
def metropolis(data1, theta_seed1, theta_std1, data2=None, theta_seed2=None, theta_std2=None, samples=10000):
    theta_curr1 = theta_seed1
    posterior_thetas1 = []
    graph_thetas1 = []
    scaleA = np.std(data1, ddof=1)
    n1 = len(data1)
    calc_mean1 = np.mean(data1)
    post_std = theta_std1
    
    if data2 is not None:
        theta_curr2 = theta_seed2
        calc_mean2 = np.mean(data2)
        posterior_thetas2 = []
        theta_diffs = []
        effect_sizes = []
        graph_thetas2 = []
        scaleB = np.std(data2, ddof=1)
        actual_diff = calc_mean1 - calc_mean2
        actual_effect = actual_diff/np.sqrt((scaleA**2 + scaleB**2)/2)
        print("Performing MCMC for two groups")
        print("Mean of Group 1:", calc_mean1)
        print("Mean of Group 2:", calc_mean2)
        print("Measured Mean Difference:", actual_diff)
        print("Measured Effect Size:", actual_effect)
    
    for i in range(samples):
        theta_prop1 = np.random.normal(loc=theta_curr1, scale=post_std)
        likelihood_prop1 = 1
        if i == 0:
            likelihood_curr1 = 1
        #scaleA = min([np.random.normal(loc=scaleA, scale=0.05), 0])
        if data2 is not None:
            theta_prop2 = np.random.normal(loc=theta_curr2, scale=theta_std2)
            likelihood_prop2 = 1
            likelihood_curr2 = 1
            #scaleB = min([np.random.normal(loc=scaleB, scale=0.05), 0])
        #print(theta_prop1)
        
        #data1 = np.random.normal(loc=calc_mean1, scale=scaleA, size=n1)
        #mean1 = data1.mean()
        for datum in data1:
            pd_prop = stats.norm.pdf(x=datum, loc=theta_prop1, scale=scaleA)
            likelihood_prop1 *= pd_prop
            if i == 0:
                pd_curr = stats.norm.pdf(x=datum, loc=theta_curr1, scale=scaleA)
                likelihood_curr1 *= pd_curr
        
        posterior_prop1 = likelihood_prop1 * stats.norm.pdf(x=theta_prop1, loc=theta_curr1, scale=theta_std1)
        if i == 0:
            posterior_curr1 = likelihood_curr1 * stats.norm.pdf(x=theta_curr1, loc=theta_curr1, scale=theta_std1)
        #posterior_prop1 = likelihood_prop1 * stats.uniform.pdf(x=theta_prop1, loc=theta_curr1, scale=theta_std1)
        #posterior_curr1 = likelihood_curr1 * stats.uniform.pdf(x=theta_curr1, loc=theta_curr1, scale=theta_std1)
        
        if data2 is not None:
            for datum in data2:
                pd_prop = stats.norm.pdf(x=datum, loc=theta_prop2, scale=scaleB)
                likelihood_prop2 *= pd_prop
                if i == 0 :
                    pd_curr = stats.norm.pdf(x=datum, loc=theta_curr2, scale=scaleB)
                    likelihood_curr2 *= pd_curr
                
            posterior_prop2 = likelihood_prop2 * stats.norm.pdf(x=theta_prop2, loc=theta_curr2, scale=theta_std2)
            if i == 0:
                posterior_curr2 = likelihood_curr2 * stats.norm.pdf(x=theta_curr2, loc=theta_curr2, scale=theta_std2)
            #posterior_prop2 = likelihood_prop2 * stats.uniform.pdf(x=theta_prop2, loc=theta_curr2, scale=theta_std2)
            #posterior_curr2 = likelihood_curr2 * stats.uniform.pdf(x=theta_curr2, loc=theta_curr2, scale=theta_std2)
        
        # Prevents division by zero:
        if posterior_curr1 == 0.0:
            posterior_curr1 = 2.2250738585072014e-308
        if data2 is not None and posterior_curr2 == 0.0:
            posterior_curr2 = 2.2250738585072014e-308
            
        p_accept_theta_prop1 = posterior_prop1/posterior_curr1
        rand_unif = np.random.uniform()
        if p_accept_theta_prop1 >= rand_unif:
            #post_mean, post_std, posterior = make_posterior(calc_mean1, theta_prop1, scaleA, post_std)
            theta_curr1 = theta_prop1
            posterior_curr1 = posterior_prop1
            #scaleA = scaleA
        posterior_thetas1.append(theta_curr1)
        if i % (samples/10) == 0:
            graph_thetas1.append(theta_curr1)
        
        if data2 is not None:
            #print(posterior_prop2, posterior_curr2)
            p_accept_theta_prop2 = posterior_prop2/posterior_curr2
            rand_unif = np.random.uniform()
            if p_accept_theta_prop2 >= rand_unif:
                theta_curr2 = theta_prop2
                posterior_curr2 = posterior_prop2
                
            posterior_thetas2.append(theta_curr2)
            theta_diff = theta_curr1 - theta_curr2
            theta_diffs.append(theta_diff)
            effect_sizes.append(theta_diff/np.sqrt((scaleA**2 + scaleB**2)/2))
        
            if i % (samples/10) == 0:
                graph_thetas2.append(theta_curr2)
                
    if data2 is not None:
        # Visualizing results of MCMC
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, 
                                                                 ncols=2,
                                                                 figsize=(12,12))
        ax1.hist(data1, density=True, alpha=0.6)
        ax1.set_title("Data Group 1 w. Post. Pred")
        ax1.axvline(x=calc_mean1, ls=':', label='Group 1 Mean: {}'.format(calc_mean1))
        ax1.legend()
        xs = np.linspace(data1.min(), data1.max(), 1000)
        for theta in graph_thetas1:
            ys = stats.norm.pdf(xs, loc=theta, scale=scaleA)
            ax1.plot(xs, ys, color='gray')
        ax2.hist(posterior_thetas1, density=True, alpha=0.6)
        ax2.set_title("Posterior for Theta, Group 1")
        ax2.axvline(x=np.mean(posterior_thetas1), ls=':', label='Mean of Posterior 1: {}'.format(np.mean(posterior_thetas1)))
        ax2.legend()
        ax3.hist(data2, density=True, alpha=0.6)
        ax3.set_title("Data Group 2 w. Post. Pred")
        ax3.axvline(x=calc_mean2, ls=':', label='Group 2 Mean: {}'.format(calc_mean2))
        ax3.legend()
        xs = np.linspace(data2.min(), data2.max(), 1000)
        for theta in graph_thetas2:
            ys = stats.norm.pdf(xs, loc=theta, scale=scaleB)
            ax3.plot(xs, ys, color='gray')
        ax4.hist(posterior_thetas2, density=True, alpha=0.6)
        ax4.set_title("Posterior for Theta, Group 2")
        ax4.axvline(x=np.mean(posterior_thetas2), ls=':', label='Mean of Posterior 2:: {}'.format(np.mean(posterior_thetas2)))
        ax4.legend()
        ax5.hist(theta_diffs, density=True, alpha=0.6)
        ax5.set_title("Differences btw Theta 1 and 2")
        ax5.axvline(x=np.mean(theta_diffs), ls=':', label='Mean Difference: {}'.format(np.mean(theta_diffs)))
        ax5.legend()
        ax6.hist(effect_sizes, density=True, alpha=0.6)
        ax6.set_title("Effect Sizes")
        ax6.axvline(x=np.mean(effect_sizes), ls=':', label='Mean Effect Size: {}'.format(np.mean(effect_sizes)))
        ax6.legend()
        plt.show()
        
        # Producing probability of null hypothesis:
        sizes = np.array(theta_diffs)
        sizes_mu = sizes.mean()
        sizes_std = sizes.std()
        conf_interval = stats.norm.interval(0.95, loc=sizes_mu, scale=sizes_std)
        if np.mean(theta_diffs) >= 0:
            calc_p_val = ((sum(sizes < 0) / len(sizes)) * 2)
            norm_p_val = (stats.norm.cdf(0, loc=sizes_mu, scale=sizes_std) * 2)
        else:
            calc_p_val = ((sum(sizes > 0) / len(sizes)) * 2)
            norm_p_val = (stats.norm.sf(0, loc=sizes_mu, scale=sizes_std) * 2)

        print("P_value numerically:", calc_p_val)
        print("P_value from normal dist:", norm_p_val)
        print("95% Confidence Interval for Mean Difference:", conf_interval)
        
        return theta_diffs
        
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        ax1.set_title('Group 1 Data w Post. Pred')
        ax1.hist(data1, density=True, alpha=0.6)
        ax1.axvline(x=calc_mean1, ls=':', color='g', label='Measured mean: {}'.format(calc_mean1))
        xs = np.linspace(min(data1), max(data1), 1000)
        for theta in graph_thetas1:
            ys = stats.norm.pdf(xs, loc=theta, scale=scaleA)
            ax1.plot(xs, ys, color='gray')
        ax1.legend()
        
        ax2.set_title('MCMC Mean Frequencies')
        ax2.hist(posterior_thetas1, density=True, alpha=0.6)
        mcmc_theta1 = np.mean(posterior_thetas1)
        ax2.axvline(x=mcmc_theta1, ls=':', color='g', label='MCMC mean: {}'.format(mcmc_theta1))
        ax2.legend()
        plt.show()
        
        return posterior_thetas1

def norm_pdf(x, mu, std):
    var = std**2
    part1 = 1/(np.sqrt(2*np.pi)*std)
    part2 = np.exp(-1*((x-mu)**2)/(2*var))
    pd = part1 * part2
    return pd

# A function to run permutation tests:
def permutation(dataframe, feature, target, control=0.0, alternate='both'):
    
    for name, group in dataframe.groupby(feature)[target]:
        if len(group) == 0:
            continue
        p_vals = {}
        
        if name == control:
            control_group = group
            # To manage exploding numbers of combinations, need to take sample if n too large
            N = len(control_group)
            if N > 50:
                # Use Slovin's formula to figure out the sample size that we will need
                e = .05
                n = int(round((N / (1 + N*(e**2))), 0))
                print("Sampling control group with size {}".format(n))
                control_group = np.random.choice(control_group, size=n, replace=False)
        else:
            mean_diff = group.mean()
            further_diffs = 0
            group_dict = {}
            groupings = combT(list(group), list(control_group))
            print("Number of Groupings for {} group:".format(name), len(groupings))
            for grouping in groupings:
                mean1 = np.mean(grouping[0])
                mean2 = np.mean(grouping[1])
                diff = mean1 - mean2
                if alternate == 'lower':
                    if diff <= mean_diff:
                        further_diffs += 1
                elif alternate == 'both':
                    if np.abs(diff) >= np.abs(mean_diff):
                        further_diffs += 1
                elif alternate == 'higher':
                    if diff >= mean_diff:
                        further_diffs += 1
                else:
                    print("Error: invalid alternate hypothesis. Options are 'both', 'lower', 'higher'")
            p_val = further_diffs / len(groupings)
            group_dict['p-value'] = p_val
            p_vals[name] = group_dict
            
    test_results = pd.DataFrame.from_dict(p_vals)
    return test_results

def pooled_variance(groups):
    info = {}
    names = []
    for name, group in groups:
        names.append(name)
        info[name] = {}
        info[name]['n'] = len(group)
        info[name]['var'] = group.var(ddof=1)
    k = len(info.keys())
    numer = sum([(info[name]['n'] - 1)*info[name]['var'] for name in names])
    denom = sum([info[name]['n'] for name in names]) - k
    pooled_var = np.sqrt(numer/denom)
    return pooled_var

def sigma_trim(df, col, num_sigmas=3):
    range_ = num_sigmas * df[col].std(ddof=1)
    mean = df[col].mean()
    upper_boundary = mean + range_
    lower_boundary = mean - range_
    trimmed_df = df[(df[col] <= upper_boundary) & (df[col] >= lower_boundary)]
    print("Length of old DataFrame:", len(df))
    print("Length of Trimmed DataFrame:", len(trimmed_df))
    print(" ")
    return trimmed_df

def standardize(x):
    return((x-np.mean(x))/np.sqrt(np.var(x)))

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def trim(data, percent, side='both'):
    data = np.array(sorted(data))
    trim = int(percent*data.size/100.0)
    if side == 'both':
        return data[trim:-trim]
    elif side == 'low':
        return data[trim:]
    elif side == 'high':
        return data[:-trim]
    else:
        print("Error: improper value for side. Options are 'both', 'high', 'low'.")

def trim_df_by_col(df, col, percent, side='both'):
    trimmed_data = trim(df[col], percent, side)
    trimmed_df = df[df[col].isin(trimmed_data)]
    print("Length of old DataFrame:", len(df))
    print("Length of trimmed DataFrame:", len(trimmed_df))
    print(" ")
    return trimmed_df    

def tukey_trim(data, col, coef=1.5):
    # Found a useful function in the scikit_posthocs that uses the Tukey fence method
    trimmed = sp.outliers_iqr(data[col], coef=coef)
    print("Length of old DataFrame:", len(data))
    data = data[data[col].isin(trimmed)]
    print("Length of Tukey trimmed DataFrame", len(data))
    return(data)

def visualize_distribution(data, targets=None, drop_na=True):
    text_color = plt.rcParams.get('ytick.color')
    if type(targets) == str:
        targets = [targets]
    results = check_normality(data, targets, display_results=False)
    if type(data) != pd.core.frame.DataFrame:
        targets = ['Data']
        temp = pd.DataFrame()
        temp['Data'] = data
        data = temp
    
    for target in targets:
        if drop_na == True:
            targ = data[target].dropna()
        else:
            targ = data[target]
        mean = round(np.mean(targ), 5)
        median = round(np.median(targ), 5)
        print("Variable: {}".format(target))
        test_results = check_normality(data, [target], drop_na=drop_na)
        display(targ.describe())
        print("Skew: {}".format(skew(targ)))
        print("Kurtosis: {}".format(kurtosis(targ)))
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12,6), 
                                   gridspec_kw={"height_ratios": (.15, .85),
                                               "hspace": 0.05})
        ax1.set_title('Distribution of {}'.format(target), color=text_color)
        sns.boxplot(targ, ax=ax1)
        ax1.set(xlabel='')
        ax1.annotate(s='Median: {}'.format(median), 
                    xy=(0.84, 0.1), xycoords="axes fraction")
        sns.distplot(targ, ax=ax2)
        ax2.axvline(x=mean, ls=':', label='Mean: {}'.format(mean),
                    linewidth=2,
                    #color='gray'
                    )
        ax2.legend()
        plt.show()
        
        fig2, (ax,ax2) = plt.subplots(ncols=2, figsize=(12,5))
        stats.probplot(targ, dist='norm', plot=ax)
        ax.set_title('Q-Q Plot for {}'.format(target), color=text_color)
        ecdf(targ, ax=ax2)
        
        plt.show()
        
    #return results

def winsorize_df(df, col, percent):
    percent *= .01
    print("Length of old DataFrame:", len(df))
    dataframe = df.copy()
    dataframe[col] = stats.mstats.winsorize(df[col], limits=(percent,percent), inplace=False)
    print("Length of Winsorized DataFrame:", len(dataframe))
    print(" ")
    return dataframe

def dunnets_tstat(expr, ctrl, pooled_var):
    expr_mean = expr.mean()
    ctrl_mean = ctrl.mean()
    n1, n2 = len(expr), len(ctrl)
    return (expr_mean - ctrl_mean) / np.sqrt(pooled_var * ((1/(n1))+(1/(n2))))