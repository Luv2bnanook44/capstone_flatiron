import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def cigarette_use_by_year(data):

    years = [i for i in range(1995, 2020)]
    graph = data

    conds = [
        (graph['evr_smoked_cig'] == 'Never'),
        ((graph['evr_smoked_cig'] != 'Never') & (
            graph['evr_smoked_cig'] != 'Unknown')),
        (graph['evr_smoked_cig'] == 'Unknown')]
    choices = [0, 1, 'Unknown']
    graph['cig_binary'] = np.select(conds, choices)

    cig_users_lifetime = [graph[(graph['yr_adminst'] == yr) & (graph['cig_binary'] != 'Unknown')]['cig_binary'].astype(
        int).sum() / len(graph[(graph['evr_smoked_cig'] != 'Unknown') & (graph['yr_adminst'] == yr)]) for yr in years]

    plt.title('Cigarette Users: 1995-2019', size=20)
    plt.plot(years, cig_users_lifetime, color='tab:blue', marker='o', lw=4)
    plt.xlabel('Year of Survey')
    plt.ylabel('Percent of Students')


def marijuana_use_by_year(data):

    weed_users_lifetime = None
    weed_users_yr = None
    weed_users_month = None
    years = [i for i in range(1995, 2020)]
    graph = data
    times = ['lifetime', 'yr', 'month']
    for time in times:
        conds = [
            (graph[f'weed_hash_{time}_freq'] == '0'),
            ((graph[f'weed_hash_{time}_freq'] != '0') & (
                graph[f'weed_hash_{time}_freq'] != 'Unknown')),
            (graph[f'weed_hash_{time}_freq'] == 'Unknown')]
        choices = [0, 1, 'Unknown']
        graph[f'weed_{time}_binary'] = np.select(conds, choices)
        if time == 'lifetime':
            weed_users_lifetime = [graph[(graph['yr_adminst'] == yr) & (graph['weed_lifetime_binary'] != 'Unknown')]['weed_lifetime_binary'].astype(
                int).sum() / len(graph[(graph[f'weed_hash_{time}_freq'] != 'Unknown') & (graph['yr_adminst'] == yr)]) for yr in years]
        elif time == 'yr':
            weed_users_yr = [graph[(graph['yr_adminst'] == yr) & (graph['weed_yr_binary'] != 'Unknown')]['weed_yr_binary'].astype(
                int).sum() / len(graph[(graph[f'weed_hash_{time}_freq'] != 'Unknown') & (graph['yr_adminst'] == yr)]) for yr in years]
        elif time == 'month':
            weed_users_month = [graph[(graph['yr_adminst'] == yr) & (graph['weed_month_binary'] != 'Unknown')]['weed_month_binary'].astype(
                int).sum() / len(graph[(graph[f'weed_hash_{time}_freq'] != 'Unknown') & (graph['yr_adminst'] == yr)]) for yr in years]
        else:
            print("Whoops, something's wrong here.")
    plt.title('Marijuana/Hash Users: 1995-2019', size=20)
    plt.plot(
        years,
        weed_users_lifetime,
        label='Lifetime',
        color='tab:blue',
        marker='o')
    plt.plot(
        years,
        weed_users_yr,
        label='Past Year',
        color='tab:green',
        marker='o')
    plt.plot(
        years,
        weed_users_month,
        label='Past Month',
        color='tab:red',
        marker='o')
    plt.xlabel('Year of Survey')
    plt.ylabel('Percent of Students')
    plt.legend()


def other_drug_use_by_year(data):

    drug_cols_lifetime = [
        'lsd_lifetime_freq',
        'pysd_lifetime_freq',
        'coke_lifetime_freq',
        'amph_lifetime_freq',
        'meth_lifetime_freq',
        'sedbarb_lifetime_freq',
        'tranq_lifetime_freq',
        'heroin_lifetime_freq',
        'narcotic_lifetime_freq',
        'inhal_lifetime_freq',
        'mdma_lifetime_freq',
        'pcp_lifetime_freq',
        'dietpill_lifetime_freq']
    drug_cols_yr = ['lsd_yr_freq', 'pysd_yr_freq',
                    'coke_yr_freq', 'amph_yr_freq', 'meth_yr_freq',
                    'sedbarb_yr_freq', 'tranq_yr_freq', 'heroin_yr_freq',
                    'narcotic_yr_freq', 'inhal_yr_freq', 'mdma_yr_freq',
                    'pcp_yr_freq', 'dietpill_yr_freq']
    drug_cols_month = [
        'lsd_month_freq',
        'pysd_month_freq',
        'coke_month_freq',
        'amph_month_freq',
        'meth_month_freq',
        'sedbarb_month_freq',
        'tranq_month_freq',
        'heroin_month_freq',
        'narcotic_month_freq',
        'inhal_month_freq',
        'mdma_month_freq',
        'pcp_month_freq',
        'dietpill_month_freq']
    all_drug_cols = drug_cols_lifetime + drug_cols_yr + drug_cols_month

    drug_users_lifetime = None
    drug_users_yr = None
    drug_users_month = None

    years = [i for i in range(1995, 2020)]
    graph = data
    for col in all_drug_cols:
        conds = [
            ((graph[col] == '0') | (
                graph[col] == 0)), ((graph[col] != '0') & (
                    graph[col] != 0) & (
                    graph[col] != 'Unknown')), (graph[col] == 'Unknown')]
        choices = [0, 1, 'Unknown']

        graph[col] = np.select(conds, choices)

    times = ['lifetime', 'yr', 'month']

    for time in times:

        if time == 'lifetime':
            dc = [
                ((graph['lsd_lifetime_freq'] == '0') & (
                    graph['pysd_lifetime_freq'] == '0') & (
                    graph['coke_lifetime_freq'] == '0') & (
                    graph['amph_lifetime_freq'] == '0') & (
                    graph['meth_lifetime_freq'] == '0') & (
                        graph['sedbarb_lifetime_freq'] == '0') & (
                            graph['tranq_lifetime_freq'] == '0') & (
                                graph['heroin_lifetime_freq'] == '0') & (
                                    graph['narcotic_lifetime_freq'] == '0') & (
                                        graph['inhal_lifetime_freq'] == '0') & (
                                            graph['mdma_lifetime_freq'] == '0') & (
                                                graph['pcp_lifetime_freq'] == '0')),
                ((graph['lsd_lifetime_freq'] == '1') | (
                    graph['pysd_lifetime_freq'] == '1') | (
                    graph['coke_lifetime_freq'] == '1') | (
                    graph['amph_lifetime_freq'] == '1') | (
                    graph['meth_lifetime_freq'] == '1') | (
                    graph['sedbarb_lifetime_freq'] == '1') | (
                    graph['tranq_lifetime_freq'] == '1') | (
                    graph['heroin_lifetime_freq'] == '1') | (
                    graph['narcotic_lifetime_freq'] == '1') | (
                    graph['inhal_lifetime_freq'] == '1') | (
                    graph['mdma_lifetime_freq'] == '1') | (
                    graph['pcp_lifetime_freq'] == '1')),
                ((graph['lsd_lifetime_freq'] == 'Unknown') & (
                    graph['pysd_lifetime_freq'] == 'Unknown') & (
                    graph['coke_lifetime_freq'] == 'Unknown') & (
                    graph['amph_lifetime_freq'] == 'Unknown') & (
                    graph['meth_lifetime_freq'] == 'Unknown') & (
                    graph['sedbarb_lifetime_freq'] == 'Unknown') & (
                    graph['tranq_lifetime_freq'] == 'Unknown') & (
                    graph['heroin_lifetime_freq'] == 'Unknown') & (
                    graph['narcotic_lifetime_freq'] == 'Unknown') & (
                    graph['inhal_lifetime_freq'] == 'Unknown') & (
                    graph['mdma_lifetime_freq'] == 'Unknown') & (
                    graph['pcp_lifetime_freq'] == 'Unknown'))]
            dch = [0, 1, 'Unknown']
            graph['total_drugs_lifetime'] = np.select(dc, dch)
            drug_users_lifetime = [graph[(graph['yr_adminst'] == yr) & (graph['total_drugs_lifetime'] != 'Unknown')]['total_drugs_lifetime'].astype(
                int).sum() / len(graph[(graph['total_drugs_lifetime'] != 'Unknown') & (graph['yr_adminst'] == yr)]) for yr in years]
        elif time == 'yr':
            dc = [
                ((graph['lsd_yr_freq'] == '0') & (
                    graph['pysd_yr_freq'] == '0') & (
                    graph['coke_yr_freq'] == '0') & (
                    graph['amph_yr_freq'] == '0') & (
                    graph['meth_yr_freq'] == '0') & (
                        graph['sedbarb_yr_freq'] == '0') & (
                            graph['tranq_yr_freq'] == '0') & (
                                graph['heroin_yr_freq'] == '0') & (
                                    graph['narcotic_yr_freq'] == '0') & (
                                        graph['inhal_yr_freq'] == '0') & (
                                            graph['mdma_yr_freq'] == '0') & (
                                                graph['pcp_yr_freq'] == '0')),
                ((graph['lsd_yr_freq'] == '1') | (
                    graph['pysd_yr_freq'] == '1') | (
                    graph['coke_yr_freq'] == '1') | (
                    graph['amph_yr_freq'] == '1') | (
                    graph['meth_yr_freq'] == '1') | (
                    graph['sedbarb_yr_freq'] == '1') | (
                    graph['tranq_yr_freq'] == '1') | (
                    graph['heroin_yr_freq'] == '1') | (
                    graph['narcotic_yr_freq'] == '1') | (
                    graph['inhal_yr_freq'] == '1') | (
                    graph['mdma_yr_freq'] == '1') | (
                    graph['pcp_yr_freq'] == '1')),
                ((graph['lsd_yr_freq'] == 'Unknown') & (
                    graph['pysd_yr_freq'] == 'Unknown') & (
                    graph['coke_yr_freq'] == 'Unknown') & (
                    graph['amph_yr_freq'] == 'Unknown') & (
                    graph['meth_yr_freq'] == 'Unknown') & (
                    graph['sedbarb_yr_freq'] == 'Unknown') & (
                    graph['tranq_yr_freq'] == 'Unknown') & (
                    graph['heroin_yr_freq'] == 'Unknown') & (
                    graph['narcotic_yr_freq'] == 'Unknown') & (
                    graph['inhal_yr_freq'] == 'Unknown') & (
                    graph['mdma_yr_freq'] == 'Unknown') & (
                    graph['pcp_yr_freq'] == 'Unknown'))]
            dch = [0, 1, 'Unknown']
            graph['total_drugs_yr'] = np.select(dc, dch)
            drug_users_yr = [graph[(graph['yr_adminst'] == yr) & (graph['total_drugs_yr'] != 'Unknown')]['total_drugs_yr'].astype(
                int).sum() / len(graph[(graph['total_drugs_yr'] != 'Unknown') & (graph['yr_adminst'] == yr)]) for yr in years]
        elif time == 'month':
            dc = [
                ((graph['lsd_month_freq'] == '0') & (
                    graph['pysd_month_freq'] == '0') & (
                    graph['coke_month_freq'] == '0') & (
                    graph['amph_month_freq'] == '0') & (
                    graph['meth_month_freq'] == '0') & (
                        graph['sedbarb_month_freq'] == '0') & (
                            graph['tranq_month_freq'] == '0') & (
                                graph['heroin_month_freq'] == '0') & (
                                    graph['narcotic_month_freq'] == '0') & (
                                        graph['inhal_month_freq'] == '0') & (
                                            graph['mdma_month_freq'] == '0') & (
                                                graph['pcp_month_freq'] == '0')),
                ((graph['lsd_month_freq'] == '1') | (
                    graph['pysd_month_freq'] == '1') | (
                    graph['coke_month_freq'] == '1') | (
                    graph['amph_month_freq'] == '1') | (
                    graph['meth_month_freq'] == '1') | (
                    graph['sedbarb_month_freq'] == '1') | (
                    graph['tranq_month_freq'] == '1') | (
                    graph['heroin_month_freq'] == '1') | (
                    graph['narcotic_month_freq'] == '1') | (
                    graph['inhal_month_freq'] == '1') | (
                    graph['mdma_month_freq'] == '1') | (
                    graph['pcp_month_freq'] == '1')),
                ((graph['lsd_month_freq'] == 'Unknown') & (
                    graph['pysd_month_freq'] == 'Unknown') & (
                    graph['coke_month_freq'] == 'Unknown') & (
                    graph['amph_month_freq'] == 'Unknown') & (
                    graph['meth_month_freq'] == 'Unknown') & (
                    graph['sedbarb_month_freq'] == 'Unknown') & (
                    graph['tranq_month_freq'] == 'Unknown') & (
                    graph['heroin_month_freq'] == 'Unknown') & (
                    graph['narcotic_month_freq'] == 'Unknown') & (
                    graph['inhal_month_freq'] == 'Unknown') & (
                    graph['mdma_month_freq'] == 'Unknown') & (
                    graph['pcp_month_freq'] == 'Unknown'))]
            dch = [0, 1, 'Unknown']
            graph['total_drugs_month'] = np.select(dc, dch)
            drug_users_month = [graph[(graph['yr_adminst'] == yr) & (graph['total_drugs_month'] != 'Unknown')]['total_drugs_month'].astype(
                int).sum() / len(graph[(graph['total_drugs_month'] != 'Unknown') & (graph['yr_adminst'] == yr)]) for yr in years]
        else:
            print("Whoops, something's wrong here.")
    plt.title('Users of Drugs Other than Marijuana: 1995-2019', size=20)
    plt.plot(
        years,
        drug_users_lifetime,
        label='Lifetime',
        color='tab:blue',
        marker='o')
    plt.plot(
        years,
        drug_users_yr,
        label='Past Year',
        color='tab:green',
        marker='o')
    plt.plot(
        years,
        drug_users_month,
        label='Past Month',
        color='tab:red',
        marker='o')
    plt.xlabel('Year of Survey')
    plt.ylabel('Percent of Students')
    plt.legend()


def drug_popularity(data):

    fig, ax = plt.subplots(figsize=(10, 6))

    graph = data

    drug_cols_lifetime = [
        'weed_hash_lifetime_freq',
        'lsd_lifetime_freq',
        'pysd_lifetime_freq',
        'coke_lifetime_freq',
        'amph_lifetime_freq',
        'meth_lifetime_freq',
        'sedbarb_lifetime_freq',
        'tranq_lifetime_freq',
        'heroin_lifetime_freq',
        'narcotic_lifetime_freq',
        'inhal_lifetime_freq',
        'mdma_lifetime_freq',
        'pcp_lifetime_freq']
    for col in drug_cols_lifetime:
        conds = [
            ((graph[col] == '0') | (
                graph[col] == 0)), ((graph[col] != '0') & (
                    graph[col] != 0) & (
                    graph[col] != 'Unknown')), (graph[col] == 'Unknown')]
        choices = [0, 1, 'Unknown']

        graph[col] = np.select(conds, choices)

    drugs = [
        'Marijuana',
        'LSD',
        'Other Hallucinogens',
        'Cocaine',
        'Amphetamines',
        'Meth',
        'Sedatives/Barbituates',
        'Tranquilizers',
        'Heroin',
        'Narcotics',
        'Inhalants',
        'MDMA',
        'PCP']

    usage = pd.DataFrame([graph[graph[drug] != 'Unknown'][drug].astype(int).sum(
    ) / len(graph[graph[drug] != 'Unknown']) for drug in drug_cols_lifetime])
    usage.index = drugs
    usage_sorted = usage.sort_values(0, ascending=False)

    ax.bar(list(usage_sorted.index),
           usage_sorted.values[:, 0], color='forestgreen')
    ax.set_title('Adolescent Drug Usage', size=20)
    ax.set_ylabel('Percent of Students')
    ax.set_xlabel('Drug')
    ax.tick_params(axis='x', labelrotation=90)


def best_correlations(data):
    # Table with highest correlations

    best_corrs = data.stack().reset_index().sort_values(0, ascending=False)

    # zip the variable name columns (Which were only named level_0 and level_1
    # by default) in a new column named "pairs"
    best_corrs['pairs'] = list(zip(best_corrs.level_0, best_corrs.level_1))

    # set index to pairs
    best_corrs.set_index(['pairs'], inplace=True)

    # d rop level columns
    best_corrs.drop(columns=['level_1', 'level_0'], inplace=True)

    # rename correlation column as cc rather than 0
    best_corrs.columns = ['phik']

#     best_corrs.drop_duplicates(inplace=True)

    return best_corrs


def number_of_drugs_used_past_year(data):
    num_students = data['num_drugs_yr'].value_counts().values
    total_students = data['num_drugs_yr'].value_counts().values.sum()
    percentages = [i / total_students for i in num_students]
    num_drugs = data['num_drugs_yr'].value_counts().index

    plt.bar(num_drugs, percentages, color='brown')
    plt.title('Number of Drug Types Used in Past Year', size=20)
    plt.xlabel('Number of Drugs')
    plt.ylabel('Percentage of Students')
    plt.xticks(num_drugs, num_drugs)


def class_distribution(data):
    class_amounts = data['binary_drug'].value_counts().values
    class_label = ['No Drug Use', 'Drug Use']

    plt.bar(class_label, class_amounts, color='cornflowerblue')
    plt.title('Class Distribution', size=20)
    plt.xlabel('Class Label')
    plt.ylabel('Number of Students')
