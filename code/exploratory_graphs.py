import pandas as pd
import get_sql
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from data_prep import DataPrep

current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
dc = DataPrep(filepath='../data/intro_data_5_18_2.csv')
df = dc.df

def bills_graphs():
    df['vote_required'] = df['vote_required'].apply(lambda vote: 'Majority' if vote=='Majority' else 'More than Majority')
    df['session_type'] = df['session_num'].apply(lambda session: 'Normal' if session=='0' else 'Extraordinary')
    plot_urgency(df)
    plot_appropriation(df)
    plot_vote_required(df)
    plot_taxlevy(df)
    plot_fiscal_committee(df)
    plot_house(df)
    plot_session_type(df)
    plot_days_hist(df)
    plot_days_comparison_hist(df)
    plot_seniority(df)

def plot_seniority(df):
    no_committees = df[df.party!='COM']
    avg_terms_passed = no_committees[['passed', 'nterms']].groupby('passed').mean()
    no_committees['seniority_bucket'] = pd.cut(no_committees.nterms, bins=[0,1,2,3,4,5,6,21])
    bucket_pass_rates = no_committees[['seniority_bucket', 'passed']].groupby('seniority_bucket').agg(['mean', 'count']).reset_index()
    bucket_pass_rates.plot(x='seniority_bucket', y=0, kind='bar', legend=False, title='Percent passed by average seniority of cosponsors')
    graph_filename = "../graphs/seniority_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_urgency(df):
    pd.crosstab(df.urgency, df.passed, normalize='index').reset_index().plot(x='urgency',
        y=1, kind='bar', title='Percent passed for urgency', legend=False)
    graph_filename = "../graphs/urgency_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_appropriation(df):
    pd.crosstab(df.appropriation, df.passed, normalize='index').reset_index().plot(x='appropriation',
        y=1, kind='bar', title='Percent passed for appropriation', legend=False)
    graph_filename = "../graphs/appropriation_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_vote_required(df):
    ax = pd.crosstab(df.vote_required, df.passed, normalize='index').reset_index().plot(x='vote_required',
        y=1, kind='bar', title='Percent passed based on vote required', legend=False)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)
    ax.plot()
    graph_filename = "../graphs/vote_required_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_taxlevy(df):
    pd.crosstab(df.taxlevy, df.passed, normalize='index').reset_index().plot(x='taxlevy',
        y=1, kind='bar', title='Percent passed for tax levies', legend=False)
    graph_filename = "../graphs/taxlevy_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_fiscal_committee(df):
    pd.crosstab(df.fiscal_committee, df.passed, normalize='index').reset_index().plot(x='fiscal_committee',
        y=1, kind='bar', title='Percent passed for fiscal committee', legend=False)
    graph_filename = "../graphs/fiscal_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_house(df):
    pd.crosstab(df.measure_type, df.passed, normalize='index').reset_index().plot(x='measure_type',
        y=1, kind='bar', title='Percent passed by house', legend=False)
    graph_filename = "../graphs/house_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_session_type(df):
    ax = pd.crosstab(df.session_type, df.passed, normalize='index').reset_index().plot(x='session_type',
        y=1, kind='bar', title='Percent passed by session type', legend=False)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)
    ax.plot()
    graph_filename = "../graphs/session_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_days_hist(df):
    df.plot(y='days_since_start',
        kind='hist', title='No. bills introduced by number of days since start of session')
    graph_filename = "../graphs/days_hist_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_days_comparison_hist(df):
    bins = np.arange(801, step=10)
    fig, ax_list = plt.subplots(2,1)
    ax_list[0].hist(df[df.passed==1].dropna()['days_since_start'], bins=bins, normed=True)
    ax_list[0].set_title("Passed")
    ax_list[1].hist(df[df.passed==0].dropna()['days_since_start'], bins=bins, normed=True)
    ax_list[1].set_title("Not passed")
    graph_filename = "../graphs/comp_days_hist_"+current_time+".png"
    plt.savefig(graph_filename)

def authors_graphs():
    plot_cosponsor_boxplot(df)
    plot_parties(df)

def plot_cosponsor_boxplot(df):
    data = [df[df.passed==0]['n_authors'].values, df[df.passed==1]['n_authors'].values]
    labels = ["Not passed", "Passed"]
    fig, ax = plt.subplots(1,1)
    ax.boxplot(data, labels=labels)
    ax.set_title("No. cosponsors for passed and not passed bills")
    graph_filename = "../graphs/cosponsor_boxplot_"+current_time+".png"
    plt.savefig(graph_filename)

def plot_parties(df):
    ax = pd.crosstab(df.party, df.passed, normalize='index').reset_index().plot(x='party',
        y=1, kind='bar', title='Percent passed by party', legend=False)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=0)
    ax.plot()
    graph_filename = "../graphs/party_"+current_time+".png"
    plt.savefig(graph_filename)

def bill_version_graphs():
    query = """select b.bill_id, count(bill_version_id) as n_versions, passed from bill_version_tbl bv
        join bill_tbl b on b.bill_id=bv.bill_id
        group by b.bill_id"""
    df = get_sql.get_df(query)
    plot_bill_version_hist(df)
    plot_bill_version_comp(df)

def plot_bill_version_hist(df):
    bins = np.arange(16, step=1)
    df.plot(y='n_versions', kind='hist', bins=bins, title='Number of versions for all bills')
    graph_filename = "../graphs/versions_hist_all"+current_time+".png"
    plt.savefig(graph_filename)

def plot_bill_version_comp(df):
    bins = np.arange(16, step=1)
    ylim = [0, .5]
    fig, ax_list = plt.subplots(2,1)
    ax_list[0].hist(df[df.passed==1].dropna()['n_versions'], bins=bins, normed=True)
    ax_list[0].set_title("Passed")
    ax_list[0].set_ylim(ylim)
    ax_list[1].hist(df[df.passed==0].dropna()['n_versions'], bins=bins, normed=True)
    ax_list[1].set_title("Not passed")
    ax_list[1].set_ylim(ylim)
    graph_filename = "../graphs/comp_versions_hist_"+current_time+".png"
    plt.savefig(graph_filename)

def bill_amendment_graphs():
    query = """select bv1.bill_version_id, bv1.bill_id, count(bv2.bill_version_id) as n_prev_versions, b.passed from bill_version_tbl bv1
        join bill_version_tbl bv2 on bv1.bill_id=bv2.bill_id
        join bill_tbl b on bv1.bill_id=b.bill_id
        where bv1.bill_version_id like '%AMD' and bv2.version_num > bv1.version_num
        group by bv1.bill_version_id"""
    amendment_df = get_sql.get_df(query)
    plot_n_amendments(amendment_df)

def plot_n_amendments(df):
    pd.crosstab(df.n_prev_versions, df.passed, normalize='index').reset_index().plot(x='n_prev_versions',
        y=1, kind='bar', title='Number amendments vs chance of passing', legend=False)
    graph_filename = "../graphs/n_amendments_"+current_time+".png"
    plt.savefig(graph_filename)

bills_graphs()
