import pandas as pd
import get_sql
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from data_cleaning import DataCleaning

current_time = datetime.now().strftime(format='%m-%d-%y-%H-%M')
dc = DataCleaning()
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
    graph_filename = "../graphs/paty_"+current_time+".png"
    plt.savefig(graph_filename)

authors_graphs()
