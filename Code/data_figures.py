import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
mpl.rc('font', family='Times New Roman', size=14)


def plot_figure(data1, data2, label1, label2, legend1, legend2, title, path):
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VMT (Tens of Millions)', color=color)

    mpl.rc('font', size=14)
    ax1.plot(saltlake_week['Day'], saltlake_week[data1], color=color, label=legend1)

    mpl.rc('font', size=18)
    ax1.set_title(title)
    mpl.rc('font', size=14)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='lower left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    mpl.rc('font', size=18)
    ax2.set_ylabel(label2, color=color)  # we already handled the x-label with ax1
    mpl.rc('font', size=14)
    ax2.plot(saltlake_week['Day'], saltlake_week[data2], color=color, label=legend2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper left')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(path)
    plt.show()


saltlake_week = pd.read_csv('Data/saltlake_week.csv')

saltlake_week['Day'] = pd.to_datetime(saltlake_week['Day'])
saltlake_week['Cases'] = pd.to_numeric(saltlake_week['Cases'])
saltlake_week['Vaccinated'] = pd.to_numeric(saltlake_week['Percent_Fully_Vaccinated_12&Older'])*100  # set as percent

plot_figure('VMT (Veh-Miles)', 'Cases', 'VMT (Tens of Millions)', 'Covid Cases', 'VMT', 'Covid Cases',
            'Salt Lake County VMT vs Covid Cases', 'figures/covid_vmt.png')
plot_figure('VMT (Veh-Miles)', 'Vaccinated', 'VMT (Tens of Millions)',
            'Percent Fully Vaccinated\n(5 & Older)', 'VMT', 'Percent Fully Vaccinated',
            'Salt Lake County VMT vs Vaccination Rate', 'figures/vaccinated_vmt.png')
plot_figure('VMT (Veh-Miles)', 'News Sentiment', 'VMT (Tens of Millions)',
            'News Sentiment', 'VMT', 'News Sentiment',
            'Salt Lake County VMT vs News Sentiment', 'figures/sentiment_vmt.png')

# python data_figures.py
