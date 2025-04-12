#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:04:21 2025

@author: martin
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime
from toptracer_cv import Toptracer
from sklearn.metrics import pairwise_distances


def combine_data(source='Toptracer'):
    with st.expander(f'Upload personal {source} data'):
        files = st.file_uploader(
            'Upload files created by this app',
            accept_multiple_files=True, key=f'{source}_csvs'
        )
        if files:
            df = pd.concat([pd.read_csv(f) for f in files])
            st.text('Successfully loaded and combined data')

            df.to_csv('data.csv', index=False)
            if os.path.exists('data.csv'):
                fname = datetime.now().strftime("%Y-%m-%d %H-%M combined.csv")
                with open('data.csv', "rb") as f:
                    st.download_button(f"Download {source} data",
                                       data=f, file_name=fname)


def extract_from_screenshots(source='Toptracer'):
    with st.expander(f'Extract {source} data'):
        with st.form(f'Extract {source} data'):

            files = st.file_uploader(
                'Upload screenshots of {source} data', accept_multiple_files=True,
                key=f'{source}_screenshots'
            )
            club_name = st.text_input(
                'Enter a name for the club in the screenshots')
            date = st.date_input(
                'Select the date when these data were collected')
            submitted = st.form_submit_button("Submit")
            if submitted:
                top = Toptracer(files)
                df = top.df
                df['Club'] = club_name
                df['Date'] = date

                st.text('Data successfully extracted!')
                st.text('You can analyse the data in the other tabs')

                # save the file for downloading
                df.to_csv('data.csv', index=False)

        if os.path.exists('data.csv'):
            fname = datetime.now().strftime(f"%Y-%m-%d %H-%M {club_name}.csv")
            with open('data.csv', "rb") as f:
                st.download_button(f"Download {source} data",
                                   data=f, file_name=fname, key=source)


def offline_analysis(df, toptracer, club=None):
    # subset for the specific club
    multiple_clubs = df['Club'].nunique() > 1
    if multiple_clubs and club is not None:
        sub = df[df['Club'] == club]
    else:
        club = df['Club'].values[0]
        sub = df[df['Club'] == club]

    side_col = 'Offline' if toptracer else 'Side'
    dist_col = 'Distance' if toptracer else 'Total'
    min_ = sub[side_col].min() - 5
    max_ = sub[side_col].max() + 5
    fig = px.scatter(sub, x=side_col, y=dist_col,
                     title=f'{club} - {side_col} vs {dist_col}')

    # plot the data
    colors = ('red', 'orange', 'green', 'orange', 'red')
    x_pairs = ((min_, -20), (-20, -5), (-5, 5), (5, 20), (20, max_))

    for color, (x0, x1) in zip(colors, x_pairs):

        percent = ((sub[side_col] > x0) & (sub[side_col] <= x1)).mean()
        text = f'{percent:.1%}'
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=color, opacity=0.3, line_width=0,
            layer="below",
            annotation=dict(
                text=text,
                x=(x0 + x1) / 2,
                xanchor="center",
                yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
        )
    fig.update_traces(marker=dict(size=8,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    with st.expander(f'{side_col.title()} Analysis'):
        st.plotly_chart(fig)


def speed_vs_launch(df, club=None):
    # subset for the specific club
    multiple_clubs = df['Club'].nunique() > 1
    if multiple_clubs and club is not None:
        sub = df[df['Club'] == club]
    else:
        club = df['Club'].values[0]
        sub = df[df['Club'] == club]

    fig = px.scatter(sub, x='Launch Angle', y='Ball Speed')

    with st.expander('Launch Angle vs Ball Speed'):
        st.plotly_chart(fig)


def get_medoid(group_df):
    features = group_df.iloc[:, :-2].to_numpy()
    dists = pairwise_distances(features)
    medoid_idx = np.argmin(dists.sum(axis=0))
    result = group_df.iloc[medoid_idx]
    result.name = ''
    return result


def most_common_shot(df):
    medoids = df.groupby(['Club', 'Date']).apply(
        get_medoid).reset_index(drop=True)
    order = df.columns[-2:].tolist() + df.columns[:-2].tolist()
    medoids = medoids[order].sort_values(['Club', 'Date'])

    with st.expander('Most Representative Shot by Club and Date'):
        st.dataframe(medoids)
