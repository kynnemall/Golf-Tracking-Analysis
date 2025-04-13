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
    with st.expander(f'Combine {source} csv files'):
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


def offline_analysis(df, toptracer, club=None, date=None, side_col=None):
    # subset for the specific club
    sub = df[(df['Club'] == club) & (df['Date'] == date)]
    sub_club = df[df['Club'] == club]

    if side_col is None:
        side_col = 'Offline' if toptracer else 'Side'
    dist_col = 'Distance' if toptracer else 'Total'
    min_ = sub[side_col].min() - 5
    max_ = sub[side_col].max() + 5
    fig = px.scatter(
        sub, x=side_col, y=dist_col,
        title=f'Session {date}, {club} - {side_col} vs {dist_col}'
    )

    # plot the data
    colors = ('red', 'orange', 'green', 'orange', 'red')
    x_pairs = (
        (min_, -20), (-20, -5), (-5, 5), (5, 20), (20, max_)
    )

    club_min = sub_club[side_col].min() - 5
    club_max = sub_club[side_col].max() + 5
    club_x_pairs = (
        (club_min, -20), (-20, -5), (-5, 5), (5, 20), (20, club_max)
    )

    for color, (x0, x1) in zip(colors, x_pairs):
        percent = ((sub[side_col] > x0) & (sub[side_col] <= x1)).mean()
        if percent > 0:
            text = f'{percent:.1%}'
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor=color, opacity=0.5, line_width=0,
                layer="below",
                annotation=dict(
                    text=text,
                    x=(x0 + x1) / 2,
                    xanchor="center",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color='black')
                )
            )
    fig.update_traces(marker=dict(size=8,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    ymin = sub[dist_col].min() * 0.95
    ymax = sub[dist_col].max() * 1.08
    fig.update_layout(yaxis_range=[ymin, ymax])

    with st.expander(f'{side_col.title()} Analysis'):
        st.plotly_chart(fig)
        if df['Date'].nunique() > 1:
            trend_fig = _offline_time_trend(sub_club, club_x_pairs, side_col)
            st.plotly_chart(trend_fig)


def _offline_time_trend(df, x_pairs, side_col):
    ds = []
    for date in df['Date'].unique():
        sub = df[df['Date'] == date]
        percents = []
        for (x0, x1) in x_pairs:
            percent = ((sub[side_col] > x0) & (sub[side_col] <= x1)).mean()
            percents.append(round(percent * 100, 1))
        red = percents[0] + percents[-1]
        orange = percents[1] + percents[-2]
        green = percents[2]
        for val, name in zip((green, orange, red), ('Green', 'Orange', 'Red')):
            ds.append([date, val, name])
    ds = pd.DataFrame(ds, columns=['Date', 'Percent', 'Color'])

    # format the graph
    cmap = {col: col for col in ds['Color'].unique()}
    fig = px.area(ds, x='Date', y='Percent', color='Color',
                  color_discrete_map=cmap, title=f'{side_col} over time')
    return fig


def speed_vs_launch(df, club=None, date=None):
    sub = df[(df['Club'] == club) & (df['Date'] == date)]
    fig = px.scatter(sub, x='Launch Angle', y='Ball Speed', color='Distance',
                     title=f'Session {date}, {club}')
    fig.update_traces(
        dict(marker_line_width=1, marker_line_color="DarkSlateGrey"))

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
    groups = ['Club', 'Date']
    medoids = df.groupby(groups).apply(
        get_medoid)
    medoids['# Shots'] = df.groupby(groups)['Carry'].count()
    medoids.reset_index(drop=True, inplace=True)
    groups.append('# Shots')
    order = groups + [c for c in medoids.columns if c not in groups]
    medoids = medoids[order].sort_values(groups)

    with st.expander('Most Representative Shot by Club and Date'):
        st.dataframe(medoids)
