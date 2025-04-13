#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 09:05:00 2025

@author: martin
"""

import os
import pandas as pd
import streamlit as st
from ui_funcs import combine_data, extract_from_screenshots, offline_analysis, \
    speed_vs_launch, most_common_shot

if os.path.exists('data.csv'):
    df = pd.read_csv('data.csv')
    top_ds = 'Offline' in df.columns
else:
    df = None
    top_ds = None

loader, single, multi = st.tabs(['Process Screenshots', 'Single Club Analysis',
                                 'Club Comparisons'])

with loader:
    st.text(
        """
        It's recommended to upload screenshots from one club at a time.
        
        You can also upload multiple data files generated by this app to
        combine into one big file for later analysis.
        """
    )
    combine_data('Toptracer')
    combine_data('Trackman')
    extract_from_screenshots('Toptracer')
    extract_from_screenshots('Trackman')


with single:
    if df is None:
        st.text(
            """
            No dataset in memory. Please upload 
            screenshots or a data file to proceed"""
        )
    else:
        st.text('Detected uploaded data')
        club_options = df['Club'].unique()
        club = st.selectbox('Select a club', club_options)

        dates = df['Date'].unique()
        date = st.selectbox('Select a Date', dates)
        offline_analysis(df, top_ds, club, date)
        offline_analysis(df, top_ds, club, date, side_col='Curve')
        speed_vs_launch(df, club, date)

with multi:
    if df is None:
        st.text(
            """
            No dataset in memory. Please upload 
            screenshots or a data file to proceed"""
        )
    else:
        st.text('Detected uploaded data')
        most_common_shot(df)
