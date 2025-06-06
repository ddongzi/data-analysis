# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 00:48:40 2025

@author: 63517
"""


import streamlit as st
import pandas as pd
import numpy as np

import akshare as ak
from datetime import date

stock_sse_summary_df = ak.stock_sse_summary() 
summary_date = stock_sse_summary_df.iloc[6]['股票']
st.write(f'上证交易所股票总貌{summary_date}')
st.dataframe(stock_sse_summary_df)

stock_szse_summary_df = ak.stock_szse_summary(summary_date) 
st.write(f'深圳交易所股票总貌{summary_date}')
st.dataframe(stock_szse_summary_df)
