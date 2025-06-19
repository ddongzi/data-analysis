# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 00:34:00 2025

@author: 63517
"""

import streamlit as st
x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)


