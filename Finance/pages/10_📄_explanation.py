# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 01:15:12 2025

@author: 63517
"""
import streamlit as st

content = ''
with open('explanation.md','r',encoding='utf-8') as f:
    content = f.read()
    
st.markdown(content)

