# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:26:04 2025

@author: 63517
"""
import streamlit as st
import akshare as ak
import plotly.express as px
macro_cnbs_df = ak.macro_cnbs()
st.write('杠杆率')
st.dataframe(macro_cnbs_df)


fig = px.line(data_frame=macro_cnbs_df, 
              title='宏观杠杆率变化',
              x='年份', y=['实体经济部门','居民部门','非金融企业部门',
                         '政府部门'],
              labels={
                  'value':'杠杆率', # value是默认的y标签
                  'variable':'部门', # 图例标签
                  '年份':'年份', # X轴标签
                  }
              )
st.plotly_chart(fig)
st.markdown('''
            ### 📌
            - 整体杠杆率呈持续上升趋势，反映出宏观债务水平的持续扩大。
            - 自 2020 年第三季度起，居民部门杠杆率趋于稳定，维持在 60% 至 62% 区间。
            - 2020 年第三季度后，政府与非金融企业部门杠杆率同步上行，是推动总体杠杆率上升的主要因素。
            ''')
