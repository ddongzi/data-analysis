# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:26:04 2025

@author: 63517
"""
import pandas as pd
import streamlit as st
import akshare as ak
import plotly.express as px
macro_cnbs_df = ak.macro_cnbs()
st.write('## 杠杆率')
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


st.write('## 企业商品价格指数')
macro_china_qyspjg_df = ak.macro_china_qyspjg()
st.dataframe(macro_china_qyspjg_df)

st.write('## 城镇调查失业率')
macro_china_urban_unemployment_df = ak.macro_china_urban_unemployment()
macro_china_urban_unemployment_df_wide = macro_china_urban_unemployment_df.pivot(index='date',columns='item',values='value')
macro_china_urban_unemployment_df_wide.index = pd.to_datetime(macro_china_urban_unemployment_df_wide.index, format='%Y%m')
fig = px.line(data_frame=macro_china_urban_unemployment_df_wide,
              x=macro_china_urban_unemployment_df_wide.index,
              y=macro_china_urban_unemployment_df_wide.columns,
              labels={
                  'item':'项目',
                  'value':'失业率',
                  'date':'日期'
                  },
              title='失业率统计（按月份）'
              )
st.plotly_chart(fig)
st.markdown('''
            ### 📌
            - 国家统计局的统计数据不知是否可靠。且部分数据缺失。
            - 城镇失业率大概在5%-6%之间。
            ''')
            
st.write('## LPR')    
macro_china_lpr_df = ak.macro_china_lpr()
macro_china_lpr_df.set_index('TRADE_DATE', inplace=True)
fig = px.line(data_frame=macro_china_lpr_df,
              title="中国LPR利率变化趋势",  # 图标题
              y=['LPR1Y', 'LPR5Y'],
              labels={
                "TRADE_DATE": "日期",
                "value": "LPR",
                "variable": "项目"
                }
        )
st.plotly_chart(fig)
st.markdown('''
            ### 📌
            - LPR 于 2019 年 8 月正式推出，因此早期数据中 LPR 列为空（NaN）属正常情况。
            - 当前（最新）**1年期 LPR 为 3.5, 5年期LPR为3.0**，处于历史最低位。
            - 自推出以来，**LPR 呈下降趋势**：三年内下调约 **0.8 个百分点**，五年内累计下调约 **1.0 个百分点**。表明贷款难度下降，但消费力长期不足。
            - **1年期与5年期 LPR 的利差逐步收窄**，从最初的 0.65 降至当前的 0.4，表明长期贷款成本相对下降，**货币政策更偏向刺激中长期信贷**，也可能反映房地产、基建等长期资金需求较为疲软。
            ''')
            
st.write('## 社会融资规模')
st.write(' 反映每个月中国实体经济从各渠道获得的新增资金总量以及结构') 
macro_china_shrzgm_df = ak.macro_china_shrzgm()
macro_china_shrzgm_df['月份'] = pd.to_datetime(macro_china_shrzgm_df['月份'], format='%Y%m')
macro_china_shrzgm_df.set_index('月份', inplace=True)
macro_china_shrzgm_df.columns = [ col.replace('其中-','') for col in macro_china_shrzgm_df.columns]

fig = px.line(data_frame=macro_china_shrzgm_df,
              x = macro_china_shrzgm_df.index,
              y = macro_china_shrzgm_df.columns,
              title='社融增量统计',
              labels={
                  'value':'人名币(亿元)',
                  'variable':'融资来源',
                  '月份':'日期'
                  }
              )
fig.update_layout(
    xaxis = { # 对X数据进行范围选择
        'type':'date', # 时间选择器
        'rangeselector': { # 选择器
            'buttons': [ # 几个选择按钮
                {
                    'count': 1, # 一个单位
                    'label': '1Year', # 按钮文字
                    'step':'year', # 步长单位：month,year,day
                    'stepmode':'backward', # 方向。backward表示过去一年，todate表示今年以来(基于X最大年份)
                },
                {
                    'label':'ALL',
                    'step':'all',
                }
            ]
        },
    }
)
st.plotly_chart(fig)
st.markdown('''
            ### 📌
            - 社融增量主要来自人名币贷款。
            - 每年的趋势大致相同，一月份是社融增量高峰。这体现了政策、银行、企业贷款需求的周期性。
            ''')
