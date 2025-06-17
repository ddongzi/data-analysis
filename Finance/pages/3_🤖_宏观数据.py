# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:26:04 2025

@author: 63517
"""
import pandas as pd
import streamlit as st
import akshare as ak
import plotly.express as px

### 数据 ？？？ 怎么把akshare的数据 都返回为缓存数据



######### 
st.write('## 杠杆率')
macro_cnbs_df = ak.macro_cnbs()
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
######################
st.write('## 国民经济情况')
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
st.write(' 反映每个月中国实体经济从各渠道获得的新增资金总量以及结构')
st.markdown('''
            ### 📌
            - 社融增量主要来自人名币贷款。
            - 每年的趋势大致相同，一月份是社融增量高峰。这体现了政策、银行、企业贷款需求的周期性。
            ''')

macro_china_gdp_yearly_df = ak.macro_china_gdp_yearly()
fig = px.line(data_frame=macro_china_gdp_yearly_df,
              x='日期',
              y='今值',
              title='GDP年率报告(季度)',
              labels={
                  '今值':'GDP年同比增长率(%)',
                  'variable':'项目',
                  '日期':'季度'
                  }
              )
st.plotly_chart(fig)

st.markdown('''
            ### 📌
            - 自 2011 年以来，中国 GDP 年增速呈持续放缓趋势，从 9.5% 降至约 5%，反映出经济由高速增长阶段逐步迈向高质量、稳态发展阶段。
            - 2020 年一季度，受新冠疫情严重冲击，GDP 增速首次转负，达到 -6.5%。此后虽逐步恢复，到 2023 年三季度，增速才趋于稳定。
            - 然而，增速“稳定”并不意味着 GDP 总量已完全恢复至疫情前的自然增长轨迹。🔍增速仅反映相对变化率，要评估真实经济恢复情况，应结合GDP总量综合判断。
            ''')
            
macro_china_cpi_monthly_df = ak.macro_china_cpi_monthly()
fig = px.line(data_frame=macro_china_cpi_monthly_df,
              x='日期',
              y= ['今值','预测值'],
              title='中国CPI月率',
              labels={
                  'value':'CPI月环比增长率(%)',
                  'variable':'项目',
                  '日期':'月度'
                  }
              )
st.plotly_chart(fig)  
st.markdown('''
            ### 📌
            - 长期来看，CPI波动幅度逐渐减小，消费物价趋于稳定，因此CPI更适合反映短期几个月内的价格变动。
            - 2025年3、4、6月CPI同比连续为负，尽管5月小幅回升至0.1%，但反弹力度有限，显示整体消费需求偏弱。
            - 若CPI持续负增长，将可能引发通缩风险，其传导机制包括：
              1. 供大于求 → 商品价格下行 → 企业销售困难、盈利下降 → 减少投资、裁员减产；
              2. 企业经营压力传导至居民 → 收入下降 → 消费进一步收缩，形成恶性循环；
              3. 通缩期间货币购买力上升 → 债务实际负担加重 → 企业、个人违约风险上升，可能引发信用危机与破产潮。
            - 当前LPR（贷款市场报价利率）呈下降趋势，反映出政府正在通过信贷宽松和低利率政策提振消费和投资，缓解通缩压力。
            ''')       
            
macro_china_ppi_yearly_df = ak.macro_china_ppi_yearly()
fig = px.line(data_frame=macro_china_cpi_monthly_df,
              x='日期',
              y= '今值',
              title='中国PPI年率',
              labels={
                  '今值':'PPI月环比增长率(%)',
                  'variable':'项目',
                  '日期':'月度'
                  }
              )
st.plotly_chart(fig) 
st.markdown('''
            ### 📌
            - 长期来看，PPI波动幅度逐渐减小，生产端趋于稳定，因此CPI更适合反映短期几个月内的价格变动。
            - 2025以来，和CPI同步。
 ''')       

######################### 
st.write('## 金融指标')
macro_china_fx_reserves_yearly_df = ak.macro_china_fx_reserves_yearly()
macro_china_fx_reserves_yearly_df['日期'] = pd.to_datetime(macro_china_fx_reserves_yearly_df['日期'])
macro_china_fx_reserves_yearly_df['year'] =  macro_china_fx_reserves_yearly_df['日期'].dt.year
macro_china_fx_reserves_yearly_df['month'] =  macro_china_fx_reserves_yearly_df['日期'].dt.month
plotdf = macro_china_fx_reserves_yearly_df.groupby(['year','month'])['今值'].sum()
plotdf = plotdf.reset_index()
plotdf['日期'] = pd.to_datetime({
    'year':plotdf['year'],
    'month':plotdf['month'],
    'day':1, # 第一天默认
    })
fig = px.line(data_frame=macro_china_fx_reserves_yearly_df,
              x='日期',
              y= '今值',
              title='中国外汇储备',
              labels={
                  '今值':'外汇储备(单位：亿美元)',
                  'variable':'项目',
                  '日期':'月度'
                  }
              )
st.plotly_chart(fig) 
st.markdown('''
        ### 📌
        - 最新外汇储备 32850 亿美元

 ''')       
 
macro_china_m2_yearly_df = ak.macro_china_m2_yearly()
macro_china_m2_yearly_df['日期'] = pd.to_datetime(macro_china_m2_yearly_df['日期'])
macro_china_m2_yearly_df['year'] =  macro_china_m2_yearly_df['日期'].dt.year
macro_china_m2_yearly_df['month'] =  macro_china_m2_yearly_df['日期'].dt.month
plotdf = macro_china_m2_yearly_df.groupby(['year','month'])['今值'].sum()
plotdf = plotdf.reset_index()
plotdf['日期'] = pd.to_datetime({
    'year':plotdf['year'],
    'month':plotdf['month'],
    'day':1, # 第一天默认
    })

fig = px.line(data_frame=plotdf,
              x='日期',
              y= '今值',
              title='中国M2货币供应',
              labels={
                  '今值':'同比增长率%',
                  'variable':'项目',
                  }
              )
st.plotly_chart(fig) 
st.markdown('''
        ### 📌 
        - 5月的M2货币同比增长7.9%
        ### ✅ 带加入后面的货币供应量数据一起
 ''')       

# 城市
cities = ['上海', '北京', '成都']
house_type = st.selectbox('选择房屋类型', ['新建商品住宅', '二手住宅'])
city1 = st.selectbox('城市1',  cities)
city2 = st.selectbox('城市2',  cities)
method = st.selectbox('方式', ['环比', '同比', '定基'])
compos_colname = f'{house_type}价格指数-{method}'
macro_china_new_house_price_df = ak.macro_china_new_house_price(city_first=city1, 
                                                                city_second=city2)
fig = px.line(data_frame=macro_china_new_house_price_df,
              x='日期',
              y= compos_colname,
              color='城市',
              title=f'{city1} vs {city2} 新房价价格指数',
              labels={
                  '今值':f'{method}增长率%',
                  'variable':'项目',
                  }
              )

st.plotly_chart(fig) 

