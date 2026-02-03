# %%
import pandas as pd
import akshare as ak
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
pio.templates.default = "plotly_dark"
pio.templates.default = "plotly_white"


# %% [markdown]
# # 宏观杠杆率

# %% [markdown]
# ## 杠杆率

# %%
macro_cnbs_df = ak.macro_cnbs()

# %%
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
fig.show()

# %% [markdown]
# 📌
# - 整体杠杆率呈持续上升趋势，反映出宏观债务水平的持续扩大。
# - 自 2020 年第三季度起，居民部门杠杆率趋于稳定，维持在 60% 至 62% 区间。
# - 2020 年第三季度后，政府与非金融企业部门杠杆率同步上行，是推动总体杠杆率上升的主要因素。

# %% [markdown]
# # 国民经济运行情况

# %% [markdown]
# ## 企业商品价格指数

# %%
macro_china_qyspjg_df = ak.macro_china_qyspjg()

# %%
macro_china_qyspjg_df.head()

# %% [markdown]
# ## 城镇调查失业率

# %%
macro_china_urban_unemployment_df = ak.macro_china_urban_unemployment()
macro_china_urban_unemployment_df_wide = macro_china_urban_unemployment_df.pivot(index='date',columns='item',values='value')
macro_china_urban_unemployment_df_wide.index = pd.to_datetime(macro_china_urban_unemployment_df_wide.index, format='%Y%m')

# %%
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
fig.show()

# %% [markdown]
# 📌
# - 国家统计局的统计数据不知是否可靠。且部分数据缺失。
# - 城镇失业率大概在5%-6%之间。

# %% [markdown]
# ## LPR品种数据

# %%
macro_china_lpr_df = ak.macro_china_lpr()
macro_china_lpr_df.set_index('TRADE_DATE', inplace=True)

# %%
fig = px.line(data_frame=macro_china_lpr_df,
              title="中国LPR利率变化趋势",  # 图标题
              y=['LPR1Y', 'LPR5Y'],
              labels={
                "TRADE_DATE": "日期",
                "value": "LPR",
                "variable": "项目"
                }
        )
fig.show()

# %% [markdown]
# 📌
# - LPR 于 2019 年 8 月正式推出，因此早期数据中 LPR 列为空（NaN）属正常情况。
# - 当前（最新）**1年期 LPR 为 3.5, 5年期LPR为3.0**，处于历史最低位。
# - 自推出以来，**LPR 呈下降趋势**：三年内下调约 **0.8 个百分点**，五年内累计下调约 **1.0 个百分点**。表明贷款难度下降，但消费力长期不足。
# - **1年期与5年期 LPR 的利差逐步收窄**，从最初的 0.65 降至当前的 0.4，表明长期贷款成本相对下降，**货币政策更偏向刺激中长期信贷**，也可能反映房地产、基建等长期资金需求较为疲软。
# 

# %% [markdown]
# ## 社融增量统计

# %%
macro_china_shrzgm_df = ak.macro_china_shrzgm()
macro_china_shrzgm_df['月份'] = pd.to_datetime(macro_china_shrzgm_df['月份'], format='%Y%m')
macro_china_shrzgm_df.set_index('月份', inplace=True)
macro_china_shrzgm_df.columns = [ col.replace('其中-','') for col in macro_china_shrzgm_df.columns]

# %%
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
fig.show()

# %% [markdown]
# 📌 反映每个月中国实体经济从各渠道获得的新增资金总量以及结构
# - 社融增量主要来自人名币贷款。
# - 每年的趋势大致相同，一月份是社融增量高峰。这体现了政策、银行、企业贷款需求的周期性。

# %% [markdown]
# ## GDP年率

# %%
macro_china_gdp_yearly_df = ak.macro_china_gdp_yearly()

# %%
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
fig.show()

# %% [markdown]
# 📌
# - 自 2011 年以来，中国 GDP 年增速呈持续放缓趋势，从 9.5% 降至约 5%，反映出经济由高速增长阶段逐步迈向高质量、稳态发展阶段。
# - 2020 年一季度，受新冠疫情严重冲击，GDP 增速首次转负，达到 -6.5%。此后虽逐步恢复，到 2023 年三季度，增速才趋于稳定。
# - 然而，增速“稳定”并不意味着 GDP 总量已完全恢复至疫情前的自然增长轨迹。🔍增速仅反映相对变化率，要评估真实经济恢复情况，应结合GDP总量综合判断。
# 

# %% [markdown]
# ## CPI

# %%
macro_china_cpi_monthly_df = ak.macro_china_cpi_monthly()

# %%
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
fig.show()

# %% [markdown]
# 📌
# - 长期来看，CPI波动幅度逐渐减小，消费物价趋于稳定，因此CPI更适合反映短期几个月内的价格变动。
# - 2025年3、4、6月CPI同比连续为负，尽管5月小幅回升至0.1%，但反弹力度有限，显示整体消费需求偏弱。
# - 若CPI持续负增长，将可能引发通缩风险，其传导机制包括：
#     1. 供大于求 → 商品价格下行 → 企业销售困难、盈利下降 → 减少投资、裁员减产；
#     2. 企业经营压力传导至居民 → 收入下降 → 消费进一步收缩，形成恶性循环；
#     3. 通缩期间货币购买力上升 → 债务实际负担加重 → 企业、个人违约风险上升，可能引发信用危机与破产潮。
# - 当前LPR（贷款市场报价利率）呈下降趋势，反映出政府正在通过信贷宽松和低利率政策提振消费和投资，缓解通缩压力。
# 

# %% [markdown]
# ## PPI

# %%
macro_china_ppi_yearly_df = ak.macro_china_ppi_yearly()

# %%
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
fig.show()

# %% [markdown]
# 📌
# - 长期来看，PPI波动幅度逐渐减小，生产端趋于稳定，因此CPI更适合反映短期几个月内的价格变动。
# - 2025以来，和CPI同步。

# %% [markdown]
# # 金融指标

# %% [markdown]
# ## 外汇储备

# %%
macro_china_fx_reserves_yearly_df = ak.macro_china_fx_reserves_yearly()
macro_china_fx_reserves_yearly_df['日期'] = pd.to_datetime(macro_china_fx_reserves_yearly_df['日期'])
macro_china_fx_reserves_yearly_df['year'] =  macro_china_fx_reserves_yearly_df['日期'].dt.year
macro_china_fx_reserves_yearly_df['month'] =  macro_china_fx_reserves_yearly_df['日期'].dt.month

# %%
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
fig.show()

# %% [markdown]
# 📌
# - 最新外汇储备 **32850** 亿美元

# %% [markdown]
# ## M2货币供应量

# %%
macro_china_m2_yearly_df = ak.macro_china_m2_yearly()
macro_china_m2_yearly_df['日期'] = pd.to_datetime(macro_china_m2_yearly_df['日期'])
macro_china_m2_yearly_df['year'] =  macro_china_m2_yearly_df['日期'].dt.year
macro_china_m2_yearly_df['month'] =  macro_china_m2_yearly_df['日期'].dt.month
plotdf = macro_china_m2_yearly_df.groupby(['year','month'])['今值'].sum()

# %%
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
fig.show()

# %% [markdown]
# 📌 
# - 5月的M2货币同比增长7.9%  
# - ✅ 带加入后面的货币供应量数据一起

# %% [markdown]
# ##  新房价价格指数

# %%
city1 = '上海'
city2 = '成都'
macro_china_new_house_price_df = ak.macro_china_new_house_price(city_first=city1, 
                                                                city_second=city2)

# %%
# 把指标扁平化作为 合并一列，
plotdf = pd.melt(
    macro_china_new_house_price_df,
    id_vars=['日期', '城市'],
    value_vars=[
        '新建商品住宅价格指数-同比', 
        '新建商品住宅价格指数-环比', 
        '新建商品住宅价格指数-定基',
        '二手住宅价格指数-同比', 
        '二手住宅价格指数-环比', 
        '二手住宅价格指数-定基'
    ],
    var_name='项目',
    value_name='增长率%'
)


# %%
fig = px.line(
    data_frame=plotdf,
    x='日期',
    y='增长率%',
    color='项目',        # 关键在这里：颜色按“项目”来区分！
    line_dash='城市',    # 如果你想同时区分城市，可以加这个
    title=f'{city1} vs {city2} 房价指数对比',
    labels={
        '增长率%': '增长率 %',
        '项目': '价格指数类型',
        '城市': '城市',
    }
)
fig.show()


# %%


# %% [markdown]
# ## 企业景气及企业家信心指数

# %%
macro_china_enterprise_boom_index_df = ak.macro_china_enterprise_boom_index()

# %%
def jidu_to_datetime(s):
    if not isinstance(s, str):
        return pd.NaT  # 表示时间缺失值
    year = s[:4]
    quarter = s[6]
    month = str((int(quarter) - 1) * 3 + 1 )
    return pd.to_datetime(f'{year}-{month}-01')

# %%
macro_china_enterprise_boom_index_df['quarter'] = macro_china_enterprise_boom_index_df['季度'].apply(jidu_to_datetime)

# %%
fig = px.line(
    data_frame= macro_china_enterprise_boom_index_df,
    x = 'quarter',
    y = ['企业景气指数-指数', '企业景气指数-同比', '企业景气指数-环比', '企业家信心指数-指数', '企业家信心指数-同比',
       '企业家信心指数-环比'],
    title='中国企业景气指数和企业家信心指数',
    labels={
        'quarter':'日期',
        'value': '增长率%',
        'variable':'指数类型',
    }
)
fig.show()

# %% [markdown]
# ## 税收收入

# %%
macro_china_national_tax_receipts_df = ak.macro_china_national_tax_receipts()

# %%
# copy保证新的副本
plot_year_df = macro_china_national_tax_receipts_df[macro_china_national_tax_receipts_df['季度'].str.contains('1-4季度')].copy()


# %%
plot_year_df['年份'] = plot_year_df['季度'].apply(lambda x:pd.to_datetime(f'{x[:4]}', format='%Y'))

# %%


# %%


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
fig = make_subplots(specs=[
    [{"secondary_y": True}]
])
fig.add_trace(
    go.Scatter(
        x = plot_year_df['年份'],
        y = plot_year_df['税收收入合计'],
        name= '税收收入' # 图例使用
    ),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(
        x = plot_year_df['年份'],
        y = plot_year_df['较上年同期'],
        name= '年增长率'
    ),
    secondary_y=True
)
fig.update_layout(
    title_text = '中国税收入年度报告'
)
fig.update_xaxes(
    title_text = 'Year'
)
fig.update_yaxes(
    title_text = '税收收入，单位：亿美元'
)
fig.update_yaxes(
    title_text = '较去年同比增长率：%',
    secondary_y= True
)

fig.show()

# %% [markdown]
# ## 银行理财产品发行数量

# %%
macro_china_bank_financing_df = ak.macro_china_bank_financing()

# %%
macro_china_bank_financing_df['日期'] = pd.to_datetime(macro_china_bank_financing_df['日期'])


# %%
# 配合%显示tickformat
macro_china_bank_financing_df['涨跌幅'] = macro_china_bank_financing_df['涨跌幅'].apply(lambda x:x/100)
macro_china_bank_financing_df['近3月涨跌幅'] = macro_china_bank_financing_df['近3月涨跌幅'].apply(lambda x:x/100)
macro_china_bank_financing_df['近6月涨跌幅'] = macro_china_bank_financing_df['近6月涨跌幅'].apply(lambda x:x/100)
macro_china_bank_financing_df['近1年涨跌幅'] = macro_china_bank_financing_df['近1年涨跌幅'].apply(lambda x:x/100)
macro_china_bank_financing_df['近2年涨跌幅'] = macro_china_bank_financing_df['近2年涨跌幅'].apply(lambda x:x/100)
macro_china_bank_financing_df['近3年涨跌幅'] = macro_china_bank_financing_df['近3年涨跌幅'].apply(lambda x:x/100)


# %%
# 2015 年之后比较稳定
macro_china_bank_financing_df = macro_china_bank_financing_df[macro_china_bank_financing_df['日期'].dt.year > 2015]

# %%
plotdf = macro_china_bank_financing_df

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols= 1,
    shared_xaxes=  True,
    vertical_spacing=0.08, 
)
# 上子图：发行数量
fig.add_trace(
    go.Scatter(    
        x = plotdf['日期'],
        y = plotdf['最新值'],
        name = '银行理财产品发行数量'
    ),
    row = 1, col = 1,
)
# 下子图： 主线 短期波动。
fig.add_trace(
    go.Scatter(
        x = plotdf['日期'],
        y = plotdf['涨跌幅'],
        name = '月涨跌幅' ,
    ),
    row = 2, col = 1,
    
)
# 下子图： 副线1 短期波动。
fig.add_trace(
    go.Scatter(
        x = plotdf['日期'],
        y = plotdf['近6月涨跌幅'],
        name = '近6月涨跌幅' ,        
    ),
    row = 2, col = 1,
    
)
# 下子图： 副线2 长期波动。
fig.add_trace(
    go.Scatter(
        x = plotdf['日期'],
        y = plotdf['近1年涨跌幅'],
        name = '近1年涨跌幅' ,
    ),
    row = 2, col = 1,
    
)
# 下子图： 副线3 长期波动。
fig.add_trace(
    go.Scatter(
        x = plotdf['日期'],
        y = plotdf['近3年涨跌幅'],
        name = '近3年涨跌幅' ,
    ),
    row = 2, col = 1,
    
)
# 整个图
fig.update_layout(
    title = '银行发行理财产品数量报告',
    
)
fig.update_yaxes(
    title = '理财产品发行数量',
    row = 1, col = 1
)
# 指定子图
fig.update_yaxes(
    title = '增长率%', 
    tickformat=".0%",
    row = 2, col = 1 
)

fig.show()

# %%


# %%



