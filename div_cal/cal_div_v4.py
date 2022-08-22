import time
import numpy as np
import pandas as pd

# ----------------参数和命名----------------#
LAG_PERIOD = 4  # 滞后期
REFER_DATE = 'ann_date'  # 分红确认的日期: 可选ann_date,s_div_prelandate
MERGE_COLUMN = ['report_year', 'dvd_pre_tax_sum', REFER_DATE + '_max']  # 计算出的列的命名
TARGET_YEAR = 2020
# ----------------读取原始数据----------------#

DIV_TABLE = pd.read_parquet('AShareDividend.parquet')

# ----------------筛选计算列----------------#
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE = DIV_TABLE[['stockcode', 'report_period', REFER_DATE, 'cash_dvd_per_sh_pre_tax', 's_div_baseshare']]

##################################################################
# 1.转换分红的面板数据为方便计算的矩阵
##################################################################

# ----------------排序后保留20期最近的历史记录----------------#
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总股息
# 按照stockcode升序后,再按照ann_date降序
DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0], inplace=True)

# ---------------取排序号的前N个数据----------------#
df_group = DIV_TABLE.groupby(['stockcode']).head(LAG_PERIOD).copy()

# ---------------计算组内日期排序----------------#
# 由于已经排序,cumcount值就是日期从近到远的顺序
df_group['ANNDATE_MAX'] = df_group.groupby(['stockcode'])['ann_date'].cumcount()

# ---------------转置----------------#
INFO_TABLE = pd.pivot_table(df_group, index=['stockcode'], columns=['ANNDATE_MAX'],
                            values=['ann_date', 'report_period', 'dvd_pre_tax'])

##################################################################
# 2.MV_TABLE表与INFO_TABLE表进行矩阵计算
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet')
MV_TABLE = MV_TABLE[['stockcode', 'ann_date', ]]
MV_INFO_TABLE = pd.merge(MV_TABLE, INFO_TABLE[['ann_date', 'report_period', 'dvd_pre_tax']], how='left', on='stockcode')
# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['stockcode'] == '600738.SH']

# ---------------速度记录---------------#
st1 = time.time()
# MV_INFO_TABLE.fillna(0.0, inplace=True)
print('fillna', time.time() - st1)
##################################################################
# 矩阵计算
##################################################################
for i in range(LAG_PERIOD):
    # ---------------可用信息矩阵----------------#
    MV_INFO_TABLE[('info', i)] = np.where(MV_INFO_TABLE['ann_date'] > MV_INFO_TABLE[('ann_date', i)], 1, 0)
    print('可用信息矩阵', time.time() - st1)

    # ---------------可用报告期矩阵-info_report_year---------------#
    # df1 = MV_INFO_TABLE[('info', i)]
    # df2 = MV_INFO_TABLE[('report_period', i)].astype('str').str[:4].astype('float')
    # MV_INFO_TABLE[('info_report_year', i)] = pd.eval('df1 * df2')
    MV_INFO_TABLE[('info_report_year', i)] = np.where(MV_INFO_TABLE[('info', i)] == 1,
                                                      MV_INFO_TABLE[('report_period', i)].astype(
                                                          'str').str[:4].astype('float'), 0.0)
    print('可用报告期矩阵', time.time() - st1)
    # ---------------年化因子矩阵----------------#
    # 取出日期
    MV_INFO_TABLE[('info_report_ar', i)] = np.where(
        MV_INFO_TABLE[('report_period', i)] != 0.0,
        MV_INFO_TABLE[('report_period', i)].astype('str').str[4:].str.lstrip('0'), 0.0)
    # 求年化
    MV_INFO_TABLE[('info_report_ar', i)] = MV_INFO_TABLE[('info_report_ar', i)].astype('float') / 1231.0
    # 年化因子
    MV_INFO_TABLE[('info_report_ar', i)] = np.where(
        MV_INFO_TABLE[('info_report_ar', i)] != 0.0,
        (1.0 / MV_INFO_TABLE[('info_report_ar', i)]) - 1.0, 0.0)
    MV_INFO_TABLE[('info_report_ar', i)] = MV_INFO_TABLE[('info_report_ar', i)] * MV_INFO_TABLE[('info', i)]
    print('年化因子矩阵', time.time() - st1)
    # ---------------可用分红矩阵----------------#
    MV_INFO_TABLE[('dvd_pre_tax_sum', i)] = MV_INFO_TABLE[('dvd_pre_tax', i)] * MV_INFO_TABLE[('info', i)]
    print('可用分红矩阵', time.time() - st1)

print('矩阵计算完成', time.time() - st1)

for i in range(LAG_PERIOD):
    # ---------------目标年份矩阵----------------#
    MV_INFO_TABLE[('year', i)] = TARGET_YEAR - i
    MV_INFO_TABLE[('year_sum', i)] = 0.0
print(time.time() - st1)
# ---------------在目标年份矩阵中迭代合并-得到最终的总分红----------------#
for i in range(LAG_PERIOD):
    # ---------------分红累积矩阵----------------#
    # 从右向左累积
    right = LAG_PERIOD - 1 - i  # LAG_PERIOD 4: 0,1,2,3 ;right 3,2,1,0
    left = right - 1
    # 从右往左累积到最左边的列
    if left >= 0:
        MV_INFO_TABLE[('dvd_pre_tax_sum', left)] = np.where(
            MV_INFO_TABLE[('info_report_year', left)] == MV_INFO_TABLE[('info_report_year', right)],  # 只累加相同年份
            MV_INFO_TABLE[('dvd_pre_tax_sum', left)] + MV_INFO_TABLE[('dvd_pre_tax_sum', right)],
            MV_INFO_TABLE[('dvd_pre_tax_sum', left)])

    # ---------------填充目标日期矩阵----------------#
    for j in reversed(range(LAG_PERIOD)):  # 从同一年使用最新的累计分红 ,并排除0,保证分红更新
        MV_INFO_TABLE[('year_sum', i)] = np.where(
            (MV_INFO_TABLE[('year', i)] == MV_INFO_TABLE[('info_report_year', j)]) & (
                    MV_INFO_TABLE[('dvd_pre_tax_sum', j)] > 0),
            MV_INFO_TABLE[('dvd_pre_tax_sum', j)], MV_INFO_TABLE[('year_sum', i)])

print(time.time() - st1)

# ---------------测试数据---------------#
MV_INFO_TABLE.sort_values(by='ann_date', ascending=False, inplace=True)
# MV_INFO_TABLE = MV_INFO_TABLE[MV_INFO_TABLE['ann_date'].astype('str').str[:-4].isin(['2017','2018', '2019'])]