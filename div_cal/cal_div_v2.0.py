import numpy as np
import pandas as pd
from tqdm import tqdm
import time

##################################################################
# 参数
##################################################################
LAG_NUM = 30  # 历史信息数量
LAG_PERIOD = 4  # 当前交易日滞后期
OBS_J = 3  # 观测期 : 用于历史信息的平均预测和线性预测
PRE_K = 1  # 预测期
assert OBS_J < LAG_PERIOD < LAG_NUM

##################################################################
# 1.转换分红的面板数据 按照股票名存储历史信息
##################################################################
DIV_TABLE = pd.read_parquet('AShareDividend.parquet',
                            columns=['s_div_progress', 'stockcode', 'report_period', 'ann_date',
                                     'cash_dvd_per_sh_pre_tax', 's_div_baseshare'])  # 读取原始数据 筛选计算列
DIV_TABLE = DIV_TABLE[DIV_TABLE['s_div_progress'] == '3']  # 只保留3
DIV_TABLE['dvd_pre_tax'] = DIV_TABLE['cash_dvd_per_sh_pre_tax'] * DIV_TABLE['s_div_baseshare'] * 10000  # 计算总分红
# 按照stockcode升序后,再按照ann_date降序 # 取每个股票最近N个历史数据
DIV_TABLE = (DIV_TABLE.sort_values(['stockcode', 'ann_date'], ascending=[1, 0]).groupby(['stockcode']).head(LAG_NUM))
DIV_TABLE['ANNDATE_MAX'] = DIV_TABLE.groupby(['stockcode'])['ann_date'].cumcount()  # 由于已经排序,cumcount值就是日期从近到远的顺序
DIV_P_TABLE = pd.pivot_table(DIV_TABLE, index=['stockcode'], columns=['ANNDATE_MAX'],
                             values=['ann_date', 'report_period', 'dvd_pre_tax']).fillna(0)  # 转置:按照信息排序后转置
DIV_P_TABLE.columns = [i[0] + '_{}'.format(i[1]) for i in DIV_P_TABLE.columns]  # 重命名列名


##################################################################
# 2.合并MV_TABLE表与DIV_P_TABLE表 分块合并方便矩阵运算
##################################################################
MV_TABLE = pd.read_parquet('mv.parquet', columns=['stockcode', 'ann_date', ]).astype(
    {'ann_date': 'uint32', }).set_index('stockcode')
# ---------------测试数据---600738.SH在2018年有3次分红 已核对--------- #
MV_TABLE = MV_TABLE[MV_TABLE.index == '600738.SH'].sort_values(by='ann_date', ascending=False)
st = time.time()
print('start', time.time() - st)
np_code = MV_TABLE.index.to_numpy().reshape(MV_TABLE.shape[0], 1)
np_trade_date = MV_TABLE.iloc[:, 0].to_numpy('<u4').reshape(-1, 1)  # 存储trade_date列
MV_TABLE = MV_TABLE.iloc[:, :0]  # 移除trade_date
np_ann_date = MV_TABLE.join(DIV_P_TABLE.iloc[:, LAG_NUM * 0:LAG_NUM * 1], how='left').to_numpy('<u4')  # ann_date
np_dvd = MV_TABLE.join(DIV_P_TABLE.iloc[:, LAG_NUM * 1:LAG_NUM * 2], how='left').to_numpy('<f8')  # dvd_pre_tax
np_report_period = MV_TABLE.join(DIV_P_TABLE.iloc[:, LAG_NUM * 2:], how='left').to_numpy('<u4')  # report_period
print('end', time.time() - st)
del MV_TABLE  # 释放内存

##################################################################
# 3.求出用于计算的不同矩阵 np_info矩阵激活可用历史信息,保证没有使用未来信息
##################################################################
print('start 基础矩阵', time.time() - st)
np_info = np_ann_date < np_trade_date  # trade_date与ann_date比较,得到可用信息矩阵
np_dvd = np_dvd * np_info  # 激活历史分红矩阵
np_report_period = np_report_period * np_info  # 激活历史报告期矩阵
np_report_year = (np_report_period // 10000).astype('<u2')  # 报告年份矩阵
np_report_day = (np_report_period % 10000).astype('<u2')  # 报告日矩阵 :用于确认季度分红
print('ok 基础矩阵', time.time() - st)

##################################################################
# 4.在目标输出列中填充  mask_same_year 矩阵激活历史信息到目标滞后期
##################################################################
print('start 填充', time.time() - st)
np_lag_year = (np_trade_date // 10000 - (np.arange(LAG_PERIOD) + 1)).astype('<u2')  # 生成目标滞后年份矩阵
np_dvd_lag = np_trade_date  # 在trade_date上生成储存其他信息的矩阵
np_dvd_lag_ar = np_trade_date
for i in tqdm(range(LAG_PERIOD)):  # 在目标滞后年份矩阵中填充
    mask_same_year = np_report_year == np.take(np_lag_year, [i], axis=1)  # 生成同年份激活矩阵
    mask_dvd = (mask_same_year * np_dvd).sum(axis=1).reshape(-1, 1)  # 累加同一报告期分红
    msak_day = (mask_same_year * np_report_day).max(axis=1).reshape(-1, 1) / 1231  # 获得最新的报告期并年化
    # mask_dvd_ar = mask_dvd * np.divide(1, msak_day, where=msak_day != 0)
    mask_dvd_ar = np.where(msak_day != 0, mask_dvd / msak_day, 0)
    np_dvd_lag = np.concatenate((np_dvd_lag, mask_dvd), axis=1)  # 拼接目标滞后年份
    np_dvd_lag_ar = np.concatenate((np_dvd_lag_ar, mask_dvd_ar), axis=1)  # 拼接目标滞后年份
print('end 填充', time.time() - st)


##################################################################
# 5.预期分红计算 df_div
##################################################################
print('start 输出', time.time() - st)
# ---------------实际值---------------#
np_real = np.take(np_dvd_lag, [1], axis=1)
# ---------------预期值:滞后法---------------#
np_lag = np.where(np.take(np_dvd_lag, [1], axis=1) == 0, np.take(np_dvd_lag, [2], axis=1),
                  np.take(np_dvd_lag, [1], axis=1))  # 滞后1期
np_lag = np.where(np_lag == 0, np.take(np_dvd_lag, [3], axis=1), np_lag)  # 滞后2期
# ---------------预期值:平均法---------------#
np_avg = np.average(np.take(np_dvd_lag, list(range(1, OBS_J + 1)), axis=1), axis=1).reshape(-1, 1)
# ---------------预期值:年化滞后法---------------#
np_ar = np.where(np.take(np_dvd_lag_ar, [1], axis=1) == 0, np.take(np_dvd_lag_ar, [2], axis=1),
                 np.take(np_dvd_lag_ar, [1], axis=1))  # 滞后1期
# ---------------预期值:线性回归法---------------#
Y = np.fliplr(np_dvd_lag[:, 2:2 + OBS_J]).T  # 用后面预测前面
X = np.array([[1] * OBS_J, range(OBS_J)]).T  # 系数矩阵
X_PRE = np.array([[1] * PRE_K, range(OBS_J, OBS_J + PRE_K)]).T  # 待预测期矩阵
np_ols = X_PRE.dot(np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)).T  # OLS参数计算公式: Beta=(X'Y)/(X'X), Y=BetaX
np_ols = np.where(np_ols < 0, 0, np_ols)  # 清除为0的预测值
# ---------------拼接目标输出---------------#
np_target = np.concatenate((np_code, np_trade_date, np_real, np_lag, np_avg, np_ar, np_ols), axis=1)
print('end 输出', time.time() - st)

