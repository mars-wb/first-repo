import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# 读取数据
try:
    df = pd.read_csv('Lecture 1 Data.csv', encoding='gbk')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('Lecture 1 Data.csv', encoding='gb18030')
    except UnicodeDecodeError:
        df = pd.read_csv('Lecture 1 Data.csv', encoding='latin-1')

# 数据清洗：处理 default 列
if 'default' in df.columns:
    if df['default'].dtype == 'object':
        # print("Original 'default' values:", df['default'].unique())
        # 根据实际情况修改映射关系
        df['default'] = df['default'].replace({True: 1, False: 0})
        df['default'] = df['default'].replace({'是': 1, '否': 0})  # 如果还存在中文
        # print("Mapped 'default' values:", df['default'].unique())

    # 检查并处理 'default' 列的缺失值
    if df['default'].isnull().any():
        # print("Found missing values in 'default' column. Dropping rows with missing values.")
        df.dropna(subset=['default'], inplace=True)

    # 确保 'default' 列是数值型 (int 或 float)
    df['default'] = pd.to_numeric(df['default'], errors='coerce')

# 数据清洗：处理 deal 列
if 'deal' in df.columns:
    if df['deal'].dtype == 'object':
        # print("Original 'deal' values:", df['deal'].unique())
        df['deal'] = df['deal'].map({'是': 1, '否': 0})
        # print("Mapped 'deal' values:", df['deal'].unique())

    # 检查并处理 'deal' 列的缺失值
    if df['deal'].isnull().any():
        # print("Found missing values in 'deal' column. Dropping rows with missing values.")
        df.dropna(subset=['deal'], inplace=True)

    # 确保 'deal' 列是数值型 (int 或 float)
    df['deal'] = pd.to_numeric(df['deal'], errors='coerce')

# 数据清洗：处理 tencentscore, gaodescore, highcontact 列
for col in ['tencentscore', 'gaodescore', 'highcontact']:
    if col in df.columns:
        # 检查数据类型
        # print(f"Data type of '{col}': {df[col].dtype}")

        # 检查唯一值
        # print(f"Unique values in '{col}': {df[col].unique()}")

        # 检查并处理缺失值
        if df[col].isnull().any():
            # print(f"Found missing values in '{col}' column. Filling with median.")
            df[col].fillna(df[col].median(), inplace=True)

        # 确保列是数值型
        if col == 'highcontact':
            df[col] = df[col].astype(int)  # 将 highcontact 列强制转换为 int
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 再次检查数据类型
        # print(f"Data type of '{col}' after conversion: {df[col].dtype}")

# 1. 计算描述性统计 (精简版)
def generate_summary_statistics_simplified(df):
    """
    生成关键变量的描述性统计 (精简版)
    """
    variables = {
        'age': '年龄',
        'instalments_amount': '贷款金额',
        'nominalrates': '利率',
        'tencentscore': '腾讯信用分',
        'gaodescore': '高德信用分',
        'highcontact': '是否有频繁联系人',
        'deal': '是否批准',
        'default': '是否违约'
    }

    stats_list = []
    for var, name in variables.items():
        if var in df.columns:
            if df[var].dtype in ['bool', 'object']:
                stats_dict = {
                    'Variable': name,
                    'Mean': df[var].mean(),
                    'Std': '-',
                    'Min': '-',
                    'Max': '-'
                }
            else:
                stats_dict = {
                    'Variable': name,
                    'Mean': round(df[var].mean(), 3),
                    'Std': round(df[var].std(), 3),
                    'Min': round(df[var].min(), 3),
                    'Max': round(df[var].max(), 3)
                }
            stats_list.append(stats_dict)

    stats_df = pd.DataFrame(stats_list)
    return stats_df

def delinquency_credit_regression(df):
    """
    研究违约可能性与信用分数的关系
    """
    if 'default' not in df.columns or 'tencentscore' not in df.columns or 'gaodescore' not in df.columns:
        raise ValueError("Required columns are missing for delinquency regression")

    X = sm.add_constant(df[['tencentscore', 'gaodescore']])
    y = df['default']

    model = sm.Logit(y, X)
    results = model.fit(disp=0)

    return results

def approval_credit_regression(df):
    """
    研究贷款审批与信用分数的关系
    """
    if 'deal' not in df.columns or 'tencentscore' not in df.columns or 'gaodescore' not in df.columns:
        raise ValueError("Required columns are missing for approval credit regression")
    X = sm.add_constant(df[['tencentscore', 'gaodescore']])
    y = df['deal']

    model = sm.Logit(y, X)
    results = model.fit(disp=0)

    return results

def approval_contact_regression(df):
    """
    研究贷款审批与是否有频繁联系人的关系
    """
    if 'deal' not in df.columns or 'highcontact' not in df.columns:
        raise ValueError("Required columns are missing for approval contact regression")

    X = sm.add_constant(df['highcontact'])
    y = df['deal']

    model = sm.Logit(y, X)
    results = model.fit(disp=0)

    return results

def format_regression_results_simplified(results, variables):
    """
    格式化回归结果，只保留指定变量
    """
    summary = results.summary2().tables[1]

    # 筛选出需要的变量
    summary = summary[summary.index.isin(variables)]

    formatted_results = pd.DataFrame({
        'Variable': summary.index,
        'Coefficient': summary['Coef.'].round(3),
        'Std. Error': summary['Std.Err.'].round(3),
        'z-value': summary['z'].round(3),
        'P>|z|': summary['P>|z|'].round(3)
    })
    return formatted_results

# 执行分析
summary_stats = generate_summary_statistics_simplified(df)
delinq_results = delinquency_credit_regression(df)
approval_credit_results = approval_credit_regression(df)
approval_contact_results = approval_contact_regression(df)

# 打印精简版报告
print("\nFintech Theory and Practice - Assignment for Lecture One (Simplified Report)\n")

print("1. Summary Statistics Table (Simplified):")
print(summary_stats.to_string(index=False))

print("\n2. Logit Regression Results - Delinquency and Credit Scores:")
print(format_regression_results_simplified(delinq_results, ['tencentscore', 'gaodescore']).to_string(index=False))

print("\n3. Logit Regression Results - Approval and Credit Scores:")
print(format_regression_results_simplified(approval_credit_results, ['tencentscore', 'gaodescore']).to_string(index=False))

print("\n4. Logit Regression Results - Approval and High Contact:")
print(format_regression_results_simplified(approval_contact_results, ['highcontact']).to_string(index=False))