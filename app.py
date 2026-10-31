# -*- coding: utf-8 -*-
"""
DataPilot 数据质量分析与智能预处理 Agent（通用版 Streamlit）
- 支持上传 CSV / Excel，也可一键加载 SeoulBike 演示数据
- 自动识别数值列/类别列/日期列；异常值用 IQR 通用规则
- 用户自选目标列做低/中/高三分类
运行: streamlit run app.py
"""
import os, numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

SEED = 20261030
np.random.seed(SEED)
_HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_PROB = os.path.join(_HERE, "data", "seoul_injected.csv")
DEMO_LOG = os.path.join(_HERE, "data", "injection_log.csv")
SEoul_RULES = {
    'Hour': lambda s: ~s.between(0,23),
    'Humidity(%)': lambda s: ~s.between(0,100),
    'Rainfall(mm)': lambda s: (s<0)|(s>100),
    'Snowfall (cm)': lambda s: (s<0)|(s>30),
    'Wind speed (m/s)': lambda s: (s<0)|(s>10),
    'Temperature(°C)': lambda s: (s<-30)|(s>45),
}

st.set_page_config(page_title="DataPilot 数据质量 Agent", layout="wide", page_icon="📊")
st.title("📊 DataPilot｜数据质量分析与智能预处理 Agent（通用版）")
st.caption("上传任意 CSV/Excel，或加载演示数据 · 种子 20261030 · Profile→Plan→Execute→Validate→Model")

def detect_types(df):
    numeric_cols, cat_cols = [], []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0:
            cat_cols.append(c); continue
        coerced = pd.to_numeric(s, errors='coerce')
        if coerced.notna().mean() > 0.9:
            numeric_cols.append(c)
        else:
            cat_cols.append(c)
    return numeric_cols, cat_cols

def diagnose(df_raw, id_col):
    biz = [c for c in df_raw.columns if c != id_col]
    miss = df_raw.isna().sum(); miss = miss[miss>0]
    dup = int(df_raw.duplicated(subset=biz, keep='first').sum())
    num_cols, cat_cols = detect_types(df_raw[biz])
    type_issues = []
    for c in num_cols:
        s = df_raw[c]
        coerced = pd.to_numeric(s, errors='coerce')
        bad = s.notna() & coerced.isna()
        if bad.sum()>0:
            type_issues.append({'列':c,'问题':'数值列含无法解析的文本','数量':int(bad.sum())})
    cat_issues = []
    for c in cat_cols:
        v = df_raw[c].dropna().astype(str)
        norm = v.str.strip().str.title()
        diff = (v != norm).sum()
        if diff>0:
            cat_issues.append({'列':c,'非标准取值数':int(diff),'示例':list(v[v!=norm].unique()[:3])})
    out_rows = []
    for c in num_cols:
        s = pd.to_numeric(df_raw[c], errors='coerce').dropna()
        if len(s)<10: continue
        q1,q3 = s.quantile([.25,.75]); iqr=q3-q1
        lo,hi = q1-1.5*iqr, q3+1.5*iqr
        cs = pd.to_numeric(df_raw[c], errors='coerce')
        n = int(((cs<lo)|(cs>hi)).sum())
        if n: out_rows.append({'列':c,'IQR下界':round(lo,2),'IQR上界':round(hi,2),'异常值数':n})
    return dict(miss=miss, dup=dup, num_cols=num_cols, cat_cols=cat_cols,
                type_issues=pd.DataFrame(type_issues), cat_issues=pd.DataFrame(cat_issues),
                outliers=pd.DataFrame(out_rows))

st.header("① 数据上传与字段配置")
up = st.file_uploader("上传 CSV 或 Excel", type=['csv','xlsx','xls'])
if st.button("📂 加载演示：SeoulBike 问题注入数据"):
    st.session_state['src']=DEMO_PROB; st.session_state['demo']=True; st.session_state['name']="SeoulBike问题注入(演示)"
if up is not None:
    st.session_state['src']=up; st.session_state['demo']=False; st.session_state['name']=up.name
if 'src' not in st.session_state:
    st.info("请上传 CSV/Excel，或点击按钮加载演示数据。"); st.stop()
src = st.session_state['src']
if isinstance(src, str):
    df_raw = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[''])
else:
    src.seek(0)
    if src.name.endswith('.csv'):
        df_raw = pd.read_csv(src, dtype=str, keep_default_na=False, na_values=[''])
    else:
        df_raw = pd.read_excel(src, dtype=str)
        df_raw = df_raw.where(pd.notna(df_raw), np.nan)
st.success(f"已加载 **{st.session_state['name']}**：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 字段")
st.dataframe(df_raw.head(8), use_container_width=True)
cols = list(df_raw.columns)
id_candidates = ['(无ID列)'] + cols
id_default = cols.index('__实验行ID__')+1 if '__实验行ID__' in cols else 0
id_col = st.selectbox("追踪/ID列（重复检测时排除；没有则选“无”）", id_candidates, index=id_default)
id_col = None if id_col=='(无ID列)' else id_col
date_candidates = ['(无日期列)'] + cols
date_default = next((i+1 for i,c in enumerate(cols) if 'date' in c.lower()), 0)
date_col = st.selectbox("日期列（可选，用于时间排序）", date_candidates, index=date_default)
date_col = None if date_col=='(无日期列)' else date_col
target_col = st.selectbox("建模目标列（数值会自动分箱为低/中/高）", cols,
                          index=cols.index('Rented Bike Count') if 'Rented Bike Count' in cols else 0)

st.header("② 质量诊断")
diag = diagnose(df_raw, id_col)
m1,m2,m3,m4 = st.columns(4)
m1.metric("缺失单元格", int(diag['miss'].sum()))
m2.metric("重复行", diag['dup'])
m3.metric("异常值列数", len(diag['outliers']))
m4.metric("识别到数值列", len(diag['num_cols']))
cc1,cc2 = st.columns(2)
with cc1:
    st.subheader("各列缺失数")
    if len(diag['miss']):
        st.plotly_chart(px.bar(x=diag['miss'].index, y=diag['miss'].values, labels={'x':'列','y':'缺失数'}), use_container_width=True)
    else: st.write("无缺失")
with cc2:
    st.subheader("IQR 异常值概览")
    st.dataframe(diag['outliers'], use_container_width=True)
cc3,cc4 = st.columns(2)
with cc3:
    st.subheader("类型问题")
    st.dataframe(diag['type_issues'], use_container_width=True)
with cc4:
    st.subheader("类别一致性")
    st.dataframe(diag['cat_issues'], use_container_width=True)
if st.session_state.get('demo') and os.path.exists(DEMO_LOG) and id_col=='__实验行ID__':
    st.subheader("🎯 检测召回率（仅演示数据，对照注入日志）")
    inj = pd.read_csv(DEMO_LOG)
    rows=[]
    biz=[c for c in df_raw.columns if c!=id_col]
    lm=inj[inj['问题类型']=='缺失值']; inj_s=set(zip(lm['实验行ID'],lm['字段']))
    det=set()
    for c in df_raw.columns:
        for i in df_raw[df_raw[c].isna()].index: det.add((df_raw.loc[i,id_col],c))
    rows.append(('缺失值',len(inj_s),len(det&inj_s)))
    ld=inj[inj['问题类型']=='重复记录']; inj_d=set(ld['实验行ID'])
    det_d=set(df_raw[df_raw.duplicated(subset=biz,keep=False)][id_col])
    rows.append(('重复记录',len(inj_d),len(inj_d&det_d)))
    lo=inj[inj['问题类型']=='异常值']; inj_o=set(zip(lo['实验行ID'],lo['字段']))
    det_o=set()
    for col,rule in SEoul_RULES.items():
        s=pd.to_numeric(df_raw[col],errors='coerce')
        for i in df_raw[rule(s)&s.notna()].index: det_o.add((df_raw.loc[i,id_col],col))
    rows.append(('异常值(域规则)',len(inj_o),len(det_o&inj_o)))
    rdf=pd.DataFrame(rows,columns=['类型','注入数','命中'])
    rdf['召回率']=(rdf['命中']/rdf['注入数']*100).round(1).astype(str)+"%"
    st.dataframe(rdf, use_container_width=True)

st.header("③ Agent 执行过程")
if st.button("🚀 开始清洗", type="primary"):
    tool_log=[]
    df = df_raw.copy()
    biz = [c for c in df.columns if c != id_col]
    before = {'行数':len(df),'缺失单元格':int(df.isna().sum().sum()),'重复行':diag['dup'],
              '异常值':int(diag['outliers']['异常值数'].sum()) if len(diag['outliers']) else 0,
              '类型问题':int(diag['type_issues']['数量'].sum()) if len(diag['type_issues']) else 0,
              '类别非标准':int(diag['cat_issues']['非标准取值数'].sum()) if len(diag['cat_issues']) else 0}
    df = df.drop_duplicates(subset=biz, keep='first').reset_index(drop=True)
    tool_log.append(f"drop_duplicates：删除 {before['重复行']} 行重复（排除ID列）")
    for c in diag['num_cols']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    tool_log.append(f"to_numeric：{len(diag['num_cols'])} 个数值列统一类型")
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        df = df.sort_values(date_col).reset_index(drop=True)
        tool_log.append(f"parse_dates：解析 {date_col} 并按时间排序")
    for c in diag['cat_cols']:
        df[c] = df[c].apply(lambda v: np.nan if pd.isna(v) else str(v).strip().title())
    tool_log.append("normalize_categories：strip + 标题化")
    out_flags=pd.Series(False,index=df.index)
    for c in diag['num_cols']:
        s=df[c].dropna()
        if len(s)<10: continue
        q1,q3=s.quantile([.25,.75]); iqr=q3-q1; lo,hi=q1-1.5*iqr,q3+1.5*iqr
        bad=(df[c]<lo)|(df[c]>hi)
        df[c+'_outlier']=bad.fillna(False)
        out_flags|=bad.fillna(False)
        df.loc[bad,c]=np.nan
    tool_log.append(f"mark_outliers：IQR 标记 {int(out_flags.sum())} 个异常点并置缺失")
    n=len(df); tr=int(n*.7); va=int(n*.85)
    df['split']='test'; df.loc[:tr-1,'split']='train'; df.loc[tr:va-1,'split']='val'
    trdf=df[df['split']=='train']
    for c in diag['num_cols']: df[c]=df[c].fillna(trdf[c].median())
    for c in diag['cat_cols']:
        if trdf[c].notna().any(): df[c]=df[c].fillna(trdf[c].mode()[0])
    tool_log.append("impute：先切分后填补（中位数/众数，仅用train）")
    after={'行数':len(df),'缺失单元格':int(df[biz].isna().sum().sum()),'重复行':0,
           '异常值':int(out_flags.sum()),'类型问题':0,'类别非标准':0}
    st.session_state['clean']=df; st.session_state['before']=before
    st.session_state['after']=after; st.session_state['tools']=tool_log
    st.success("✅ 清洗完成")
st.subheader("🧠 Agent 清洗计划")
st.markdown("1) 去重（排除ID列）　2) 数值列类型统一　3) 日期解析排序　4) 类别 strip+标题化　5) IQR 异常标记置缺失　6) 先切分后 train 中位数/众数填补")
if 'tools' in st.session_state:
    st.subheader("🔧 工具调用记录")
    for t in st.session_state['tools']: st.code("✓ "+t)

st.header("④ 清洗效果对比")
if 'clean' in st.session_state:
    cmp=pd.DataFrame({"清洗前":st.session_state['before'],"清洗后":st.session_state['after']})
    st.dataframe(cmp, use_container_width=True)
    st.plotly_chart(go.Figure(data=[
        go.Bar(name='清洗前',x=list(st.session_state['before'].keys()),y=list(st.session_state['before'].values())),
        go.Bar(name='清洗后',x=list(st.session_state['after'].keys()),y=list(st.session_state['after'].values()))]),
        use_container_width=True)
    st.subheader("清洗后预览")
    st.dataframe(st.session_state['clean'].head(10), use_container_width=True)
    st.download_button("⬇️ 下载清洗后 CSV", st.session_state['clean'].to_csv(index=False).encode('utf-8-sig'),
                       "cleaned_data.csv","text/csv")
else:
    st.info("请先在区域③点击「开始清洗」。")

st.header("⑤ 分类建模")
if 'clean' in st.session_state:
    df_c=st.session_state['clean']
    model_name=st.selectbox("选择模型",["RandomForest","GradientBoosting","DecisionTree","LogisticRegression"])
    if st.button("🎯 训练模型"):
        yraw=pd.to_numeric(df_c[target_col],errors='coerce')
        if yraw.notna().mean()>0.9:
            trm=df_c['split']=='train'
            q1,q2=yraw[trm].quantile([1/3,2/3]).values
            y=yraw.apply(lambda v:'低' if v<=q1 else('中' if v<=q2 else '高'))
            st.info(f"目标列「{target_col}」为数值，按 train 三分位分箱：低≤{int(q1)} < 中≤{int(q2)} < 高")
        else:
            y=df_c[target_col].astype(str)
        drop=[target_col,id_col,date_col,'split']+[c for c in df_c.columns if c.endswith('_outlier')]
        X=pd.get_dummies(df_c.drop(columns=[d for d in drop if d in df_c.columns]))
        tr=df_c['split']=='train'; te=df_c['split']=='test'
        mp={"RandomForest":RandomForestClassifier(300,random_state=SEED,n_jobs=-1,class_weight='balanced'),
            "GradientBoosting":GradientBoostingClassifier(random_state=SEED),
            "DecisionTree":DecisionTreeClassifier(random_state=SEED,class_weight='balanced'),
            "LogisticRegression":LogisticRegression(max_iter=1000,class_weight='balanced')}
        clf=mp[model_name].fit(X[tr],y[tr])
        pred=clf.predict(X[te])
        f1=f1_score(y[te],pred,average='macro')
        st.session_state['f1']=f1; st.session_state['y']=y; st.session_state['pred']=pred
        st.success(f"**测试集 Macro-F1 = {f1:.4f}**")
    if 'f1' in st.session_state:
        k1,k2=st.columns(2)
        with k1:
            st.metric("Macro-F1",f"{st.session_state['f1']:.4f}")
            vc=st.session_state['y'].value_counts()
            st.plotly_chart(px.pie(values=vc.values,names=vc.index,title="类别分布"),use_container_width=True)
        with k2:
            labels=sorted(st.session_state['y'].unique())
            cm=confusion_matrix(st.session_state['y'][df_c['split']=='test'],st.session_state['pred'],labels=labels)
            st.plotly_chart(go.Figure(data=go.Heatmap(z=cm,x=['预测'+l for l in labels],y=['真实'+l for l in labels],
                text=cm,texttemplate='%{text}',colorscale='Blues')),use_container_width=True)
        rep=classification_report(st.session_state['y'][df_c['split']=='test'],st.session_state['pred'])
        st.code(rep)
        st.download_button("⬇️ 下载实验报告(txt)",
            f"DataPilot Report\nModel:{model_name}\nMacro-F1:{st.session_state['f1']:.4f}\n\n{rep}".encode('utf-8'),
            "experiment_report.txt","text/plain")
else:
    st.info("请先完成区域③清洗。")
