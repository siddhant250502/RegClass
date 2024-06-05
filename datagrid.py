import streamlit as st
import pandas as pd
import os
from imblearn.pipeline import Pipeline
import statistics
import datetime
import graphviz
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, r2_score
import plotly.graph_objects as go
import time
from scipy.stats import norm
from sklearn.decomposition import PCA
import numpy as np
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
import pickle
from streamlit_option_menu import option_menu
import toml
import math
from streamlit_extras.stylable_container import stylable_container
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import kaleido
import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree


st.set_page_config(page_title="AI Model Analysis",layout="wide")
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
hide_streamlit_style = """
<style>
#MainMenu {
    visibility: hidden;
    }
footer {
    visibility: hidden;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
font = """
<style>
    body {
        font-family: 'Roboto' !important;
    }
</style>
"""
st.markdown(font, unsafe_allow_html=True)
 
with open('.streamlit/config.toml', 'r') as f:
    config = toml.load(f)

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = None

if 'btn_state' not in st.session_state:
    st.session_state['btn_state'] = False
    
if "page" not in st.session_state:
    st.session_state.page = 0
    
if 'file_path' not in st.session_state:
    st.session_state['file_path'] = ' '

if 'reg' not in st.session_state:
    st.session_state['reg'] = False

if 'filter_df' not in st.session_state:
    st.session_state['filter_df'] = None
    
if 'data_header_df' not in st.session_state:
    st.session_state['data_header_df'] = None
    
if 'histogram_df' not in st.session_state:
    st.session_state['histogram_df'] = None
    
if 'corr_df' not in st.session_state:
    st.session_state['corr_df'] = None
    
if 'EXCLUDE_df' not in st.session_state:
    st.session_state['EXCLUDE_df'] = None
    
if 'col_name' not in st.session_state:
    st.session_state['col_name'] = set()
    
if "prev_tab" not in st.session_state:
    st.session_state['prev_tab'] = "Data Header"

if "model" not in st.session_state:
    st.session_state['model'] = 0
    
if "checkpoint" not in st.session_state:
    st.session_state['checkpoint'] = None
    
if "model_type" not in st.session_state:
    st.session_state['model_type'] = None

    
def nextpage(): st.session_state.page += 1
def restart(): st.session_state.page = 0
def back(): 
    st.session_state.page -= 1 
    st.session_state.button_clicked = None
    st.session_state['col_name'] = set()
    st.session_state['filter_df'] = st.session_state['data_header_df'] = st.session_state['histogram_df'] = st.session_state['corr_df'] = st.session_state['EXCLUDE_df'] = st.session_state['checkpoint'] = st.session_state['model_type'] = None
    st.session_state['prev_tab'] = "Data Header"
placeholder = st.empty()

def click_btn():
    st.session_state['btn_state'] = True

def r_square(df):
    dict_corr = {'Column 1': [], 'Column 2': [], 'R-Squared': []}#
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            lr = LinearRegression()
            lr.fit([[i] for i in df[df.columns[i]]], [[i] for i in df[df.columns[j]]])
            dict_corr['Column 1'].append(df.columns[i])
            dict_corr['Column 2'].append(df.columns[j])
            dict_corr['R-Squared'].append(lr.score([[i] for i in df[df.columns[i]]], [[i] for i in df[df.columns[j]]]))
    return(pd.DataFrame(dict_corr).sort_values(by = 'R-Squared', ascending=False))
    

def data_cleaning(data1):
    # Format columns
    for i in data1.columns:
        if type(data1[i]) == str:
            data1[i] = data1[i].str.replace(",","")
    # Removing columns with string data
    dt = []
    for num,i in enumerate(data1.dtypes):
        if i != "int64" and i != "float64":
            dt.append(num)
    data1.drop(data1.columns[dt],axis=1,inplace=True)
    
    # Dealing with Null values
    data1.fillna(0, inplace=True)
    # data1 = mice(data1)
    # dropping colummns with one uniques value
    for i in data1.columns:
        if len(data1[f'{i}'].unique()) <= 1:
            data1.drop(i, inplace=True, axis=1) 
        elif len(data1[f'{i}'].unique())==2  and True in np.isnan(data1[f'{i}'].unique()):
            data1.drop(i, inplace=True, axis=1) 
    return data1

def mice(df):
    imputer = IterativeImputer(max_iter=10, random_state=21)
    imputed_values = imputer.fit_transform(df)
    df = pd.DataFrame(imputed_values, columns=df.columns)
    return df
    
def interactive_plot(df, x_axis, y_axis, ols):
    new_data = {True:'INCLUDE', False:'EXCLUDE'}
    df = df.replace({'EXCLUDE/INCLUDE':new_data})
    try:
        if x_axis and y_axis:
            if ols:
                fig = px.scatter(df[df['EXCLUDE/INCLUDE']=='INCLUDE'], x=x_axis, y=y_axis, color='EXCLUDE/INCLUDE', color_discrete_map={'INCLUDE':'#30B9EF'}, symbol='EXCLUDE/INCLUDE', symbol_map={'INCLUDE':'circle-open'}, trendline='ols', trendline_color_override='#000000', height=450, custom_data=[x_axis, y_axis]) 
                fig.update_traces(hovertemplate=None)
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color='EXCLUDE/INCLUDE', color_discrete_map={'EXCLUDE':'#D22B2B', 'INCLUDE':'#30B9EF'}, symbol='EXCLUDE/INCLUDE', symbol_map={'EXCLUDE':'x', 'INCLUDE':'circle-open'}, height=450, custom_data=[df[x_axis], df[y_axis]])    
        fig.update_layout(title_text='Column Vs Column Scatter Plot', legend_title_text='Data Points')
        return fig
    except Exception as e:
        st.error(e)
        
def std(df):
    x = 0
    mean = df.mean()
    for i in df:
        x+=(mean-i)**2
    stddev = math.sqrt(x/(len(df)-1))
    return stddev

def difference(x1):    
    return abs(x1.iloc[1]-x1.iloc[0]) 

def est_std(df, opt):
    d2 = {1:1.128, 2:1.128, 3:1.693, 4:2.059, 5:2.326, 6:2.534, 7:2.704, 8:2.847, 9:2.970, 10:3.078}
    df.columns = [i.upper() for i in df.columns]
    sg = {}
    for i in df.columns:
        if "(" in i and ")" in i:
            i=i[:i.index("(")]
        x=sg.get(i,0)
        sg[i]=x+1
    for i in sg.keys():
        if sg[i]>1:
            df[f'{i}_Avg'] = 0
            for j in df.columns[:-1]:
                if i in j:
                    df[f'{i}_Avg'] += df[j]
            df[f'{i}_Avg']=df[f'{i}_Avg']/sg[i]
    ma = df[opt.upper()].rolling(window=2).apply(difference)
    if "(" in opt and ")" in opt or'_avg' in opt:
        opt=opt[:-4]
    return (ma.mean()/d2[sg[opt.upper()]])

def plot_columns(df, opt, sigma, est_sigma, curve):
    new_data = {True:'INCLUDE', False:'EXCLUDE'}
    df = df.replace({'EXCLUDE/INCLUDE':new_data})
    mean = statistics.mean(df[opt])
    if sigma:
        stddev = std(df[opt])
    if est_sigma:
        stddev = est_std(df, opt)
        # cor = []
        # for i in df[opt]:
        #     cor.append((i-mean)**2)
        # stddev = math.sqrt(sum(cor)/len(df[opt])-1)
    fig = px.histogram(df, x=opt, color_discrete_map={'EXCLUDE':'#D22B2B', 'INCLUDE':'#30B9EF'}, color='EXCLUDE/INCLUDE')
    fig.update_layout(legend_title_text='Data Points')
    fig.update_traces(xbins_size= (df[opt].max()-df[opt].min())/math.sqrt(len(df[opt]))) 
    fig0 = px.histogram(df, x=opt, color_discrete_sequence=['#30B9EF'], histnorm = 'probability density')
    f = fig0.full_figure_for_development(warn=False)
    xbins = f.data[0].xbins
    plotbins = list(np.arange(start=xbins['start'], stop=xbins['end']+xbins['size'], step=xbins['size']))
    counts, bins = np.histogram(list(f.data[0].x), bins=plotbins)
    # st.write(bins)
    bins = [i+(df[opt].max()-df[opt].min())/math.sqrt(len(df[opt]))/2 for i in bins]
    if curve:
        fig1 = px.line(x=bins[:-1], y=counts)
        fig1.update_traces(line_width=0.9, patch = {"line_shape":"spline"})
        fig.add_trace(fig1.data[0])
    if sigma or est_sigma:
        one_sigma_plus = mean+(1*stddev)
        one_sigma_minus = mean-(1*stddev)
        two_sigma_plus = mean+(2*stddev)
        two_sigma_minus = mean-(2*stddev)
        three_sigma_plus = mean+(3*stddev)
        three_sigma_minus = mean-(3*stddev)
        fig.add_vline(mean, line_dash="dash",annotation_text="Mean", annotation_position="left")
        fig.add_vrect(x0=one_sigma_minus, x1=one_sigma_plus, annotation_text='1-sigma', fillcolor="#00FF00", opacity=0.2, line_width=0.5)
        # fig.add_vrect(x0=one_sigma_plus, x1=mean, annotation_text='one-sigma', fillcolor="green", opacity=0.25)
        fig.add_vrect(x0=two_sigma_minus, x1=one_sigma_minus, fillcolor="#FFFF00", opacity=0.2, line_width=0.5, annotation_text='2-sigma')
        fig.add_vrect(x0=two_sigma_plus, x1=one_sigma_plus,  fillcolor="#FFFF00", opacity=0.2, line_width=0.5)
        fig.add_vrect(x0=three_sigma_minus, x1=two_sigma_minus, fillcolor="purple", opacity=0.2, line_width=0.5)
        fig.add_vrect(x0=three_sigma_plus, x1=two_sigma_plus, annotation_text='3-sigma', fillcolor="#A020F0", opacity=0.2, line_width=0.5)
    if sigma and est_sigma:
        st.error('Please Select one')
        return px.bar(None)
    else:
        fig.update_layout(bargap=0.1)
        return fig
    
    
def correlation(df):
    res_df = pd.DataFrame(index=df.columns, columns=df.columns)
    for i in res_df.columns:
        for j in res_df.index:
            lr = LinearRegression()
            lr.fit([[i] for i in df[i]], [[i] for i in df[j]])
            res_df[i][j] = lr.score([[i] for i in df[i]], [[i] for i in df[j]])
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=res_df, x=df.columns, y=df.columns, colorscale='RdBu_r',  hovertemplate='X-Value: %{x}<br>Y-Value: %{y}<br>R2: %{z}<extra></extra>'))
    fig.update_layout(modebar_remove=['zoom', 'pan'])
    return fig
    
def highlight_survived(s):
    return ['background-color: white']*len(s) if s['EXCLUDE/INCLUDE'] else ['background-color: grey']*len(s)

def regre(df, indep_vars, dep_vars):
    start = time.time()
    df = data_cleaning(df)
    X = df[indep_vars]
    y = df[dep_vars[0]]
    oversample = SMOTE()
    # undersample = 

    if len(y.unique()) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        rfr = RandomForestRegressor(max_depth = 5, max_leaf_nodes=12)
        st.session_state['model_type'] = 'regression'
        try:
            rfr.fit(X_train, y_train)
        except ValueError as e:
            st.error(e)
        try:
            importances = rfr.feature_importances_
            forest_importances = pd.Series(importances, index=X.columns)
            forest_importances = forest_importances.sort_values(ascending=False)
            with st.container(border=True):
                st.info("**Visualization of how each Independent variables Impact the Dependent variables**")
                st.bar_chart(forest_importances, color='#30B9EF')
        except Exception as e:
            st.error(e)
        st.session_state['model'] = rfr
        try:
            y_pred = rfr.predict(X_test)
        except Exception as e:
            st.error(e)
        st.session_state.predicted = y_pred
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse**0.5
        progress_bar = st.progress(0)
        status_text = st.empty()
        col6, col7 = st.columns([3,1])
        with col6.container(border=True):
            for i in range(1, 101):
                status_text.text("%i%% Complete" % i)
                progress_bar.progress(i)
                time.sleep(0.02)
            plot_chart(y_test, y_pred)
        with col7.container(border=True):
            acc = rfr.score(X_test, y_test)*100
            st.info(f'''**Error Margin**: {rmse.round(3)} \n\n **Accuracy**: {acc.round(2)}% \n\n **Time Took**: {round((time.time()-start),2)} secs''')
            
            
    else:
        rfc = RandomForestClassifier(max_depth = 5, max_leaf_nodes=12)
        st.session_state['model_type'] = 'classification'
        # steps = [('over', oversample)]
        # pipeline = Pipeline(steps=steps)
        # X_sm, y_sm = pipeline.fit_resample(X=X, y=y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        
        try:
            rfc.fit(X_train, y_train)
        except Exception as e:
            st.error(e)
        try:
            importances = rfc.feature_importances_
            forest_importances = pd.Series(importances, index=X.columns)
            forest_importances = forest_importances.sort_values(ascending=False)
            with st.container(border=True):
                st.info("**Visualization of how each Independent variables Impact the Dependent variables**")
                st.bar_chart(forest_importances, color='#1168f5')
        except Exception as e:
            st.error(e)
        st.session_state['model'] = rfc
        try:
            y_pred = rfc.predict(X_test)
        except Exception as e:
            # st.warning("Can't predict due to some error in the Dependant/Independant Variable")
            st.error(e)
        st.session_state.predicted = y_pred
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(1, 101):
            status_text.text("%i%% Complete" % i)
            progress_bar.progress(i)
            time.sleep(0.02)
        acc = accuracy_score(y_test, y_pred)
        acc = round(acc, 2)
        cm = confusion_matrix(y_test, y_pred)
        col6, col7 = st.columns([3,1])
        with col7.container(border=True):
            st.info(f'''**Accuracy**: {round(acc, 2)*100}% \n\n **Time Took**: {round((time.time()-start),2)} secs''') 
        with col6.container(border=True):
            st.info("**Confusion Matrix**")
            fig = go.Figure(data=go.Heatmap(
                   z=cm,
                   x=['True', 'False'],
                   y=['True', 'False'],
                   text=[[cm[0][0], cm[0][1]],
                         [cm[1][0], cm[1][1]]],
                   texttemplate='%{text}',
                   hoverongaps = True))
            fig.update_traces(showscale=False)
            st.plotly_chart(fig)

def plot_chart(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i for i in range(50)], y=y_test[-50:], mode='markers', name='Testing points', line=dict(color="red")))
    fig.add_trace(go.Scatter(x=[i for i in range(50)], y=y_pred[-50:], mode='lines', name='Regression Line',opacity=0.9, line=dict(color="#30B9EF")))
    fig.update_layout(
    title="Regression Analysis",
    xaxis_title="Test Data",
    yaxis_title="Prediction Data",
    legend_title="Legend",
    )
    st.plotly_chart(fig, use_container_width=True)

def oversampling(X_train, y_train):
    smt=SMOTE()
    X_train_sm, Y_train_sm = smt.fit_resample(X_train, y_train)
    return X_train_sm, Y_train_sm
       
  
if st.session_state.page == 0:
    def delete_but(file):
        os.remove(path=file)

    st.title('List of Datasets')
    cwd = os.getcwd()
    files = os.listdir(cwd)
    documents = [f for f in files if os.path.isfile(os.path.join(cwd, f)) and f[-3:] == 'csv']
    data = pd.DataFrame({'Select': [False for i in range(len(documents))],
            'Index': [i+1 for i in range(len(documents))],
            'File Name': documents,
            'Timestamp': [datetime.datetime.fromtimestamp(os.path.getatime(os.path.join(cwd, f))) for f in documents],
            'File Size': [str((os.stat(f).st_size)//1024+1)+" KB" for f in documents]
    })

    res = st.data_editor(data,
                        column_config={"Select": st.column_config.CheckboxColumn(default=False), 
                                        "File Name": st.column_config.Column(width="large"), 
                                        "Timestamp": st.column_config.Column(width="large"),
                                        "File Size": st.column_config.Column(width="small")},
                        hide_index=True, 
                        use_container_width=True,
                        height=350,
                        )
    st.session_state['file_path'] = res.loc[res.Select.idxmax()]['File Name']
    with stylable_container(
        key='btn',
        css_styles=["""
            button {
                
                background-color: #30B9EF;
                color: #fff;
            }
            """,
            """
                button:hover {
                    background-color: #fff;
                    color: #30B9EF;
                }
            """,]
    ):
        upload_file = st.file_uploader("Upload your dataset", type=['csv'])
    if upload_file is not None:
            if upload_file.name[-3:] == 'csv':
                pd.read_csv(upload_file).to_csv(upload_file.name)
            else:
                pd.read_excel(upload_file).to_excel(upload_file.name)
            st.session_state['file_path'] = upload_file.name
    
    
    if len(res[res.Select == True])==1 or upload_file is not None:
        col7, col8, c9 = st.columns([1,0.055,0.07])
        with col8:
            
            st.button("Next", on_click=nextpage, disabled=(st.session_state.page > 1))
        with c9:
            if st.button("Delete"):
                if os.path.isfile(st.session_state['file_path']):
                    ind = res.index[res['File Name']==st.session_state['file_path']]
                    os.remove(st.session_state['file_path'])
                    st.rerun()
                else:
                    st.warning("File doesn't exist")
    else:
        st.write('Please select only 1 option / database')
    


elif st.session_state.page == 1:
    with stylable_container(
        key='btn',
        css_styles="""
            button {
                background: none!important;
                border: none;
                padding: 0!important;
                color: black;
                text-decoration: none;
                cursor: pointer;
                border: none !important;
                }
            button:hover {
                color: #30B9EF ;
            }
            button:focus {
                color: #30B9EF ;
                
            }
        """
    ):
        st.button('Back to Datasets', on_click=back)

    with st.container(border=True):
           
        col1, col2, col3 = st.columns(3)
 
        t1, t2, t3, t4 = st.tabs(['Dataset Info','AI Model', 'Decision Tree Visualization', 'Predictor'])
                  
        # Update content based on button click
        # if option=='Dataset Info':
        with t1:
            with st.container(border=True):
                option = option_menu(
                menu_title=None,
                options=['Data Header', 'Data Statistics', 'Correlation Matrix', 'Plot'],
                icons=['table', 'bar-chart-line', 'graph-up', 'diagram-2'],
                menu_icon=None,
                default_index=0,
                orientation='horizontal'
                    )
                if option=='Data Header':
                    prev_tab = st.session_state.prev_tab
                    st.session_state.prev_tab = option
                    if prev_tab=='Data Header':
                        if st.session_state['data_header_df'] is not None:
                            df = st.session_state['data_header_df']
                        else:
                            if st.session_state['file_path'] is not None:
                                df = pd.read_csv(st.session_state['file_path'])
                                df.columns = [i.upper() for i in df.columns]
                                df = data_cleaning(df)
                                st.session_state['filter_df'] = df
                                df['EXCLUDE/INCLUDE'] = True
                            else:
                                st.warning("File type Not supported")
                              
                    elif prev_tab=='Data Statistics':
                        if st.session_state['histogram_df'] is not None:
                            df = st.session_state['histogram_df']
                        else:
                            df = st.session_state['data_header_df']
                    elif prev_tab=='Correlation Matrix':
                        if st.session_state['corr_df'] is not None:
                            df = st.session_state['corr_df']
                        else:
                            df = st.session_state['histogram_df']
                    elif prev_tab=='Plot':
                        if st.session_state['filter_df'] is not None:
                            df = st.session_state['filter_df']
                        else:
                            df = st.session_state['corr_df']
                    
                    with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                    ):
                        st.subheader('Header of the DataSet')
                    c01, c02, c03 = st.columns(3)
                    
                    with c02:
                        select = st.toggle('Select All')
                    with c01:
                        deselect = st.toggle('Deselect All')
                    if select:
                        df['EXCLUDE/INCLUDE'] = True
                    if deselect:
                        df['EXCLUDE/INCLUDE'] = False 
                    try:
                        with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                        ):
                          st.subheader('Column Selection')
                        res = []
                        for i in df.columns:
                            res.append(True)
                        res_df = pd.DataFrame(data=[res], columns=df.columns)
                        res_df = st.data_editor(
                            res_df,
                            column_config={
                                f'{i}':st.column_config.CheckboxColumn()
                            },
                            # hide_index=True,
                            width=1600
                        )
                        cols = []
                        for i in range(len(res_df.iloc[0])):
                            if res_df.iloc[0][i] == True:
                                cols.append(res_df.columns[i])
                        df = df[cols]
                        while True:
                            df = st.data_editor(
                                df,
                                column_config={
                                    "Exclude/Include": st.column_config.CheckboxColumn()
                                },
                                use_container_width=True,
                                hide_index=False, 
                                height=350,
                                key='table'
                            )       
                    except:
                        pass
                    if st.button("Save Data"):
                        st.session_state['data_header_df'] = df[cols]
                    
                elif option == 'Data Statistics':
                    prev_tab = st.session_state.prev_tab
                    st.session_state.prev_tab = option
                    try:
                        if prev_tab=='Data Header':
                            if st.session_state['data_header_df'] is None:
                                st.session_state['data_header_df'] = pd.read_csv(st.session_state['file_path'])
                                st.session_state['data_header_df'].columns = [i.upper() for i in st.session_state['data_header_df'].columns]
                                st.session_state['data_header_df'] = data_cleaning(st.session_state['data_header_df'])
                                st.session_state['data_header_df']['EXCLUDE/INCLUDE'] = True  
                        elif prev_tab=='Correlation Matrix':
                            if st.session_state['corr_df'] is not None:
                                st.session_state['data_header_df'] = st.session_state['corr_df']
                            else:
                                st.session_state['data_header_df'] = st.session_state['histogram_df']
                        elif prev_tab=='Plot':
                            
                            if st.session_state['filter_df'] is not None:
                                st.session_state['data_header_df'] = st.session_state['filter_df']
                            else:
                                st.session_state['data_header_df'] = st.session_state['corr_df']
                    
                        with stylable_container(
                            key='h3',
                            css_styles="""
                                h3 {
                                    font-size: 16px;
                                }
                            """
                        ):
                            st.subheader('Dataset statistics')
                        with st.expander('Statistical info of the dataset'):
                            st.dataframe(st.session_state['data_header_df'].describe(), use_container_width=True)
                            st.write('----')
                        with stylable_container(
                            key='h3',
                            css_styles="""
                                h3 {
                                    font-size: 16px;
                                }
                            """
                        ):
                            st.subheader('Column Distribution ')
                        opt = st.selectbox('Select any Column', options=st.session_state['data_header_df'].columns[:-1])
                        c4, c10, c6, c100 = st.columns(4)
                        with c4:
                            sigma = st.toggle('Real Stdev')
                        with c10:
                            est_sigma = st.toggle('Est. Stdev')
                        with c6:
                            curve = st.toggle('Curve')
                        if opt:
                            bar_chart = plot_columns(st.session_state['data_header_df'], opt, sigma, est_sigma, curve)

                            bar_chart_selected = plotly_events(
                                bar_chart,
                                select_event=True,
                                
                            )
                        new_df = pd.DataFrame(None, columns=[st.session_state['data_header_df'].columns])
                        if bar_chart_selected:
                            x = (st.session_state['data_header_df'][opt].max()-st.session_state['data_header_df'][opt].min())/math.sqrt(len(st.session_state['data_header_df'][opt]))
                            for i in range(len(bar_chart_selected)):
                                min = bar_chart_selected[i]['x']-(x/2)
                                max = bar_chart_selected[i]['x']+(x/2)
                                temp_df = st.session_state['data_header_df'][st.session_state['data_header_df'][opt]>=min]
                                temp_df = temp_df[temp_df[opt]<=max]
                                new_df = pd.concat([temp_df, new_df])
                                new_df = new_df[new_df.columns[:len(st.session_state['data_header_df'].columns)]]
                                new_df = new_df.drop_duplicates()
                            with stylable_container(
                            key='h3',
                            css_styles="""
                                h3 {
                                    font-size: 16px;
                                }
                            """
                            ):
                                st.subheader('Filtered Dataset Preview')   
                            st.dataframe(new_df[new_df['EXCLUDE/INCLUDE']==True], use_container_width=True, hide_index=True)     
                            st.session_state['histogram_df'] = new_df
                            st.session_state['histogram_df'].reset_index(inplace=True, drop=True)
                    except AttributeError:
                        st.error("You haven't saved the data in Data Header")
                    except ValueError:
                        st.error("You haven't saved the data in Data Header")
                   
                elif option=='Correlation Matrix':
                    prev_tab = st.session_state.prev_tab
                    st.session_state.prev_tab = option
                    # st.write(prev_tab)
                    # tab1, tab2 = st.tabs(['Heatmap', 'Table'])
                    if prev_tab=='Data Header':
                        if st.session_state['data_header_df'] is not None:
                            # st.session_state['histogram_df'] = st.session_state['data_header_df']
                            heatmap = correlation(st.session_state['data_header_df'][st.session_state['data_header_df'].columns[:-1]])
                            st.session_state['corr_df'] = st.session_state['data_header_df']
                        else:
                            st.session_state['data_header_df'] = pd.read_csv(st.session_state['file_path'])
                            st.session_state['data_header_df'].columns = [i.upper() for i in st.session_state['data_header_df'].columns]
                            st.session_state['data_header_df'] = data_cleaning(st.session_state['data_header_df'])
                            st.session_state['data_header_df']['EXCLUDE/INCLUDE'] = True
                            heatmap = correlation(st.session_state['data_header_df'][st.session_state['data_header_df'].columns[:-1]])
                            st.session_state['corr_df'] = st.session_state['data_header_df']
                    elif prev_tab=='Data Statistics':
                        if st.session_state['histogram_df'] is not None:
                            # st.session_state['histogram_df'] = st.session_state['histogram_df']
                            heatmap = correlation(st.session_state['histogram_df'][st.session_state['histogram_df'].columns[:-1]])
                            st.session_state['corr_df'] = st.session_state['histogram_df']
                        else:
                            # st.session_state['histogram_df'] = st.session_state['data_header_df']
                            if st.session_state['data_header_df'] is not None:
                                heatmap = correlation(st.session_state['data_header_df'][st.session_state['data_header_df'].columns[:-1]])
                                st.session_state['corr_df'] = st.session_state['data_header_df']
                            elif st.session_state['filter_df'] is not None:
                                heatmap = correlation(st.session_state['filter_df'][st.session_state['filter_df'].columns[:-1]])
                                st.session_state['corr_df'] = st.session_state['filter_df']
                    elif prev_tab=='Plot':
                        if st.session_state['filter_df'] is not None:
                            # st.session_state['histogram_df'] = st.session_state['filter_df']
                            heatmap = correlation(st.session_state['filter_df'][st.session_state['filter_df'].columns[:-1]])
                            st.session_state['corr_df'] = st.session_state['filter_df']
                        else:
                            # st.session_state['histogram_df'] = st.session_state['corr_df']
                            heatmap = correlation(st.session_state['corr_df'][st.session_state['corr_df'].columns[:-1]])
                            st.session_state['corr_df'] = st.session_state['corr_df']
                    
                    if st.session_state['histogram_df'] is None:
                        st.session_state['histogram_df'] = st.session_state['data_header_df']
                    # with tab1:
                    #     with stylable_container(
                    #         key='h3',
                    #         css_styles="""
                    #             h3 {
                    #                 font-size: 16px;
                    #             }
                    #         """
                    #     ):
                    #         st.subheader('Correlation Matrix Heatmap')
                    #     heatmap = correlation(st.session_state['histogram_df'][st.session_state['histogram_df'].columns[:-1]])
                    #     # config = {'displayModeBar':False}
                    #     heatmap_selected = plotly_events(
                    #         heatmap,
                    #         click_event=True
                    #     )
                    #     # st.plotly_chart(heatmap, use_container_width=True, config=config)
                        
                    #     if heatmap_selected:
                    #         st.session_state['col_name'].add(heatmap_selected[0]['x'])
                    #         st.session_state['col_name'].add(heatmap_selected[0]['y'])
                    #     cols = list(st.session_state['col_name'])
                    #     if len(st.session_state['col_name'])!= 0:
                    #         st.write("Columns selected:")
                    #         st.text(st.session_state['col_name'])
                    #     if st.button('Preview'):
                    #         with stylable_container(
                    #         key='h3',
                    #         css_styles="""
                    #             h3 {
                    #                 font-size: 16px;
                    #             }
                    #         """
                    #         ):
                    #             st.subheader('Filtered Dataset Preview')
                    #         st.session_state['corr_df'] = st.session_state['histogram_df'][cols]
                    #         st.dataframe(st.session_state['corr_df'], hide_index=True, use_container_width=True)
                    # with tab2:
                    table = r_square(st.session_state['corr_df'][st.session_state['corr_df'].columns[:-1]])
                    st.dataframe(table, width=600, hide_index=True)
                             
                elif option=='Plot':  
                    prev_tab = st.session_state.prev_tab
                    st.session_state.prev_tab = option
                    # st.write(prev_tab)
                    if prev_tab=='Data Header':
                        if st.session_state['data_header_df'] is not None:
                            st.session_state['corr_df'] = st.session_state['data_header_df']
                        else:
                            st.session_state['data_header_df'] = pd.read_csv(st.session_state['file_path'])
                            st.session_state['data_header_df'].columns = [i.upper() for i in st.session_state['data_header_df'].columns]
                            st.session_state['data_header_df'] = data_cleaning(st.session_state['data_header_df'])
                            st.session_state['data_header_df']['EXCLUDE/INCLUDE'] = True
                    elif prev_tab=='Data Statistics':
                        if st.session_state['histogram_df'] is not None:
                            st.session_state['corr_df'] = st.session_state['histogram_df']
                        else:
                            st.session_state['corr_df'] = st.session_state['data_header_df']
                    elif prev_tab=='Correlation Matrix':
                        if st.session_state['corr_df'] is not None:
                            st.session_state['corr_df'] = st.session_state['corr_df']
                        else:
                            st.session_state['corr_df'] = st.session_state['histogram_df']
                        
                    if st.session_state['corr_df'] is None and st.session_state['histogram_df'] is not None:
                        st.session_state['corr_df'] = st.session_state['histogram_df']
                    elif st.session_state['corr_df'] is None and st.session_state['histogram_df'] is None:
                        st.session_state['corr_df'] = st.session_state['data_header_df']
                        
                    st.session_state['EXCLUDE_df'] = st.session_state['corr_df']
                    
                    with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                    ):
                        st.subheader('Scatter Plot')
                    x_axis = st.selectbox('X-axis', options=st.session_state['corr_df'].columns[:-1], index=len(st.session_state['corr_df'].columns)-3)
                    y_axis = st.selectbox('Y-axis', options=st.session_state['corr_df'].columns[:-1], index=len(st.session_state['corr_df'].columns)-2)
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        ols = st.toggle('Regression line')
                    # with c2:
                    #     exchange = st.toggle('Show distribution')
                    with c3:
                        EXCLUDE = st.toggle('EXCLUDE Selected points')
                    
                    # OLS Results
                    # df.drop(['Exluce/INCLUDE'], axis=1, inplace=True)
                    fig = px.scatter(st.session_state['corr_df'], x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], trendline='ols') 
                    res = px.get_trendline_results(fig)
                    results = res.iloc[0]['px_fit_results']
                    r2 = results.rsquared
                    ttest = results.tvalues[0]
                    ftest = results.f_pvalue
                    line = y_axis+' = '+ str(results.params[1]) + ' * ' + x_axis +' + '+ str(results.params[0])
                    pval = str(round(results.pvalues[1],3))
                    res_df = pd.DataFrame.from_dict({"R-Squared":[r2], 'Line equation':[line], "P-Value":[pval], "f-test":[ftest], "T-Test":[ttest] })
                    with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                    ):
                        st.subheader('OLS Results')
                    st.dataframe(res_df, hide_index=True, use_container_width=True)
                    # with stylable_container(
                    #     key='tab',
                    #     css_styles="""
                    #         table {
                    #             border: None !important;
                    #             margin-top:20px;
                    #             width:1400px;
                    #         }
                    #     """
                    # ):    
                    scatter_chart = interactive_plot(st.session_state['corr_df'], x_axis, y_axis, ols)
                    with stylable_container(
                        key='tab',
                        css_styles="""
                            {
                                margin-top:30px;   
                            }
                        """
                    ):
                        scatter_chart_selected = plotly_events(
                            scatter_chart,
                            select_event=True,
                        )
                    pntind = []
                    if scatter_chart_selected:
                        
                        for i in range(len(scatter_chart_selected)):
                            pntind.append(st.session_state['corr_df'][(st.session_state['corr_df'][x_axis] == scatter_chart_selected[i]['x']) & (st.session_state['corr_df'][y_axis] == scatter_chart_selected[i]['y'])].index[0])
                        if EXCLUDE:
                            st.session_state['EXCLUDE_df']['EXCLUDE/INCLUDE'].iloc[pntind] = False
                            st.session_state['filter_df'] = pd.concat([st.session_state['EXCLUDE_df'][st.session_state['EXCLUDE_df']['EXCLUDE/INCLUDE']==False], st.session_state['corr_df']])#[st.session_state['corr_df']['EXCLUDE/INCLUDE']==True]
                            st.session_state['filter_df'] = st.session_state['filter_df'].drop_duplicates(keep='first')
                            st.rerun()
                        else:
                            st.session_state['filter_df'] = st.session_state['corr_df'].iloc[pntind]
                        st.session_state['filter_df'].reset_index(inplace=True, drop=True)
                    # st.dataframe(st.session_state['filter_df'], use_container_width=True)
                        
        with t2:
            
            if st.session_state['prev_tab']=='Data Header':
                if st.session_state['data_header_df'] is None:
                    st.session_state['filter_df'] = df
                else:
                    st.session_state['filter_df'] = st.session_state['data_header_df']
                    
            elif st.session_state['prev_tab'] == 'Data Statistics':
                if st.session_state['histogram_df'] is None:
                    st.session_state['filter_df'] = st.session_state['data_header_df']
                else:
                    st.session_state['filter_df'] = st.session_state['histogram_df']
                    
            elif st.session_state['prev_tab']=='Corelation Matrix':
                if st.session_state['corr_df'] is None:
                    st.session_state['filter_df'] = st.session_state['histogram_df']
                else:
                    st.session_state['filter_df'] = st.session_state['corr_df']
                    
            elif st.session_state['prev_tab']=='Plot':
                    st.session_state['filter_df'] = st.session_state['corr_df']
                    
            st.title('AI Model Analysis')
            col4, col5, c6  = st.columns([1,1,0.2])
            col9, col10, col11, col12 = st.columns([0.4, 0.4, 0.3, 0.7])
            try:
                with col4.container(border=True):
                    indep_vars = st.multiselect('Independent Variables', st.session_state['filter_df'].columns.values[:-1], placeholder = "Choose an option")
                with col5.container(border=True):
                    dep_vars = st.multiselect('Dependent Variables', options=[x for x in st.session_state['filter_df'].columns.values[:-1] if x not in indep_vars], placeholder = "Choose an option", max_selections=1)
                unique_vals = len(np.unique(st.session_state['filter_df'][dep_vars]))
                if unique_vals<=10:  
                    with col4.container(border=True):
                        perf_reg = st.button('Classification')
                else:
                    with col4.container(border=True):
                        perf_reg = st.button('Regression')
                st.session_state.indep_vars, st.session_state.dep_vars = indep_vars, dep_vars
                if len(indep_vars)>=1 and len(dep_vars)>=1:
                    st.session_state['filter_df']=st.session_state['filter_df'][st.session_state.indep_vars + st.session_state.dep_vars]
                    # data = st.session_state['filter_df'][indep_vars+dep_vars]

                if perf_reg:
                    regre(st.session_state['filter_df'], st.session_state.indep_vars, st.session_state.dep_vars)
                    st.session_state.reg = True
                
            except AttributeError:
                pass
            except NameError:
                if len(indep_vars) == 0:
                    st.warning('Choose Independent variable')
                if len(dep_vars) == 0:
                    st.warning('Choose Dependent variable')
            
        with t3:
            model = st.session_state['model']
            try:
                dot_data = export_graphviz(model.estimators_[0], out_file=None,
                                feature_names=st.session_state['filter_df'].columns[:-1],
                                class_names=[str(i) for i in st.session_state['filter_df'][st.session_state['filter_df'].columns[-1]].unique()],
                                filled=True, rounded=True,
                                special_characters=True)
                dd_arr = []
                for i in range(len(dot_data)):
                    if dot_data[i:i+10] == 'fillcolor=':
                        dd_arr.append(dot_data[i+11:i+18])
                for i in dd_arr:
                    dot_data = dot_data.replace(i,'white')
                st.graphviz_chart(dot_data)
            except:
                pass
            


        with t4:
            if 'val' not in st.session_state:
                st.session_state['val'] = [st.session_state['filter_df'][i].min() for i in st.session_state['filter_df'].columns[1:]]
            # if st.session_state.reg:
            try:
                model = st.session_state['model']
                dot_data = export_graphviz(model.estimators_[0], out_file=None,
                                    feature_names=st.session_state['filter_df'].columns[:-1],
                                    class_names=[str(i) for i in st.session_state['filter_df'][st.session_state['filter_df'].columns[-1]].unique()],
                                    filled=True, rounded=True,
                                    special_characters=True)
                # try:
                col1, col2 = st.columns([1,7]) 
                with col1:
                    number_input_value = []
                    slider_val = []
                    for num,column in enumerate(st.session_state.indep_vars):
                        number_input_value.append(st.number_input(column, st.session_state['filter_df'][column].min(), st.session_state['filter_df'][column].max()))
                        slider_val.append(st.slider(column, st.session_state['filter_df'][column].min(), st.session_state['filter_df'][column].max(), value=number_input_value[num], label_visibility='collapsed'))
                    if st.session_state['checkpoint'] is None:
                        st.session_state['checkpoint'] = slider_val
                    # if st.button('Save Checkpoint'):
                    # st.session_state['checkpoint'] = slider_val
                with col2:
                    dd_arr = []
                    for i in range(len(dot_data)):
                        if dot_data[i:i+10] == 'fillcolor=':
                            dd_arr.append(dot_data[i+11:i+18])
                    for i in dd_arr:
                        dot_data = dot_data.replace(i,'white')
                    samples = pd.DataFrame(data=[slider_val], columns=st.session_state['filter_df'].columns[:-1])
                    decision_paths = model.estimators_[0].decision_path(samples).toarray()[0]
                    string1 = dot_data.splitlines()
                    str1 = []
                    str2 = []
                    samples1 = pd.DataFrame(data=[st.session_state['checkpoint']], columns=st.session_state['filter_df'].columns[:-1])
                    with stylable_container(
                            key='h3',
                            css_styles="""
                                h3 {
                                    font-size: 16px;
                                }
                            """
                        ):
                        st.subheader('Initial Point')
                    st.dataframe(samples1, hide_index=True)
                    with stylable_container(
                            key='h3',
                            css_styles="""
                                h3 {
                                    font-size: 16px;
                                }
                            """
                        ):
                        st.subheader("Tree for {st.session_state.dep_vars}")
                    decision_paths1 = model.estimators_[0].decision_path(samples1).toarray()[0]
                    str3 = []
                    str4 = []
                    for n,dec in enumerate(decision_paths):
                    # st.write(dot_data.splitlines()[20])
                        if dec == 1 and n < 10:
                            for i in range(len(string1)):
                                if string1[i][:3] == f"{n} [":
                                    str1.append(string1[i])
                                    string1[i] = string1[i].replace('white', '#90EE90')
                                    str2.append(string1[i])
                        elif dec == 1 and n >= 10:
                            for i in range(len(string1)):
                                if string1[i][:4] == f"{n} [":
                                    str1.append(string1[i])
                                    string1[i] = string1[i].replace('white', '#90EE90')
                                    str2.append(string1[i])
                    for i in range(len(str1)):
                        dot_data = dot_data.replace(str1[i],str2[i])
    
                    for n,dec in enumerate(decision_paths1):
                        # st.write(dot_data.splitlines()[20])
                        if dec == 1 and n < 10:
                            for i in range(len(string1)):
                                if string1[i][:3] == f"{n} [":
                                    str3.append(string1[i])
                                    if 'white' in string1[i]:
                                        string1[i] = string1[i].replace('white', '#FF7F7F')
                                    if '#90EE90' in string1[i]:
                                        string1[i] = string1[i].replace('#90EE90', '#FF7F7F')
                                    str4.append(string1[i])
                        elif dec == 1 and n >= 10:
                            for i in range(len(string1)):
                                if string1[i][:4] == f"{n} [":
                                    str3.append(string1[i])
                                    if 'white' in string1[i]:
                                        string1[i] = string1[i].replace('white', '#FF7F7F')
                                    if '#90EE90' in string1[i]:
                                        string1[i] = string1[i].replace('#90EE90', '#FF7F7F')
                                    str4.append(string1[i])
                    for i in range(len(str3)):
                        dot_data = dot_data.replace(str3[i],str4[i])
                    st.graphviz_chart(dot_data)
                    for i in range(len(decision_paths)):
                        if decision_paths[len(decision_paths)-1-i]==1:
                            last_node = len(decision_paths)-1-i
                            break
                    for i in range(len(string1)):
                            if last_node<10 and string1[i][:3] == f"{last_node} [":
                                if st.session_state['model_type'] == 'regression':
                                    pred = string1[i].split('<br/>')[-1][:11]
                                else:
                                    pred = string1[i].split('<br/>')[-1][:9]
                            elif last_node>=10 and string1[i][:4] == f"{last_node} [":
                                if st.session_state['model_type'] == 'regression':
                                    pred = string1[i].split('<br/>')[-1][:11]
                                else:
                                    pred = string1[i].split('<br/>')[-1][:9]
                    with col1:
                         with stylable_container(
                                key='h3',
                                css_styles="""
                                    h3 {
                                        font-size: 16px;
                                    }
                                """
                            ):
                            st.subheader("Predictions")
                            st.info(f'Predicted {st.session_state.dep_vars} {pred}')
            except:
                pass
            
            
                
                        
            #     except AttributeError:
            #         st.warning('Please select your Independent and Dependent variables')
            #     except KeyError as e:
            #         st.error(e)
            #     except ValueError:
            #         st.warning('Please select your Independent and Dependent variables')
            # else:
            #     st.warning(f"Please run the AI model and the choose the predictor")
            
