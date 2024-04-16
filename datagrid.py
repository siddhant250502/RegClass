import streamlit as st
import pandas as pd
import os
import statistics
import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import plotly.graph_objects as go
import time
from sklearn.decomposition import PCA
import numpy as np
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
import pickle
from streamlit_option_menu import option_menu
import toml
import math
from streamlit_extras.stylable_container import stylable_container

st.set_page_config(page_title="AI Model Analysis",layout="wide")
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
with open('.streamlit/config.toml', 'r') as f:
    config = toml.load(f)

if "page" not in st.session_state:
    st.session_state.page = 0
if 'file_path' not in st.session_state:
    st.session_state['file_path'] = ' '

if 'reg' not in st.session_state:
    st.session_state['reg'] = False

if 'filter_df' not in st.session_state:
    st.session_state['filter_df'] = None

if 'col_name' not in st.session_state:
    st.session_state['col_name'] = set()
    
def nextpage(): st.session_state.page += 1
def restart(): st.session_state.page = 0
def back(): 
    st.session_state.page -= 1 
    st.session_state.button_clicked = None
placeholder = st.empty()

def data_cleaning(data1):
    #Format columns
    for i in data1.columns:
        if type(data1[i]) == str:
            data1[i] = data1[i].str.replace(",","")
    #Removing columns with string data
    dt = []
    for num,i in enumerate(data1.dtypes):
        if i != "int64" and i != "float64":
            dt.append(num)
    data1.drop(data1.columns[dt],axis=1,inplace=True)
    
    #dropping columns with more than 25% of null values
    for i in data1.columns:
        if data1[f'{i}'].isna().sum() >= 0.25*(len(data1[f'{i}'])):
            data1.drop(i, inplace=True, axis=1) 
    
    #dropping colummns with one uniques value
    for i in data1.columns:
        if len(data1[f'{i}'].unique()) <= 1:
            data1.drop(i, inplace=True, axis=1) 
        elif len(data1[f'{i}'].unique())==2  and True in np.isnan(data1[f'{i}'].unique()):
            data1.drop(i, inplace=True, axis=1) 
            
    #Treating Nan values
    for i in data1.columns:
        data1[i].fillna(data1[i].median(), inplace=True)
    return data1

def interactive_plot(df, x_axis, y_axis, ols, ewm, exchange):
    try:
        if x_axis and y_axis:
            if exchange:
                if ols:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], marginal_x='histogram', marginal_y='box', trendline='ols', trendline_color_override='red', height=450) # trendline='ewm',trendline_options=dict(halflife=2),
                if not ols and ewm:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], marginal_x='histogram', marginal_y='box', trendline='ewm',trendline_options=dict(halflife=2), trendline_color_override='red', height=450)
                elif not ewm and not ols:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], marginal_x='histogram', marginal_y='box', height=450)
            else:
                if ols:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], marginal_x='box', marginal_y='histogram', trendline='ols', trendline_color_override='red') # trendline='ewm',trendline_options=dict(halflife=2),
                if not ols and ewm:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], marginal_x='box', marginal_y='histogram', trendline='ewm',trendline_options=dict(halflife=2), trendline_color_override='red')
                elif not ewm and not ols:
                    fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=['#30B9EF'], marginal_x='box', marginal_y='histogram')
        fig.update_layout(title_text='Column Vs Column Scatter Plot')
        return fig
    except:
        st.warning('Please select the values')
        
def plot_columns(df, opt, sigma):
    mean = statistics.mean(df[opt])
    stddev = statistics.stdev(df[opt])
    one_sigma_plus = mean+1*stddev
    one_sigma_minus = mean-1*stddev
    two_sigma_plus = mean+2*stddev
    two_sigma_minus = mean-2*stddev
    three_sigma_plus = mean+3*stddev
    three_sigma_minus = mean-3*stddev
    bin_width= 50
    nbins = math.ceil((df[opt].max() - df[opt].min()) / bin_width)
    fig = px.histogram(df, x=opt, color_discrete_sequence=['#30B9EF'], orientation='v', nbins=nbins)
    fig.update_traces(xbins_size=go.histogram.XBins(size=10))
    if sigma:
        fig.add_vline(mean, line_dash="dash",annotation_text="Mean", annotation_position="top right")
        fig.add_vrect(x0=one_sigma_minus, x1=one_sigma_plus, annotation_text='1-sigma', fillcolor="#00FF00", opacity=0.2, line_width=0)
        # fig.add_vrect(x0=one_sigma_plus, x1=mean, annotation_text='one-sigma', fillcolor="green", opacity=0.25)
        fig.add_vrect(x0=two_sigma_minus, x1=one_sigma_minus, fillcolor="#FFFF00", opacity=0.2, line_width=0)
        fig.add_vrect(x0=two_sigma_plus, x1=one_sigma_plus, annotation_text='2-sigma', fillcolor="#FFFF00", opacity=0.2, line_width=0)
        fig.add_vrect(x0=three_sigma_minus, x1=two_sigma_minus, fillcolor="purple", opacity=0.2, line_width=0)
        fig.add_vrect(x0=three_sigma_plus, x1=two_sigma_plus, annotation_text='3-sigma', fillcolor="#A020F0", opacity=0.2, line_width=0)
    fig.update_layout(bargap=0.1)
    return fig
    
    
def correlation(df):
    fig = px.imshow(df.corr(), text_auto=True, aspect='auto', color_continuous_scale='RdBu_r')
    return fig
    
def check_mc(X):
    X_mean = X.mean()
    X_std = X.std() 
    Z = (X-X_mean)/X_std
    c = Z.cov()
    eigenvalues, eigenvectors = np.linalg.eig(c)
    idx = eigenvalues.argsort()[::-1] 
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    explained_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.argmax(explained_var >= 0.5)
    st.write(Z)
    pca = PCA(n_components=n_components)
    pca.fit(Z)
    x_pca = pca.transform(Z)
    
    X = pd.DataFrame(x_pca,
                     columns = ['PC{}'.format(i+1) for i in range(n_components)])
    return X

def regre(df):
    start = time.time()
    df = data_cleaning(df)
    X = df.drop([df.columns[-1]], axis=1)
    y = df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    if len(y.unique()) >= 10:
        rfr = RandomForestRegressor()
        try:
            rfr.fit(X_train, y_train)
        except ValueError:
            st.warning('Columns not proper')
        try:
            importances = rfr.feature_importances_
            forest_importances = pd.Series(importances, index=X.columns)
            forest_importances = forest_importances.sort_values(ascending=False)
            with st.container(border=True):
                st.info("**Visualization of how each Independent variables Impact the Dependent variables**")
                st.bar_chart(forest_importances, color='#30B9EF')
        except:
            pass
        with open('model.pkl', 'wb') as f:
            pickle.dump(rfr, f)
        try:
            y_pred = rfr.predict(X_test)
        except:
            st.warning("Can't predict")
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
        rfc = RandomForestClassifier()
        try:
            rfc.fit(X_train, y_train)
        except ValueError:
            st.warning('Columns not proper')
        try:
            importances = rfc.feature_importances_
            forest_importances = pd.Series(importances, index=X.columns)
            forest_importances = forest_importances.sort_values(ascending=False)
            with st.container(border=True):
                st.info("**Visualization of how each Independent variables Impact the Dependent variables**")
                st.bar_chart(forest_importances, color='#1168f5')
        except:
            pass
        with open('model.pkl', 'wb') as f:
            pickle.dump(rfc, f)
        try:
            y_pred = rfc.predict(X_test)
        except:
            st.warning("Can't predict due to some error in the Dependant/Independant Variable")
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
            st.write("\n\n")
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
            'Timestamp': [datetime.datetime.fromtimestamp(os.path.getatime(os.path.join(cwd, f))) for f in documents]
    })

    res = st.data_editor(data,
                        column_config={"Select": st.column_config.CheckboxColumn(default=False), 
                                        "File Name": st.column_config.Column(width="large"), 
                                        "Timestamp": st.column_config.Column(width="large")},
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
                color: black ;
                text-decoration: none;
                cursor: pointer;
                border: none !important;
                }
            button:hover {
                color: 30B9EF ;
                border: solid 1px black;
            }
            button:focus {
                outline: 1px black !important;
                color: 30B9EF !important;
                border: solid 1px black;
            }
        """
    ):
        st.button('Back to Datasets', on_click=back)

    with st.container(border=True):
        if st.session_state['file_path'] is not None:
            try:
                df = pd.read_csv(st.session_state['file_path'], delimiter=',')
            except:
                df = pd.read_csv(st.session_state['file_path'], delimiter=';')
        else:
            st.warning("File type Not supported")
        df = data_cleaning(df)
        df['Exclude/Include'] = True
        


            
        col1, col2, col3 = st.columns([0.1,0.1,0.5])
 
        
        option = option_menu(
                menu_title=None,
                options=['Dataset Info', 'AI Model', 'Predictor'],
                icons=['1-square', '2-square', '3-square'],
                menu_icon=None,
                default_index=0,
                orientation='horizontal'
            )
                   
        # Update content based on button click
        if option=='Dataset Info':
            with st.container(border=True):
                option = option_menu(
                menu_title=None,
                options=['Data Header', 'Data Statistics', 'Plot', 'Correlation Matrix'],
                icons=['table', 'bar-chart-line', 'graph-up', 'diagram-2'],
                menu_icon=None,
                default_index=0,
                orientation='horizontal'
            )
                if option=='Data Header':
                    with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                    ):
                        st.subheader('Header of the DataSet')
                    st.session_state['filter_df'] = st.data_editor(
                        df,
                        column_config={
                            "Exclude/Include": st.column_config.CheckboxColumn(default=True)
                        },
                        use_container_width=True,
                        hide_index=True, 
                        height=350,
                    )
                    st.session_state['filter_df'] = st.session_state['filter_df'][st.session_state['filter_df']['Exclude/Include']==True]
                
                elif option == 'Data Statistics':
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
                        st.dataframe(st.session_state['filter_df'].describe(), use_container_width=True)
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
                    opt = st.selectbox('Select any Column', options=df.columns)
                    sigma = st.checkbox('Show Sigma areas')
                    if opt:
                        bar_chart = plot_columns(st.session_state['filter_df'], opt, sigma)
                        bar_chart_selected = plotly_events(
                            bar_chart,
                            select_event=True
                        )
                    # pntind = []
                    new_df = pd.DataFrame(None, columns=[st.session_state['filter_df'].columns])
                    if bar_chart_selected:
                        for i in range(len(bar_chart_selected)):
                            min = bar_chart_selected[i]['x']-0.5
                            max = bar_chart_selected[i]['x']+0.4
                            temp_df = st.session_state['filter_df'][st.session_state['filter_df'][opt]>=min]
                            temp_df = temp_df[temp_df[opt]<=max]
                            new_df = pd.concat([temp_df, new_df])
                            new_df = new_df[new_df.columns[:len(st.session_state['filter_df'].columns)]]
                        st.subheader('Filtered Dataset Preview')   
                        st.dataframe(new_df, use_container_width=True, hide_index=True)     
                        
                elif option=='Plot':    
                    with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                    ):
                        st.subheader('Scatter Plot')
                    x_axis = st.selectbox('X-axis', options=st.session_state['filter_df'].columns)
                    y_axis = st.selectbox('Y-axis', options=st.session_state['filter_df'].columns)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        ols = st.checkbox('Least-Squared line')
                    with c2:
                        ewm = st.checkbox('Estimated Weighted Mean Line')
                    with c3:
                        exchange = st.checkbox('Inter-Change axes')
                    scatter_chart = interactive_plot(st.session_state['filter_df'], x_axis, y_axis, ols, ewm, exchange)
                    scatter_chart_selected = plotly_events(
                        scatter_chart,
                        select_event=True,
                    )
                    pntind = []
                    if scatter_chart_selected:
                        for i in range(len(scatter_chart_selected)):
                            pntind.append(scatter_chart_selected[i]['pointIndex'])
                        with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                        ):
                            st.subheader('Filtered Dataset Preview')
                        # st.session_state['filter_df']['Exclude/Include'][pntind] = True
                        st.write(st.session_state['filter_df'].iloc[pntind])
                        
                    
                elif option=='Correlation Matrix':
                    with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                    ):
                        st.subheader('Correlation Matrix Heatmap')
                    heatmap = correlation(st.session_state['filter_df'].drop(['Exclude/Include'], axis=1))
                    heatmap_selected = plotly_events(
                        heatmap,
                        select_event=True
                    )
                    
                    if heatmap_selected:
                        st.session_state['col_name'].add(heatmap_selected[0]['x'])
                        st.session_state['col_name'].add(heatmap_selected[0]['y'])
                    cols = list(st.session_state['col_name'])
                    if st.button('Preview'):
                        with stylable_container(
                        key='h3',
                        css_styles="""
                            h3 {
                                font-size: 16px;
                            }
                        """
                        ):
                            st.subheader('Filtered Dataset Preview')
                        st.write(st.session_state['filter_df'][cols])
                
                        
        elif option=='AI Model':
            st.title('AI Model Analysis')
            col4, col5  = st.columns([1,1])
            col9, col10, col11, col12 = st.columns([0.4, 0.4, 0.3, 0.7])
            # try:
            with col4.container(border=True):
                    indep_vars = st.multiselect('Independent Variables', df.columns.values, placeholder = "Choose an option")
            with col5.container(border=True):
                dep_vars = st.multiselect('Dependent Variables', options=[x for x in df.columns.values if x not in indep_vars], placeholder = "Choose an option", max_selections=1)
            unique_vals = len(np.unique(df[dep_vars]))
            if unique_vals<=7:
                with col9:
                    perf_reg = st.button('Classification')
            else:
                with col9:
                    perf_reg = st.button('Regression')
            st.session_state.indep_vars, st.session_state.dep_vars = indep_vars, dep_vars
            if len(indep_vars)>=1 and len(dep_vars)>=1:
                data = df[indep_vars+dep_vars]
            if perf_reg:
                regre(data)
                st.session_state.reg = True
            # except NameError:
            #     if len(indep_vars) == 0:
            #         st.warning('Choose Independent variable')
            #     if len(dep_vars) == 0:
            #         st.warning('Choose Dependent variable')
                    
        elif option=='Predictor':
            if st.session_state.reg:
                try:
                    with st.container(border=True):
                        slider_val = []
                        for i in st.session_state.indep_vars:
                            slider_val.append(st.slider(i, float(df[i].min()), float(df[i].max()), float(df[i].mean())))
                    
                    model = pickle.load(open('model.pkl', 'rb'))
                    pred = model.predict([slider_val])
                    
                    num_steps = 100  
                    colors = []
                    pred_dict = {}
                    val = pred
                    min = float(df[st.session_state.dep_vars].min())
                    maxim = float(df[st.session_state.dep_vars].max())
                    steps = (min+maxim)/100
                    for i in range(num_steps):
                        red = 255.0 if i < num_steps / 2 else 1.0 - (2.0 * (i - num_steps / 2) / num_steps)
                        green = 1.0 - abs(i - num_steps / 2) / num_steps
                        blue = 1.0 if i >= num_steps / 2 else 1.0 - (2.0 * (num_steps / 2 - i) / num_steps)
                        colors.append((red, green, blue))
                    for i in range(num_steps):
                        if min<=val and min+steps>=val:
                            pred_dict[str(min)] = (0,0,0)
                            pred_dict[str(min+steps)] = (0,0,0)
                        else:
                            pred_dict[str(min)] = colors[i]
                        min+=steps
                    color_pal = pred_dict.values()
                    plt.figure(figsize=(10, 2))
                    plt.imshow([list(color_pal)], extent=[0, num_steps, 0, 1])
                    plt.axis('off')
                    with st.container(border=True):
                        st.write('Predicted Value')
                        st.pyplot(plt, use_container_width=True)
                        st.slider('', float(df[st.session_state.dep_vars].min()), float(df[st.session_state.dep_vars].max()), float(pred[0]), disabled=True)
                        st.info(f'Predicted value is {float(pred[0])}')
                    

                except AttributeError:
                    st.warning('Please select your Independent and Dependent variables')
                except KeyError:
                    pass
                except ValueError:
                    st.warning('Please select your Independent and Dependent variables')
            else:
                st.warning(f"Please run the AI model and the choose the predictor")
            
