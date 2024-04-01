import streamlit as st
import pandas as pd
import os
import datetime
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import plotly.graph_objects as go
import time
from sklearn.decomposition import PCA
import numpy as np
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="AI Model Analysis",layout="wide")
# with open('style.scss') as f:
#     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
        
if "page" not in st.session_state:
    st.session_state.page = 0
if 'file_path' not in st.session_state:
    st.session_state['file_path'] = ' '
if 'upload_file' not in st.session_state:
    st.session_state['upload_file'] = None
if 'reg' not in st.session_state:
    st.session_state['reg'] = False
    
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
        rfr.fit(X_train, y_train)
        try:
            importances = rfr.feature_importances_
            forest_importances = pd.Series(importances, index=X.columns)
            forest_importances = forest_importances.sort_values(ascending=False)
            with st.container(border=True):
                st.info("**Visualization of how each Independent variables Impact the Dependent variables**")
                st.bar_chart(forest_importances, color='#1168f5')
        except:
            pass
        with open('model.pkl', 'wb') as f:
            pickle.dump(rfr, f)
        y_pred = rfr.predict(X_test)
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
            num_steps = 100  
            colors = []
            for i in range(num_steps):
                red = 1.0 if i < num_steps / 2 else 1.0 - (2.0 * (i - num_steps / 2) / num_steps)
                green = 1.0 - abs(i - num_steps / 2) / num_steps
                blue = 1.0 if i >= num_steps / 2 else 1.0 - (2.0 * (num_steps / 2 - i) / num_steps)
                colors.append((red, green, blue))
            colors[int(acc)] = (0,0,0)
            plt.figure(figsize=(30, 12))
            plt.imshow([colors], extent=[0, num_steps, 0, 1])
            plt.axis('off')
            # st.write('**Accuracy**')
            # st.pyplot(plt, use_container_width=True)
            # st.write("**Error Margin**")
            # st.slider("",y.min(),y.max(),rmse, disabled=True)
            
            
    else:
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
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
        y_pred = rfc.predict(X_test)
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
            # st.write('Accuracy of Model')
            # num_steps = 100  
            # colors = []
            # for i in range(num_steps):
            #     red = 1.0 if i < num_steps / 2 else 1.0 - (2.0 * (i - num_steps / 2) / num_steps)
            #     green = 1.0 - abs(i - num_steps / 2) / num_steps
            #     blue = 1.0 if i >= num_steps / 2 else 1.0 - (2.0 * (num_steps / 2 - i) / num_steps)
            #     colors.append((red, green, blue))
            # colors[round(accuracy_score(y_test,y_pred)*100)] = (0,0,0)
            # plt.figure(figsize=(10, 2))
            # plt.imshow([colors], extent=[0, num_steps, 0, 1])
            # plt.axis('off')
            # st.pyplot(plt, use_container_width=True)
            # st.slider("",0.0,100.0,acc*100, disabled=True)
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
    fig.add_trace(go.Scatter(x=[i for i in range(50)], y=y_pred[-50:], mode='lines', name='Regression Line',opacity=0.5, line=dict(color="blue")))
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
                        use_container_width=True
                        )
    st.session_state['file_path'] = res.loc[res.Select.idxmax()]['File Name']
    st.session_state['upload_file'] = st.file_uploader("Upload your dataset", type=['csv'])
    
    if len(res[res.Select == True])==1 or st.session_state['upload_file'] is not None:
        col7, col8 = st.columns([0.1,1])
        with col8:
            st.button("Next", on_click=nextpage, disabled=(st.session_state.page > 1))
        with col7:
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
    st.button('Back to Datasets', on_click=back)
    with st.container(border=True):
        if (st.session_state['file_path'] is not None and st.session_state['file_path'][-3:]=="csv") or (st.session_state['upload_file'] is not None and (st.session_state['upload_file'].name)[-3:]=="csv"):
            if st.session_state['file_path'] is not None:
                try:
                    df = pd.read_csv(st.session_state['file_path'], delimiter=',')
                except:
                    df = pd.read_csv(st.session_state['file_path'], delimiter=';')
            elif st.session_state['upload_file'] is not None:
                try:
                    df = pd.read_csv(st.session_state['upload_file'], delimiter=',')
                except:
                    df = pd.read_csv(st.session_state['upload_file'], delimiter=';')
        else:
            st.warning("File type Not supported")
        df = data_cleaning(df)
        y = df[df.columns[-1]]
        unique_vals = len(y.unique())
        
        # if unique_vals>=7:
        #     m = 'Regression'
        # else:
        #     m = 'Classification'
        st.title('AI Model Analysis')
        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = None
            
        col1, col2, col3 = st.columns([0.1,0.1,0.5])
        col4, col5  = st.columns([1,1])
        col9, col10, col11, col12 = st.columns([0.4, 0.4, 0.3, 0.7])
        # Define the buttons
        
        with col1:
            if st.button('Dataset Info'):
                st.session_state.button_clicked = 1
        with col2:
            if st.button('AI Model'):
                st.session_state.button_clicked = 2
        with col3:
            if st.button('Predictor'):
                st.session_state.button_clicked = 3
                
                
        # Update content based on button click
        if st.session_state.button_clicked == 1:
            with st.container(border=True):
                pr = ProfileReport(df)
                st_profile_report(pr)
            
        elif st.session_state.button_clicked == 2:
            try:
                with col4.container(border=True):
                        indep_vars = st.multiselect('Independent Variables', df.columns.values, placeholder = "Choose an option")
                with col5.container(border=True):
                    dep_vars = st.multiselect('Dependent Variables', options=[x for x in df.columns.values if x not in indep_vars], placeholder = "Choose an option", max_selections=1)

                with col9:
                    perf_reg = st.button(f'Perform AI Model')
                st.session_state.indep_vars, st.session_state.dep_vars = indep_vars, dep_vars
                if len(indep_vars)>=1 and len(dep_vars)>=1:
                    data = df[indep_vars+dep_vars]
                if perf_reg:
                    regre(data)
                    st.session_state.reg = True
            except NameError:
                if len(indep_vars) == 0:
                    st.warning('Choose Independent variable')
                if len(dep_vars) == 0:
                    st.warning('Choose Dependent variable')
                    
        elif st.session_state.button_clicked == 3:
            if st.session_state.reg:
                try:
                    with st.container(border=True):
                        slider_val = []
                        for i in st.session_state.indep_vars:
                            slider_val.append(st.slider(i, float(df[i].min()), float(df[i].max()), float(df[i].mean())))
                    
                    model = pickle.load(open('model.pkl', 'rb'))
                    pred = model.predict([slider_val])
                    
                    # st.write('---')
                    
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
            
