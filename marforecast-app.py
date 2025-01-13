#import scipy.stats as stats
import streamlit as st
import pandas as pd
import numpy as np

import streamfct as stf
import stablemar as mar

# Block comment(uncomment) : CTRL + K + C (U)
# To execute the .py use the TERMINAL of VSC to enter : "streamlit run marforecast-app.py"

st.session_state = stf.initstate()

#### Data Section
st.header('Forecasting extreme trajectories', divider='rainbow')
st.subheader('with semi-norm representation of stable MAR')

replication = '''**Replication :**
    To replicate the results presented in the paper "Forecasting extreme trajectories
    using seminorm representations" (de Truchis, Fries and Thomas 2025) 
    select "Climate Data" and the "SOI" series.
    For example, set the in-sample date to December 1, 1991, and the number of 
    forward lags to 2. Then, click "Estimate." Note that the estimation process may 
    converge to a local optimum with ill-located roots. If this occurs, repeat the
    estimation until the green convergence message appears. To configure the pattern
    recognition, first enable the toggle to invert the pattern, as we 
    anticipate an El Niño anomaly. Then, click "Search $k_0$". Next, set $k_0 = 1$, 
    $m_0 = 10$, and $h = 1$. Finally, click "Forecast" to reproduce the Figure 5.
        '''
st.caption(replication)

st.subheader('Data selection', divider='red')

# Data selection
option = st.selectbox(
    'Select the database you want to explore',
    ('Climate data', 'Financial data', 'Artificial data', 'Crypto'))

data, dates = stf.dataselection(option)
    
# Data visualization
assetName = st.selectbox('Select a time series (demeaned)',data.columns[1:].tolist())

df = pd.DataFrame(data[assetName])
df['Dates'] = pd.to_datetime(data['Dates'])
df.set_index('Dates', inplace=True)

# eqmax = round(float(1-stats.levy_stable.cdf(dfInS.max(),alpha, beta)),2)
qx = st.slider('Select the lower/upper quantile you want to display for $X_{t}$', 0.50, 0.99, 0.95)
st.write('You selected:', round(1-qx,2), 'and', qx)
qdf = df[assetName].quantile([qx, 1-qx])
df['UpQ'] = qdf[qx]
df['DwQ'] = qdf[1-qx]
st.line_chart(df)
df = df.drop(['UpQ', 'DwQ'], axis=1)


#### Estimation Section
st.subheader('Estimation', divider='orange')


# Define In-Sample vs Out-of-Sample
oos = st.date_input("Define when the Out-of-Sample starts. The estimation will be performed in sample.",
                    dates.iloc[-1],min_value=dates[104],max_value=dates.iloc[-1])
    
dfInS, dfInSidx, isback = stf.sampleselection(oos, df, dates)


# Define model specification
modsel = st.radio("Select a model (MARST and Convoluted MAR are under development)", ["MAR(r,s)"], 
                captions=["GCov (Gourieroux and Jasiak 2023) and SCF regression (Nolan 2020)"])

    
order, modname = stf.modelspecification(modsel)

r, s = order
    
model = mar.stablemar(order, model = modname)

# Model estimation
icol, ecol = st.columns(2)

with icol: 
    st.markdown("Starting values are randomly selected")
    init, _, inipar = model.rndinit()
    inidf = pd.DataFrame(
        inipar, 
        columns = ["Coefficients", "Valeurs initiales"])

    st.table(inidf)
    
    est_button = st.button('Estimate')

estholder = st.empty()
with ecol:
    st.markdown("Pay attention to local optima, perform multiple run")
    
    if st.session_state.thetahat is not None:
        estdf = pd.DataFrame(st.session_state.thetahat, columns = ["Estimates"])
        estholder.table(estdf)
        
    
    
if est_button:
    if st.session_state.thetahat is not None:
        estholder.empty()
        
    st.session_state.thetahat = stf.estspinner(model, init, modname, df, ecol)
    
    model.par = st.session_state.thetahat
    
    stf.drawdeltas(order, st.session_state.thetahat, assetName)
        
        
#### Pattern recognition Section   
st.subheader('Pattern recognition', divider='green')    

# Pattern recognition settings   
col1, col2, col3 = st.columns(3)

with col1:
    on = st.toggle('Flip $d_k$ upside down')
    if on:
        vartheta = -1
        st.write('You anticipate a negative bubble')
    else:
        vartheta = 1
        st.write('You anticipate a positive bubble')
        
with col2:
    mmax = st.slider('Select the parameter $m$', 1, 20, 10)
    st.write('You selected:', mmax)
    
with col3:
    borel = st.slider('Select the neighborhood $A$', -0.5, 0.5, (-0.1, 0.1))
    st.write('You selected:', borel)


# Pattern exploration  
kmax = 20

pat_button = st.button('Search $k_0$')
st.write('After running, choose $k_0$ and the corresponding $m$')

kholder = st.empty()
if st.session_state.khat is not None:
    kholder.write(st.session_state.khat)

if pat_button:
    
    if st.session_state.thetahat is not None:
        kholder.empty()
        
    model.par = st.session_state.thetahat
        
    k0 = np.full(mmax, np.nan)
    
    for j in range(mmax):
        k0[j], _, _, _, _, _ = model.pathfinder(dfInS, dfInSidx, j+1, kmax, vartheta)
    
    dfk = pd.DataFrame(k0)
    dfk.index = dfk.index + 1
    dfk = dfk.T
    dfk.index.name = 'm'
    dfk.rename(index={dfk.index[0]: 'k'}, inplace=True)
    st.session_state.khat = dfk
    kholder.write(st.session_state.khat) 
    k0mean = st.session_state.khat.mean(axis = 1)
    k0mean = round(k0mean.iloc[0], 2)
    st.write('Mean of $k_0$ sequence:', k0mean)
    

# Pattern selection
col5, col6, col7 = st.columns(3)

with col5:
    selk0 = st.slider('Select the parameter $k_0$', 1, kmax, 1)
    st.write('You selected:', selk0)

with col6:
    selm = st.slider('Select the corresponding $m$', 1, mmax, 1)
    st.write('You selected:', selm)
    
with col7:
    selh = st.slider('Select the forecast horizon $h^*$', 5, 10, 1)
    st.write('You selected:', selh)
    
    
#### Forecasting Section 
st.subheader('Predictions', divider='blue')
st.write('Depending on the fitted model, either Crash Dates or Crash Probabilities computation is available')

tab1, tab2 = st.tabs(["Crash Dates", "Crash probabilities"])

if s > 1:
    isfore = True
else:
    isfore = False

# Forecasting Crash Dates 
with tab1:
    
    h = selk0 + selh
    df_nan = pd.DataFrame(np.nan, index=range(h), columns=dfInS.columns)

    for_button = st.button('Forecast', disabled = not isfore)

    if for_button:

        model.par = st.session_state.thetahat
        xfore = model.forecast(dfInS[assetName], dfInSidx, selm, h, selk0, vartheta)
        dfpred = pd.concat([dfInS, df_nan], ignore_index=True)
        dfpred['Forecasts'] = xfore
        
        st.session_state.forecast = dfpred
        if isback:
            dfpred.index = df.index[:len(xfore)]
        
        st.line_chart(dfpred.tail(selm+h+50))
        

# Forecasting Crash Probabilities 
with tab2:

    col7, col8 = st.columns(2)

    with col7:
        maxp = st.number_input('Above this cutoff, the crash probability is considered as 1', 
                                step = 0.01, value = 0.9100, min_value = 0.5, max_value = 1.0, format='%0.5f')
        
        prob_button = st.button('Compute', disabled = isfore)
        
    with col8:
        help = '''**Need some help with interpretation ?**  
                    The forecast horizon iterates up to $h = k_0 + h^*$.  
                    *None* is reported until $h < k_0$.
                    One possible way to set the cutoff is to report
                    the probability to crash at $h = k_0 + 1$.
                '''
        st.markdown(help)
        
    if prob_button:   
        
        model.par = st.session_state.thetahat 
        proba, pfore = model.foreprob(dfInS[assetName], selk0, h, maxp, dfInSidx, vartheta)
        st.dataframe(proba, use_container_width = True, height=250)
        
        df_nan = pd.DataFrame(np.nan, index=range(h), columns=dfInS.columns)
        dfpred = pd.concat([dfInS, df_nan], ignore_index=True)
        dfpred['Forecasts'] = pfore
        if isback:
            dfpred.index = df.index[:len(pfore)]
            
        st.session_state.foreprob = dfpred
        
        # if nobacktest: newdates = pd.date_range(start=df.index[-1] + pd.DateOffset(weeks=1), periods=len(dftraj), freq='W')
        # dftraj.index = newdates

        st.line_chart(dfpred.tail(selm+h+50))


st.subheader('Backtesting', divider='violet')
st.write('This section is available only if your Out-of-Sample starts before the end of the database')

back_button = st.button('Backtest', disabled = not isback)
if back_button:
    if isback and isfore:
        
        dfback = df.merge(st.session_state.forecast['Forecasts'], on='Dates', how='left')
        dfmetrics = dfback.dropna()
        st.line_chart(dfback.iloc[dfInSidx-selm-h-80:dfInSidx+80, :2])   
        
    if isback and not isfore:

        dfback = df.merge(st.session_state.foreprob['Forecasts'], on='Dates', how='left')
        dfmetrics = dfback.dropna()
        st.line_chart(dfback.iloc[dfInSidx-selm-h-80:dfInSidx+80, :2])  
        
disclaimer = '''**Disclaimer :**
    This application aims to playfully illustrate the possible uses of the results from the paper
    *Forecasting extreme trajectories using seminorm representations*. The authors decline all
    responsibility for the use of this application and do not provide any support in case of
    technical problems insofar as it is not an econometrics software. The GCov has been adapted 
    to handle alpha-stable processes with K and H set to 2. Many tuning parameters are
    used in this application and are avalaible upon request.
        '''
st.caption(disclaimer)

# with tab1:
#       
#     ##
#     st.subheader('Backtesting', divider='violet')
#     st.write('This section is available only if your Out-of-Sample starts before the end of the database')

#     #
#     if but_for:
#         if not isBack:
#             mom1 = seminorm.MARCondExp(df.iloc[dfInSidx-selm:dfInSidx+h+1], phi_c, phi_nc, alpha, selm, selk0, dkfunction)
#             dfback = df.merge(dfpred['Forecasts'], on='Dates', how='left')
#             dfback = dfback.merge(mom1['M-Forecast'], on='Dates', how='left')
#             dfmetrics = dfback.dropna()
#             st.line_chart(dfback.iloc[dfInSidx-selm-h-100:dfInSidx+100, :2])
#             #
#             #mse = round(mean_squared_error(dfmetrics[assetName], dfmetrics['Forecasts']),2)
#             mae = round(mean_absolute_error(dfmetrics[assetName], dfmetrics['Forecasts']),2)
#             medae = round(median_absolute_error(dfmetrics[assetName], dfmetrics['Forecasts']),2)
#             #msebench = round(mean_squared_error(dfmetrics[assetName], dfmetrics['M-Forecast']),2)
#             maebench = round(mean_absolute_error(dfmetrics[assetName], dfmetrics['M-Forecast']),2)
#             medaebench = round(median_absolute_error(dfmetrics[assetName], dfmetrics['M-Forecast']),2)
#             st.write('Comparison of seminorm-based forecasts (black) with the forward conditional expectation (colored)')
#             met1, met2, met3 = st.columns(3)
#             #met1.metric("MSE", str(mse), msebench, delta_color="inverse")
#             met2.metric("MAE", str(mae), maebench, delta_color="inverse") 
#             met3.metric("MedAE", str(medae), medaebench, delta_color="inverse") 
#             st.dataframe(dfmetrics, use_container_width=True, height=250)
#         else:
#             st.write('**Your Out-of-Sample starts at the end of the database, backtesting is impossible**')

# with tab2:
#     #
#     but_prob = st.button('Compute', key='but_c', disabled=isfore)

#     #
#     col7, col8 = st.columns(2)

#     with col7:
#         maxh = st.slider('Select the max forecast horizon $h$', selk0, selk0+20, selk0+5)

#     with col8:
#         maxp = st.number_input('Above this cutoff, the crash probability is considered as 1', 
#                                step = 0.01, value = 0.9100, min_value = 0.5, max_value = 1.0, format='%0.5f')

#     #
#     if but_prob:    
#         proba, pfore = seminorm.probaMAR(dfInS[assetName], phi_c, phi_nc, alpha, selk0, maxh, maxp, dfInSidx, vartheta)
        
#         st.dataframe(proba, use_container_width=True, height=250)
        
#         df_nan = pd.DataFrame(np.nan, index=range(maxh), columns=dfInS.columns)
#         dfpred = pd.concat([dfInS, df_nan], ignore_index=True)
#         dfpred['Forecasts'] = pfore
#         if not isBack:
#             dfpred.index = df.index[:len(pfore)]
        
#         # if nobacktest: newdates = pd.date_range(start=df.index[-1] + pd.DateOffset(weeks=1), periods=len(dftraj), freq='W')
#         # dftraj.index = newdates

#         st.line_chart(dfpred.tail(selm+maxh+50))
        
#     ##
#     st.subheader('Backtesting', divider='violet')
#     st.write('This section is available only if your Out-of-Sample starts before the end of the database')

#     #
#     if but_prob:
#         if not isBack:
#             mom1 = seminorm.MARCondExp(df.iloc[dfInSidx:dfInSidx+maxh+1], phi_c, phi_nc, alpha, selm, selk0, dkfunction)
#             dfback = df.merge(dfpred['Forecasts'], on='Dates', how='left')
#             dfback = dfback.merge(mom1['M-Forecast'], on='Dates', how='left')
#             dfmetrics = dfback.dropna()
#             st.line_chart(dfback.iloc[dfInSidx-selm-h-100:dfInSidx+100, :2])
#             #
#             #mse = round(mean_squared_error(dfmetrics[assetName], dfmetrics['Forecasts']),2)
#             mae = round(mean_absolute_error(dfmetrics[assetName], dfmetrics['Forecasts']),2)
#             medae = round(median_absolute_error(dfmetrics[assetName], dfmetrics['Forecasts']),2)
#             #msebench = round(mean_squared_error(dfmetrics[assetName], dfmetrics['M-Forecast']),2)
#             maebench = round(mean_absolute_error(dfmetrics[assetName], dfmetrics['M-Forecast']),2)
#             medaebench = round(median_absolute_error(dfmetrics[assetName], dfmetrics['M-Forecast']),2)
#             st.write('Comparison of seminorm-based forecasts (black) with the forward conditional expectation (colored)')
#             met1, met2, met3 = st.columns(3)
#             #met1.metric("MSE", str(mse), msebench, delta_color="inverse")
#             met2.metric("MAE", str(mae), maebench, delta_color="inverse") 
#             met3.metric("MedAE", str(medae), medaebench, delta_color="inverse") 
#             st.dataframe(dfmetrics, use_container_width=True, height=250)
#         else:
#             st.write('**Your Out-of-Sample starts at the end of the database, backtesting is impossible**')
