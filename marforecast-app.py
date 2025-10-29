import streamlit as st
import pandas as pd
import numpy as np

import streamfct as stf
from stablemar import StableMAR

# Block comment(uncomment): CTRL + K + C (U)
# To execute the .py use the TERMINAL of VSC to enter: "streamlit run marforecast-app.py"

st.session_state = stf.initstate()

#### Data Section
st.header('Forecasting extreme trajectories', divider='rainbow')
st.subheader('with semi-norm representation of stable MAR')

replication = '''**Replication:**
    To replicate the results presented in the paper "Forecasting extreme trajectories
    using seminorm representations", select "Climate Data" and the "SOI" series.
    For example, set the in-sample date to December 1, 1991, and the number of 
    forward lags to 2. Then, click "Estimate." Note that the estimation process may 
    converge to a local optimum with ill-located roots. If this occurs, repeat the
    estimation until the green convergence message appears. To configure the pattern
    recognition, first enable the toggle to invert the pattern, as we 
    anticipate an El Niño anomaly. Then, click "Search $k_0$". Next, set $k_0 = 1$, 
    $m_0 = 10$, and $h = 1$. Finally, click "Forecast" to reproduce Figure 5.
        '''
st.caption(replication)

st.subheader('Data selection', divider='red')

# Data selection
option = st.selectbox(
    'Select the database you want to explore',
    ('Climate data', 'FRED Macro data', 'Financial data', 'Artificial data', 'Crypto data'))

data, dates = stf.dataselection(option)
    
# Data visualization
assetName = st.selectbox('Select a time series (demeaned)', data.columns[1:].tolist())

df = pd.DataFrame(data[assetName])
df['Dates'] = pd.to_datetime(data['Dates'])
df.set_index('Dates', inplace=True)

qx = st.slider('Select the lower/upper quantile you want to display for $X_{t}$', 0.50, 0.99, 0.95)
st.write('You selected:', round(1-qx, 2), 'and', qx)
qdf = df[assetName].quantile([qx, 1-qx])
df['UpQ'] = qdf[qx]
df['DwQ'] = qdf[1-qx]
st.line_chart(df)
df = df.drop(['UpQ', 'DwQ'], axis=1)


#### Estimation Section
st.subheader('Estimation', divider='orange')

# Define In-Sample vs Out-of-Sample
oos = st.date_input("Define when the Out-of-Sample starts. The estimation will be performed in sample.",
                    dates.iloc[-1].date(), min_value=dates[104].date(), max_value=dates.iloc[-1].date())

if option == 'Ocean data':
    oos = pd.to_datetime(f"{oos} 00:10:00")
    
dfInS, dfInSidx, isback = stf.sampleselection(oos, df, dates)

# Define model specification
modsel = st.radio("Select a model (MARST and Convoluted MAR are under development)", 
                  ["MAR(r,s)"], 
                  captions=["GCoV (Gourieroux and Jasiak 2023) and SCF regression (Nolan 2020)"])

order, modname = stf.modelspecification(modsel)

r, s = order

# Check if model order changed and reset if needed
stf.check_and_reset_if_order_changed(order)
    
# Create model instance
model = StableMAR(order)

# Model estimation
icol, ecol = st.columns(2)

with icol: 
    st.markdown("Starting values are randomly selected")
    init = model.generate_initial_guess(random=True)
    
    # Create display names for parameters
    param_names = []
    for i in range(r):
        param_names.append(f"psi_{i+1}")
    for i in range(s):
        param_names.append(f"phi_{i+1}")
    
    # Add stable parameter names and initial values
    param_names.extend(["alpha", "beta", "sigma"])
    
    # Generate initial values for stable parameters
    # These are just for display; actual estimation is done in two stages
    stable_init = [1.5, 0.0, 1.0]  # Default initial values for alpha, beta, sigma
    init_full = init + stable_init
    
    inidf = pd.DataFrame({
        "Coefficients": param_names,
        "Initial values": init_full
    })

    st.table(inidf)
    
    est_button = st.button('Estimate')

estholder = st.empty()
with ecol:
    st.markdown("Pay attention to local optima, perform multiple runs")
    
    if st.session_state.thetahat is not None:
        # Create parameter names
        param_names = []
        for i in range(r):
            param_names.append(f"psi_{i+1}")
        for i in range(s):
            param_names.append(f"phi_{i+1}")
        
        # Add stable parameter names
        param_names.extend(["alpha", "beta", "sigma"])
        
        # Check if the stored parameters match the current model order
        expected_length = r + s + 3  # MAR params + stable params
        if len(st.session_state.thetahat) == expected_length:
            estdf = pd.DataFrame({
                "Parameter": param_names,
                "Estimates": st.session_state.thetahat
            })
            estholder.table(estdf)
        else:
            # Model order changed, reset estimates
            estholder.empty()
            st.info("Model order changed. Please re-estimate.")
            st.session_state.thetahat = None
        
if est_button:
    if st.session_state.thetahat is not None:
        estholder.empty()
        
    st.session_state.thetahat = stf.estspinner(model, init, modname, dfInS[assetName], ecol)
    
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
    
    if st.session_state.thetahat is None:
        st.error("Please estimate the model first before searching for patterns.")
    elif len(st.session_state.thetahat) != r + s + 3:
        st.error("Model order changed. Please re-estimate before searching for patterns.")
        st.session_state.thetahat = None
    else:
        kholder.empty()
        
        # Extract only MAR parameters (first r+s elements)
        model.par = st.session_state.thetahat[:r+s]
            
        k0 = np.full(mmax, np.nan)
        
        for j in range(mmax):
            k0[j], _, _, _, _, _ = model.pathfinder(dfInS[assetName], dfInSidx, j+1, kmax, vartheta)
        
        dfk = pd.DataFrame(k0)
        dfk.index = dfk.index + 1
        dfk = dfk.T
        dfk.index.name = 'm'
        dfk.rename(index={dfk.index[0]: 'k'}, inplace=True)
        st.session_state.khat = dfk
        kholder.write(st.session_state.khat) 
        k0mean = st.session_state.khat.mean(axis=1)
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
    selh = st.slider('Select the forecast horizon $h^*$', 1, 10, 1)
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

    for_button = st.button('Forecast', disabled=not isfore)

    if for_button:
        if st.session_state.thetahat is None:
            st.error("Please estimate the model first before forecasting.")
        elif len(st.session_state.thetahat) != r + s + 3:
            st.error("Model order changed. Please re-estimate before forecasting.")
            st.session_state.thetahat = None
        else:
            # Use the complete parameter set (MAR + stable parameters)
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
                                step=0.01, value=0.9100, min_value=0.5, max_value=1.0, format='%0.5f')
        
        prob_button = st.button('Compute', disabled=isfore)
        
    with col8:
        help = '''**Need some help with interpretation?**  
                    The forecast horizon iterates up to $h = k_0 + h^*$.  
                    *None* is reported until $h < k_0$.
                    One possible way to set the cutoff is to report
                    the probability to crash at $h = k_0 + 1$.
                '''
        st.markdown(help)
        
    if prob_button:   
        if st.session_state.thetahat is None:
            st.error("Please estimate the model first before computing probabilities.")
        elif len(st.session_state.thetahat) != r + s + 3:
            st.error("Model order changed. Please re-estimate before computing probabilities.")
            st.session_state.thetahat = None
        else:
            # Use the complete parameter set (MAR + stable parameters)
            model.par = st.session_state.thetahat
            
            # Note: foreprob now returns 4 values
            proba, pfore, past_pattern, full_pattern = model.foreprob(
                dfInS[assetName], selk0, h, maxp, dfInSidx, vartheta, m=selm
            )
            
            st.dataframe(proba, use_container_width=True, height=250)
            
            df_nan = pd.DataFrame(np.nan, index=range(h), columns=dfInS.columns)
            dfpred = pd.concat([dfInS, df_nan], ignore_index=True)
            dfpred['Forecasts'] = pfore
            if isback:
                dfpred.index = df.index[:len(pfore)]
                
            st.session_state.foreprob = dfpred
            
            st.line_chart(dfpred.tail(selm+h+50))


st.subheader('Backtesting', divider='violet')
st.write('This section is available only if your Out-of-Sample starts before the end of the database')

back_button = st.button('Backtest', disabled=not isback)

if back_button:
    if isback and isfore:
        dfback = df.merge(st.session_state.forecast['Forecasts'], on='Dates', how='left')
        dfmetrics = dfback.dropna()
        st.line_chart(dfback.iloc[dfInSidx-selm-h-80:dfInSidx+80, :2])   
        
    if isback and not isfore:
        dfback = df.merge(st.session_state.foreprob['Forecasts'], on='Dates', how='left')
        dfmetrics = dfback.dropna()
        st.line_chart(dfback.iloc[dfInSidx-selm-h-80:dfInSidx+80, :2])  
        
disclaimer = '''**Disclaimer:**
    This application aims to playfully illustrate the possible uses of the results from the paper
    *Forecasting extreme trajectories using seminorm representations*. The authors decline all
    responsibility for the use of this application and do not provide any support in case of
    technical problems insofar as it is not an econometrics software. The GCoV has been adapted 
    to handle alpha-stable processes with K and H set to 2. Many tuning parameters are
    used in this application and are available upon request.
        '''
st.caption(disclaimer)
