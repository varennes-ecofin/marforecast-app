import time
import pandas as pd
import numpy as np
import stablemar as mar
import streamlit as st

def initstate():
    if 'button_enabled' not in st.session_state:
        st.session_state.button_enabled = False 
        
    if 'thetahat' not in st.session_state:
        st.session_state.thetahat = None  
        
    if 'khat' not in st.session_state:
        st.session_state.khat = None 
        
    if 'forecast' not in st.session_state:
        st.session_state.forecast = None 
        
    if 'foreprob' not in st.session_state:
        st.session_state.foreprob = None 
        
    return st.session_state    

def dataselection(option):
    
    if option == 'Climate data':
        data = pd.read_excel('AssetsMAREstim.xlsx', sheet_name='Climate')
        dates = pd.to_datetime(data['Dates'], format='%Y:%m:%d')
        data['Dates'] = dates
        data['SOI'] = data['SOI'] - data['SOI'].mean()
    elif option == 'Artificial data':
        data = pd.read_excel('AssetsMAREstim.xlsx', sheet_name='Artificial')
        dates = pd.to_datetime(data['Dates'] + '-2', format='%Y-%W-%w')
        data['Dates'] = dates
    elif option == 'FRED Macro data':
        data = pd.read_excel('AssetsMAREstim.xlsx', sheet_name='FREDMacro')
        dates = pd.to_datetime(data['Dates'], format='%Y-%m-%d')
        data['Dates'] = dates
    elif option == 'Ocean data':
        data = pd.read_excel('AssetsMAREstim.xlsx', sheet_name='Ocean')
        dates = pd.to_datetime(data['Dates'], format='%Y:%m:%d')
        data['Dates'] = dates
    elif option == 'Crypto data':
        data = pd.read_excel('AssetsMAREstim.xlsx', sheet_name='Crypto')
        dates = pd.to_datetime(data['Dates'] + '-2', format='%Y-%W-%w')
        data['Dates'] = dates
    else:
        data = pd.read_excel('AssetsMAREstim.xlsx')
        dates = pd.to_datetime(data['Dates'] + '-2', format='%Y-%W-%w')
        data['Dates'] = dates    
        
    return data, dates

def sampleselection(oos, df, dates):
    
    closest_date = df.index[abs(df.index - pd.to_datetime(oos)).argmin()]
    if closest_date == pd.to_datetime(oos):
        st.write('You selected:', oos)
    else:
        st.write('You selected', oos,' but that date has been replaced by the closest one available:', closest_date.date())
        oos = closest_date.date()

    oosdate = pd.to_datetime(oos)
    dfInS = df.loc[df.index <= oosdate]
    dfInSidx = df.index.get_loc(oosdate)

    if oos == dates.iloc[-1].date():
        # dfOoS = None
        isback = False
    else:
        # dfOoS = df.loc[df.index > oosdate]
        isback = True
        
    return dfInS, dfInSidx, isback     

def modelspecification(modsel):
    
# modsel = st.radio(
#     "Select a model",
#     ["MAR(r,s)", "MARST(r,s)", "Convoluted MAR(0,1)"],
#     captions=[
#         "GCov (Gourieroux and Jasiak 2023) and SCF regression (Nolan 2020)",
#         "Extension of the Stable MLE of Davis et al. (2009)",
#         "Deconvolution Minimum Distance estimator (Gourieroux and Zakoan 2017)",
#     ]
# )

    if modsel == "Convoluted MAR(0,1)":
        maxrlag = 0
        maxslag = 1
        inilag = 0
        modname = "ARCMAR"

    elif modsel == "MARST(r,s)":
        maxrlag = 2
        maxslag = 2
        inilag = 1
        modname = "MARST"

    else:
        maxrlag = 3
        maxslag = 3
        inilag = 1
        modname = "MAR" 
        
    rcol, scol = st.columns(2)

    with rcol:
        r = st.number_input("Number of backward lags (r)", min_value = 0, max_value = maxrlag, value = inilag)
    with scol:
        s = st.number_input("Number of forward lags (s)", min_value = 1, max_value = maxslag, value = 1)

    if r == 0 and s == 0: 
        st.write("Gcov necessitate at least an AR term. By default a MAR(0,1) is estimated")
        s = 1
        order = (r, s) 
    else:  
        order = (r, s)  
        
    return  order, modname

def drawdeltas(order, thetahat, assetName):

    r, s = order
    modelstr = "MAR(" + str(r) + "," + str(s) + ")"
    st.write('The ',assetName,' series follows a stable ',modelstr,' with infinite MA coefficients')

    psi, phi = mar.marcheck(thetahat, order)

    m = 20
    deltas = np.full(2*m, np.nan)
    
    for k in range(-m, m):
        deltas[k+m] = mar.madelta(psi, phi, k) 

    deltas = np.flip(deltas) 

    st.line_chart(deltas)
    
def estspinner(model, init, modname, df, ecol):
    with st.spinner(text = "Estimation in progress..."):
        time.sleep(2)
        
        if modname == "MAR":
            model.splitfit(df.to_numpy(), init, K = 2, H = 2)
            results = model.results
            thetahat = results['Parameters']
        elif modname == "MARST":
            model.fit(df.to_numpy(), init, method = 'bounded')
            results = model.results
            thetahat = results['Parameters']
        else:
            model.fit(df.to_numpy(), init, method = 'free')
            results = model.results
            thetahat = results['Parameters']

        psi, phi = mar.marcheck(thetahat, model.order)    
        
        r, s = model.order
        
        unitpsi = mar.rootcheck(psi)[0] if r > 0 else False
        
        unitphi = mar.rootcheck(phi)[0] if s > 0 else False
        
        if unitpsi or unitphi:
            st.warning("At least one root lies in the unit circle")
        else:
            st.success("Estimators successfully converged")
    
    with ecol:
        estdf = pd.DataFrame(thetahat, columns = ["Estimates"])
        st.table(estdf)
        
    return thetahat