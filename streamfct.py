import time
import pandas as pd
import numpy as np
from stablemar import StableMAR, madelta, root_check
import streamlit as st


def initstate():
    """
    Initialize session state variables for the Streamlit app.
    
    Returns:
        st.session_state: Streamlit session state object
    """
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
    
    if 'model_order' not in st.session_state:
        st.session_state.model_order = None
        
    return st.session_state


def check_and_reset_if_order_changed(current_order):
    """
    Check if model order has changed and reset estimates if needed.
    
    Args:
        current_order (tuple): Current model order (r, s)
    """
    # Check if this is not the first time or if order has changed
    if st.session_state.model_order is not None and st.session_state.model_order != current_order:
        # Order changed, reset all estimates
        st.session_state.thetahat = None
        st.session_state.khat = None
        st.session_state.forecast = None
        st.session_state.foreprob = None
        st.info("Model order changed. All estimates have been reset. Please re-estimate.")
    
    # Update stored order
    st.session_state.model_order = current_order    


def dataselection(option):
    """
    Load and prepare data based on user selection.
    
    Args:
        option (str): Name of the dataset to load
        
    Returns:
        tuple: (data DataFrame, dates Series)
    """
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
    """
    Split data into in-sample and out-of-sample periods.
    
    Args:
        oos: Out-of-sample start date
        df (pd.DataFrame): Full dataset
        dates (pd.Series): Date index
        
    Returns:
        tuple: (in-sample data, in-sample index, backtesting flag)
    """
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
        isback = False
    else:
        isback = True
        
    return dfInS, dfInSidx, isback     


def modelspecification(modsel):
    """
    Configure model specification based on user selection.
    
    Args:
        modsel (str): Model type selected
        
    Returns:
        tuple: (order tuple (r, s), model name)
    """
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
        r = st.number_input("Number of backward lags (r)", min_value=0, max_value=maxrlag, value=inilag)
    with scol:
        s = st.number_input("Number of forward lags (s)", min_value=1, max_value=maxslag, value=1)

    if r == 0 and s == 0: 
        st.write("GCoV necessitates at least an AR term. By default a MAR(0,1) is estimated")
        s = 1
        order = (r, s) 
    else:  
        order = (r, s)  
        
    return order, modname


def drawdeltas(order, thetahat, assetName):
    """
    Draw the MA coefficients (deltas) for visualization.
    
    Args:
        order (tuple): Model order (r, s)
        thetahat (list): Estimated parameters (MAR + stable parameters)
        assetName (str): Name of the time series
    """
    r, s = order
    modelstr = "MAR(" + str(r) + "," + str(s) + ")"
    st.write('The', assetName, 'series follows a stable', modelstr, 'with infinite MA coefficients')

    # Extract only MAR parameters (first r+s parameters)
    mar_params = thetahat[:r+s]
    psi = np.array(mar_params[:r]) if r > 0 else np.array([])
    phi = np.array(mar_params[r:r+s]) if s > 0 else np.array([])

    m = 20
    deltas = np.full(2*m, np.nan)
    
    for k in range(-m, m):
        deltas[k+m] = madelta(psi, phi, k) 

    deltas = np.flip(deltas) 

    st.line_chart(deltas)
    

def estspinner(model, init, modname, df, ecol):
    """
    Perform model estimation with spinner animation.
    
    Args:
        model (StableMAR): Model instance
        init (list): Initial parameter values
        modname (str): Model name
        df (pd.DataFrame): Data to fit
        ecol: Streamlit column for output
        
    Returns:
        list: Estimated parameters (MAR + alpha-stable parameters)
    """
    with st.spinner(text="Estimation in progress..."):
        time.sleep(2)
        
        # Fit using GCoV method (the default for MAR models in the new version)
        model.fit(df.to_numpy().flatten(), init, method='gcov', K=2, H=2, verbose=False)
        results = model.results
        thetahat = results['Parameters'].tolist()

        # Estimate alpha-stable parameters from residuals
        if 'PseudoResiduals' in results:
            residuals = results['PseudoResiduals']
        else:
            # Compute residuals manually if not available
            r, s = model.order
            residuals, _ = model._pseudo_residuals(df.to_numpy().flatten(), np.array(thetahat))
        
        stable_params = model.fit_stable_noise(residuals)
        
        # Combine MAR and stable parameters
        full_params = thetahat + stable_params

        # Check for unit roots
        r, s = model.order
        psi = np.array(thetahat[:r]) if r > 0 else np.array([])
        phi = np.array(thetahat[r:r+s]) if s > 0 else np.array([])
        
        unitpsi = root_check(psi)[0] if r > 0 else False
        unitphi = root_check(phi)[0] if s > 0 else False
        
        if unitpsi or unitphi:
            st.warning("At least one root lies in the unit circle")
        else:
            st.success("Estimators successfully converged")
    
    with ecol:
        # Create parameter names
        param_names = []
        for i in range(r):
            param_names.append(f"psi_{i+1}")
        for i in range(s):
            param_names.append(f"phi_{i+1}")
        
        # Add stable parameter names
        param_names.extend(["alpha", "beta", "sigma"])
            
        estdf = pd.DataFrame({
            "Parameter": param_names,
            "Estimates": full_params
        })
        st.table(estdf)
        
    return full_params


def estimate_stable_params(model, df):
    """
    Estimate alpha-stable parameters from residuals after MAR estimation.
    
    Args:
        model (StableMAR): Fitted model instance
        df (pd.DataFrame): Original data
        
    Returns:
        list: Estimated alpha-stable parameters [alpha, beta, sigma]
    """
    # Get pseudo-residuals from the fitted model
    if 'PseudoResiduals' in model.results:
        residuals = model.results['PseudoResiduals']
    else:
        # Compute residuals manually if not available
        r, s = model.order
        residuals, _ = model._pseudo_residuals(df.to_numpy().flatten(), np.array(model.par))
    
    # Estimate alpha-stable parameters using characteristic function method
    stable_params = model.fit_stable_noise(residuals)
    
    return stable_params
