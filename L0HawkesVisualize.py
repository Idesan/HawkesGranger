# -*- coding: utf-8 -*-
"""
Originally created on Wed Nov 24 16:15:48 2021.
Redeveloped on 06/2024 as independent code. 

@author: Tsuyohi (Ide-san) Ide (tide@us.ibm.com) 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb; sb.set_theme()

def show_causal_triggers_of(n,qself,qhist,timestamps,event_types, 
                            time_unit = '', time_format = "%Y-%m-%d %H:%M:%S",
                            num_bars=25,
                            figsize_x=16,figsize_y=4,fontsize_label=18,
                            num_bars_annotated=5, threshold_probability=0.01,
                            margin_above_bar = 15,fontsize_dt=16,
                            ylim_scale = 1.15,
                            short_event_type = False, split_symbol='-'):
    '''
    Utility function to plot triggering probabilities for instance-level 
    causal analysis.

    Parameters
    ----------
    n : int
        Which event instance to focus on.
    qself : 1D numpy array
        Self-triggering probability for N+1 events. The zero-th event is the 
        genesis event and thus qself[0] should be np.nan. See L0HawkesNh().
    qhist : 2D numpy array
        Instance triggering probabilities for N+1 events of the rows. 
        The zero-th row is for the genesis event with no history and thus is 
        all np.nan. L0HawkesNh().
    timestamps : 1D numpy array
        Event timestamps of N+1 events 
    event_types : 1D numpy array
        Event types of N+1 events
    time_unit : str, optional
        One of '', 'ns','us','ms','s','m','h','D' (case sensitive). If '' is 
        provided, no time conversion will be made in the title of the plot. 
        If one of 'ns','us','ms','s','m','h','D' are provided, which denote
        nanoseconds, microseconds, milliseconds, seconds,minutes,hours,days,
        respectively, the timestamp of the event event chosen is converted into
        time_format in the title of the plot. 
        The time origin is '1970-01-01 00:00:00'. So 18262 with 'D' gives
        a string of '2020-01-01 00:00:00'. The default is ''.
    time_format : str
        Time format used as the title of the bar chart. 
        The default is "%Y-%m-%d %H:%M:%S".
    num_bars : int, optional
        Number of bars in the histogram. The default is 25.
    figsize_x : float, optional
        Histogram size x. The default is 16.
    figsize_y : float, optional
        Histogram size y. The default is 4.
    fontsize_label : float, optional
        Fontsize of  the lables. The default is 18.
    num_bars_annotated : int, optional
        How many bars to annotate with time distance. The default is 5.
    threshold_probability : float, optional
        Threshold to omit annotation. The default is 0.01.
    margin_above_bar : int, optional
        Margin between the top of the bar and annotation text. The default is 20.
    fontsize_dt : int, optional
        Font size of the time difference annotation above the bars
    short_event_type : boolean, optional
        Whether to use only the last portion of the event name for the barchart
    split_symbol : str, optional
        If short_event_type=True, each event type name is split using this
        string as the deliminator and only the last portion will be used. 

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure object of the barchart.
    ax : matplotlib.axis.Axis
        axis object of the barchart.

    '''
    N = len(timestamps) -1
    Nh = qhist.shape[1]
    
    if num_bars > Nh + 1:
        print('num_bars has been adjusted to Nh={}'.format(Nh))
        num_bars = Nh + 1
    if n <= 0 or n > N:
        raise ValueError('n={} is invalid. Must be in [1,N].'.format(n))
    elif N != len(qself)-1 or N != qhist.shape[0] -1:
        raise ValueError('qself or qhist size inconsistent with timestamps')
    
    if time_unit == '':
        timeStr_n = '{:.1f}'.format(timestamps[n])
    else:
        try:
            timeStr_n = pd.to_datetime(timestamps[n],unit=time_unit).\
                strftime(time_format)
        except TypeError:
            raise ValueError('{}: invalid time_unit'.format(time_unit))
    
    # Create list for all the eligible instances
    history_of_n = np.arange( np.max([0,n-Nh]), n-1+1)
    qhist_n = qhist[n, 0:(np.min([n-1,Nh-1]) + 1)]
    
    History_of_n = np.concatenate([history_of_n, [n] ] )
    Qhist_n = np.concatenate([qhist_n, [qself[n]]])
    
    # Now let us consider num_bars
    idx_to_plot = History_of_n[ np.max([0,len(History_of_n)-num_bars]): ]
    q_to_plot = Qhist_n[ np.max([0,len(History_of_n)-num_bars]): ]   
    
    # Getting corresponding time stamps
    Delta_n = timestamps[n] - timestamps[idx_to_plot]
    
    # Getting corresponding event types
    xticklabel = event_types[idx_to_plot]
    if short_event_type:
        # Create a short version of event names
        for ii in range(len(xticklabel)):
            name = xticklabel[ii]
            xticklabel[ii] = name.split(split_symbol)[-1]
        
    #--- Plotting barchart
    fig,ax = plt.subplots(figsize=(figsize_x,figsize_y))
    titleStr = 'n={} (type:{}) t[n]={}'.format(n,event_types[n],timeStr_n)
    ax.set_title(titleStr,fontsize=fontsize_label)
    ax.set_ylim(0,np.max(q_to_plot)*ylim_scale)
    
    # arranging x tick labels
    colors = np.repeat('black',len(q_to_plot)); colors[-1]='red'    
    ax.bar(np.arange(0,len(q_to_plot)),q_to_plot, color=colors)    
    ax.set_xticklabels(xticklabel,fontsize=fontsize_label,rotation=90)
    ax.set_xticks(np.arange(0,len(q_to_plot)))
    ax.get_xticklabels()[-1].set_color('red')
    
    ax.tick_params(axis='y',labelsize=fontsize_label)    
       
    # Annotate bars with top num_bars_annotated probabilities
    sute = (-q_to_plot).argsort()[:num_bars_annotated]
    idxes = [sute[ii] for ii in range(len(sute)) if sute[ii] != len(q_to_plot)-1]    

    for ii in idxes:
        if q_to_plot[ii] > threshold_probability:
            rectObject = ax.patches[ii]
            y_value  = rectObject.get_height()
            x_value = rectObject.get_x() + rectObject.get_width() / 2

            label = rounded_time_label(Delta_n[ii],time_unit)            

            ax.annotate(label,(x_value,y_value),xytext=(0,margin_above_bar),
                        textcoords="offset points",ha='center',va='top',
                        fontsize=fontsize_dt)
    return fig,ax


def rounded_time_label(dt_i,time_unit):
    '''
    Returns rounded representation of time difference. 62.3 sec = 1m etc.

    Parameters
    ----------
    dt_i : double or int
        Time difference value in number.
    time_unit : str
        One of '', 'ns','us','ms','s','m','h','D' (case sensitive)

    Returns
    -------
    label : str
        Rounded representation of time difference with unit.

    '''
    
    if time_unit =='':
        label = str(int(np.round(dt_i)))
    
    elif time_unit == 'ns':
        if(dt_i < 1e3):
            label= str(int(np.round(dt_i))) + 'ns'
        elif 1e3 <= dt_i <1e6:
            label = str(int(np.round(dt_i/1e3)))+'us'
        elif 1e6 <= dt_i < 1e9:
            label = str(int(np.round(dt_i/1e6)))+'ms'
        elif 1e9 <= dt_i:
            value,unit = rounded_time_label_second(int(dt_i/1.e9))
            label = str(value)+unit
            
    elif time_unit == 'us':
        if(dt_i < 1e3):
            label= str(int(np.round(dt_i))) + 'us'
        elif 1e3 <= dt_i <1e6:
            label = str(int(np.round(dt_i/1e3)))+'ms'
        elif 1e6 <= dt_i:
            value,unit = rounded_time_label_second(int(dt_i/1.e6))
            label = str(value)+unit
            
    elif time_unit == 'ms':
        if(dt_i < 1e3):
            label= str(int(np.round(dt_i))) + 'ms'
        elif 1e3 <= dt_i:
            value,unit = rounded_time_label_second(int(dt_i/1.e3))
            label = str(value)+unit
            
    elif time_unit == 's':
        value,unit = rounded_time_label_second(dt_i)
        label = str(value)+unit
            
    elif time_unit == 'm':
        if(dt_i < 60):
            label= str(int(np.round(dt_i))) + 'm'
        elif 60 <= dt_i < 1440:
            label = str(int(np.round(dt_i/60.)))+'h'
        elif 1440 <= dt_i:
            label = str(int(np.round(dt_i/1440.)))+'D'
            
    elif time_unit == 'h':
        if(dt_i < 24):
            label= str(int(np.round(dt_i))) + 'h'
        elif 24 <= dt_i < 8760:
            label = str(int(np.round(dt_i/24.)))+'D'
        elif dt_i >= 8760:
            label = str(int(np.round(dt_i/8760.)))+'y'
            
    elif time_unit == 'D':
        if(dt_i < 2*365):
            label= str(int(np.round(dt_i))) + 'D'
        elif dt_i >= 2*365:
            label = str(int(np.round(dt_i/365.)))+'y'        
    return label


def rounded_time_label_second(dt_i):
    '''
    Convert time difference in seconds into a rounded value. 

    Parameters
    ----------
    dt_i : double
        Time difference value.

    Returns
    -------
    value : int
        Rounded time difference value.
    unit : str
        Time unit.

    '''
    
    if(dt_i <60):
        unit = 's'
        value = int(np.round(dt_i))
    elif 60 <= dt_i <3600:
        unit = 'm'
        value = int(np.round(dt_i/60.))
    elif 3600 <= dt_i < 86400:
        unit = 'h'
        value = int(np.round(dt_i/3600.))
    elif 86400 <= dt_i:
        unit = 'D'
        value = int(np.round(dt_i/86400.))
    return value,unit


def show_model(result,note='',show_diagonal=False,shorten_mode=0):
    '''
    Visualize mu, beta, A, and the log likelihood. Works with L0HawkesNh funciton.

    Parameters
    ----------
    result : dictionary
        Output computed by L0HawkesNh.
    note : string, optional
        Arbitrary note shown in the log likelihood graph. The default is ''.
    show_diagonal : boolean, optional
        Whether to show the diagonals in the heatmap plot of A. Note that 
        precisely estimating the diagonals is challenging in this model 
        because of the inherent indistinguishability between the baseline 
        intensity and self-excitation. The default is False.
    shorten_mode : int, optional
        If the original event type is too long and connected many words 
        with '-', shorten_mode=1 uses only the last word for a better plot of 
        the heatmap of A. The default is 0.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    dictionary : {'loglik','beta_mu','A'}
        'loglik' : dictionary {'fig','ax'}
            'fig' : matplotlib.figure.Figure
                Figure object for the log-likelihood graph
            'ax' : matplotlib.axis.Axis
                Axis object for the log-likelihood graph
        'beta_mu' : dictionary {'fig','ax'}
            'fig' : matplotlib.figure.Figure
                Figure object for the bar plots of beta and mu
            'ax' : list
                Contains two matplotlib.axis.Axis objects for beta and mu
        'A' : dictionary {'fig','ax'}
            'fig' : matplotlib.figure.Figure
                Figure object for the heatmap of A
            'ax' : matplotlib.axis.Axis
                Axis object for the heatmap of A
    '''

    if any(ge not in result for ge in ['event_list','learned_params']):
        raise ValueError('event_list or learned_params not found')
    if not any(shorten_mode == ge for ge in [0,1]):
        raise ValueError('event_list or learned_params not found')
    
    # Visualization
    event_list = result.get('event_list')
    learned_params = result.get('learned_params')
    
    if any(ge not in learned_params for ge in ['mu','beta','loglik','A']):
        raise ValueError('mu/beta/loglik/A not found')
   
    mu_vec = learned_params.get('mu')
    beta_vec = learned_params.get('beta')
    loglik = learned_params.get('loglik')
    A_sparse = learned_params.get('A')
    
    # Short version of event_list
    if shorten_mode==0:
        eLshort = event_list
    elif shorten_mode==1:
        event_list = [str(ge) for ge in event_list] # In case event_list is of integers
        uu = pd.Series(event_list).str.split('-')
        eLshort = np.array([uu[ii][-1] for ii in range(len(uu))])


    # Plotting likelihood, beta, mu
    fig_loglik,ax_loglik = plt.subplots(1,1,tight_layout=True)
    titleStr = 'log-likelihood (maximum={:.7g})'.format(loglik[-1]) + note
    Loglik = pd.Series(loglik,index=np.arange(1,len(loglik)+1))
    LL= pd.Series(Loglik)[10:]
    LL.plot(title=titleStr,xlabel="Number of MM iterations",color='black',
            ax=ax_loglik)

    fig_mu_beta,ax_mu_beta = plt.subplots(1,2,sharey=True)
    bb = pd.Series(beta_vec,index=eLshort)[::-1]
    bb.plot.barh(title='beta',ax=ax_mu_beta[0],color='black')
    mm = pd.Series(mu_vec,index=eLshort)[::-1]
    mm.plot.barh(title='mu',ax=ax_mu_beta[1],color='black')

    # Causal matrix
    if show_diagonal:
        AA = A_sparse
    else:
        AA = A_sparse - np.diag(np.diag(A_sparse))   
    
    AA = pd.DataFrame(AA,index=eLshort,columns=eLshort)
    fig_A, ax_A = plt.subplots(tight_layout=True,figsize=(6,5))
    sb.heatmap(AA,ax=ax_A,cmap='hot', square=True)
    AtitleStr = 'A ' + note
    ax_A.set_title(AtitleStr)
    
    # Returning fig/ax objects
    obj_loglik= {'fig':fig_loglik,'ax':ax_loglik}
    obj_beta_mu = {'fig':fig_mu_beta,'ax':ax_mu_beta}
    obj_A = {'fig':fig_A,'ax':ax_A}
    obj = {'loglik':obj_loglik,'beta_mu':obj_beta_mu,'A':obj_A}
    return obj

def show_multivariate_points(timestamps,event_types, figsize=(12,2),
                             alpha=0.5,marker='|',linewidths=1,markersize=100):
    '''
    Visualize multivariate timestamped event.

    Parameters
    ----------
    timestamps : 1D numpy array
        Timestamps as a sequence of real numbers.
    event_types : 1D numpy array
        Event types as a sequence of strings.
    figsize : tuple, optional
        Figure size. The default is (12,2).
    alpha : float, optional
        Transparency. 1 is black, 0 is fully transparent. The default is 0.5.
    marker : string, optional
        marker symbol to be passed to ax.scatter(). The default is '|'.
    linewidths : int, optional
        linewidths number to be passed to ax.scatter(). The default is 1.
    markersize : int, optional
        markersize number to be passed to ax.scatter(). The default is 100.

    Returns
    -------
    event_list : 1D numpy array
        event_list[0],.., event_list[D-1] are the 1st, 2nd, ..., Dth event type,
        corresponding to the 1st, 2nd, ... rows of the scatter plots.
    plt : matplotlib.pyplot object
        pyplot object for the scatter plot.
    fig : matplotlib.figure.Figure
        figure object of the scatter plot.
    axes : matplotlib.axis.Axis
        axis object of the scatter plot.

    '''
    event_list = np.unique(event_types)
    fig,axes = plt.subplots(len(event_list),1,figsize=figsize,sharex=True)
    if len(event_list) == 1:
        axes = [axes]
    for k in range(len(event_list)):
        mask = (event_types == event_list[k])              
        timestamps_of_k = timestamps[mask]
        yy = np.repeat(0,len(timestamps_of_k))
        axes[k].scatter(timestamps_of_k,yy,
                         marker=marker,linewidths=linewidths,
                         alpha=alpha,color='black',s=markersize)
        axes[k].axes.xaxis.set_visible(False)
        axes[k].axes.yaxis.set_visible(False)
        axes[k].axis("off")
    fig.tight_layout()
    return event_list,plt,fig,axes,