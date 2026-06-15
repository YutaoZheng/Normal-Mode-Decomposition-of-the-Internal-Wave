import numpy as np
import matplotlib.pyplot as plt
from . import Processing
from . import Cov
from speccy import sick_tricks as gary
from speccy import utils as ut
import string
import pandas as pd

def Model_fit_result(F_obs_list,P_obs_list, 
                    F_model_fit_list, P_model_fit_list,model_name='M1L2'):
    
    num_subplots = len(F_obs_list)
    rows = 2
    cols = int(np.ceil(num_subplots/rows))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7),sharex=True,sharey=True)
    axes = axes.flatten()
    fig.text(0.5, 0.99, '{} model fit result'.format(model_name), ha='center', va='center')
    
    for i in range(len(F_obs_list)):

        #plot obs
        axes[i].plot(F_obs_list[i],P_obs_list[i],label='Residual Subset',alpha=0.5)
        #plot model fit
        axes[i].plot(F_model_fit_list[i],P_model_fit_list[i],'-.',
                     label='Mode {} Model Fit'.format(i+1),linewidth=2)
        
        axes[i].set_xscale("log")
        axes[i].set_yscale("log")
        axes[i].legend(loc="upper right")
        axes[i].set_xlim(0.5,40)
        axes[i].set_ylim(1e-3,1e3)
        
    for ax in axes[-cols:]:
        ax.set_xlabel('Frequency [cpd]')
    # Dynamically set the ylabel only for left column subplots
    for ax in axes[::cols]:
        ax.set_ylabel('Wave Spectrum')

    plt.tight_layout()  # Adjust layout so titles and labels don't overlap
    plt.show()   
        
def For_list(x,y_list,label,color='r',alpha=0.05):
    for order,i in enumerate(y_list):
        if order == 0:
            plt.plot(x, i,color=color,alpha=alpha, label=label) #just for legend
        else:
            plt.plot(x, i,color=color,alpha=alpha)

def Plot_IW_amp(time_list,A_list,nmodes):
    row = 3
    column = 1
    fig, axes = plt.subplots(row, column, figsize=(10, 10),sharex=True,sharey=True)  # Create a 2x4 subplot grid
    axes =  axes.flatten()
    for mode_number in range(nmodes):
        axes[mode_number].plot(time_list[mode_number],A_list[mode_number])
        axes[mode_number].set_title(mode_number+1)

    for ax in axes[::column]:
        ax.set_ylabel('Displacement(m)')
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout so titles and labels don't overlap
    plt.show()
  
def Plot_IW_spectrum(time_list,A_list,nmodes):
    row = 3
    column = 1
    fig, axes = plt.subplots(row, column, figsize=(10, 10),sharex=True,sharey=True)  # Create a 2x4 subplot grid
    axes =  axes.flatten()
    for mode_number in range(nmodes):
        Δ = (time_list[mode_number][1]-time_list[mode_number][0]).astype('float')/1e9/86400
        Δ = Δ.values
        F_obs,P_obs = Processing.Cal_periodogram(A_list[mode_number].values,Δ)
        axes[mode_number].plot(F_obs,P_obs )
        axes[mode_number].set_title(mode_number+1)
        axes[mode_number].set_ylabel('amplitude(m)')
        axes[mode_number].set_xscale('log')
        axes[mode_number].set_yscale('log')
        axes[mode_number].set_xlabel('freqyency(cpd)')
        axes[mode_number].set_ylabel('PSD(m^2/cpd)')

    plt.tight_layout()  # Adjust layout so titles and labels don't overlap
    plt.show()

def Plot_HA_result(time_list,A_list,xcoords,
                   Mean_params_list,ϵ_list,Yd_mean_list,
                   F_ϵ_list,Puu_ϵ_list, mode_number=0):
    
    x = time_list[mode_number]
    y = A_list[mode_number]
    ϵ = ϵ_list[mode_number]
    yd_mean = Yd_mean_list[mode_number]
    #calculate spectrum
    Δ = (x[1]-x[0]).astype('float')/1e9/86400
    Δ = Δ.values
    F_obs,P_obs = Processing.Cal_periodogram(y.values,Δ)

    plt.figure(figsize=(15,12))
    plt.subplot(2, 1, 1)
    idx = 15000
    plt.plot(x[10000:idx],y[10000:idx],label='Obs',alpha=0.5)
    plt.plot(x[10000:idx],ϵ[10000:idx],'-.',label = 'Residual')
    plt.plot(x[10000:idx],yd_mean[10000:idx],label='HA',linewidth=2.5)
    plt.xlabel('days')
    plt.ylabel('Amp (m)')
    # plt.grid(b=True,ls=':')
    plt.title('Time series mode {}'.format(mode_number+1))
    plt.legend(loc="lower right")

    F_ϵ   = F_ϵ_list[mode_number]
    Puu_ϵ = Puu_ϵ_list[mode_number]
    Peaks = Processing.Coherent_peaks(xcoords[1:],Mean_params_list[mode_number],F_ϵ)

    plt.subplot(2, 1, 2)
    
    plt.plot(F_obs,P_obs,label='Obs',alpha=0.5)
    plt.plot(F_ϵ,Puu_ϵ,label='Residual')
    plt.plot(F_ϵ,Peaks,label='HA',linewidth=2.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('f_mean[cycles per day]')
    plt.ylabel('PSD(m²/cpd)')
    # plt.grid(b=True,ls=':')
    plt.legend(loc="lower right") 
    plt.ylim(1e-5, 1e5)
    plt.xlim(0.4,50)

def Plot_parameter_varibility(ax, df, parameter, parameter_unit, xlabel=False):
    # Calculate both median and std for the parameter
    stats_df = df.groupby(['Site', 'Mode'])[parameter].agg(['median', 'std']).reset_index()

    for site, site_data in stats_df.groupby('Site'):
        # Use `median` for the main line and `std` for error bars
        ax.errorbar(site_data['Mode'], site_data['median'], 
                    yerr=site_data['std'], label=site, marker='o', capsize=3)
    
    # Set y-axis label and custom x-axis ticks
    ax.set_ylabel(f'{parameter} ({parameter_unit})')
    ax.set_xticks(df['Mode'].unique())
    
    # Conditionally set x-axis label
    if xlabel:
        ax.set_xlabel('Mode')

#only works for variance parameter
def plot_η_depth_variablity(parameter_dict, parameter_name, modes, colors):
    #plot set up
    cols = len(parameter_name)
    rows= 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5),sharey=True)
    axes = axes.flatten()
    figure_order = [list(string.ascii_lowercase)[i % 26] for i in range(len(parameter_name))]
    for parameter_order, parameter_name in enumerate(parameter_name):
        parameter_df_plot_median_all_mode = []
        parameter_df_plot_std_all_mode = []
        for mode_number in modes[:3]:
            parameter_dict =  {key: variable for key, variable in parameter_dict.items() if key[2] == parameter_name and key[1] == mode_number}
            # Create a list to hold the data for plotting
            parameter_df_plot = Processing.Transfer_to_plot_df(parameter_dict)
            parameter_df_plot_median = parameter_df_plot.groupby('depth')['values'].median()
            parameter_df_plot_std    = parameter_df_plot.groupby('depth')['values'].std()
            parameter_df_plot_median_all_mode.append(parameter_df_plot_median)
            parameter_df_plot_std_all_mode.append(parameter_df_plot_std)
            #plot
            y_offset = 2
            axes[parameter_order].errorbar(parameter_df_plot_median,
                                       parameter_df_plot_median.index+mode_number * y_offset,
                                       xerr=parameter_df_plot_std, color=colors[mode_number],
                                        capsize=3,alpha=0.8)
            axes[parameter_order].set_xlabel('value (K)')

        parameter_df_plot_median_sum = pd.DataFrame(sum(df for df in parameter_df_plot_median_all_mode))
        parameter_df_plot_std_sum    = pd.DataFrame(sum(df ** 2 for df in parameter_df_plot_std_all_mode) ** 0.5)  # Summing variances
        axes[parameter_order].errorbar(parameter_df_plot_median_sum['values'],
                                   parameter_df_plot_median.index + (mode_number+1)*y_offset,
                                    xerr=parameter_df_plot_std_sum['values'], color=colors[3],
                                    capsize=3, label='Sum of Modes')
        #plots set up
        axes[parameter_order].set_title('({}) {}'.format(figure_order[parameter_order],parameter_name))
        axes[parameter_order].grid(True)

    # Create a custom legend for the locations
    modes_legend = {mode_number: color for mode_number, color in zip(modes, colors)}
    handles = [plt.Rectangle((0,0),1,1, color=modes_legend[mode_number]) for mode_number in modes ]
    labels =  modes 
    fig.legend(handles, labels, title='Modes', loc=(0.88,0.25), bbox_to_anchor=(0.9, 0.4))  

    #set y label
    for i in range(len(axes)):
        if i%2==0:
            axes[i].set_ylabel('Depth(m)')
        
    # Show plot
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rectangle to leave space for the y-axis label and legend
    # Show plot
    plt.show()


def Plot_parameter_distribution(df, modes,model_type,
                                parameter_name,parameter_units, 
                                colors,num_bins =10, cols=2):
    #figure pre-setup
    rows = int(len(parameter_name)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10),sharey=True) 
    axes = axes.flatten()
    figure_order = [list(string.ascii_lowercase)[i % 26] for i in range(len(parameter_name))]

    for parameter_order,parameter in enumerate(parameter_name):
        for mode_number in modes[:3]:
            data_mode = df.loc[df['Mode']==mode_number]
            hist, bin_edges = np.histogram(data_mode[parameter], bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Calculate the percentage of data in each bin
            percentages = hist / len(data_mode) * 100
        
            axes[parameter_order].plot(bin_centers, percentages,label=modes[mode_number], marker='o',markersize=3,color=colors[mode_number])
            axes[parameter_order].set_title('({}) {}'.format(figure_order[parameter_order],parameter))
            axes[parameter_order].set_xlabel('Value ({})'.format(parameter_units[parameter_order]))
            
    for i, ax in enumerate(axes):
        if i % cols == 0: 
            ax.set_ylabel('Percentage of Data (%)')
    axes[int(len(axes)/2)].legend()

    fig.suptitle('{} Line Plot with Bin Centers and Continuous Line'.format(model_type))
    fig.tight_layout()
    plt.show()   