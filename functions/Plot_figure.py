import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from . import Processing
from . import Cov
from speccy import sick_tricks as gary
from speccy import utils as ut
import string
import pandas as pd
import seaborn as sns

def Model_fit_result(F_obs_list,P_obs_list, 
                    F_model_fit_list, P_model_fit_list,):
    
    num_subplots = len(F_obs_list)
    rows = 2
    cols = int(np.ceil(num_subplots/rows))
    fig, axes = plt.subplots(rows, cols,sharex=True,sharey=True)
    axes = axes.flatten()
    # fig.text(0.5, 0.99, 'Model fit result', ha='center', va='center')
    for i in range(len(F_obs_list)):
        #plot obs
        axes[i].plot(F_obs_list[i],P_obs_list[i],label='Residual Subset',alpha=0.5)
        #plot model fit
        axes[i].plot(F_model_fit_list[i],P_model_fit_list[i],
                     label='Mode {} Model Fit'.format(i+1),linewidth=2,alpha=0.8)
        axes[i].set_xscale("log")
        axes[i].set_yscale("log")
        axes[i].legend(loc='lower left')
        axes[i].set_xlim(0.5,40)
        axes[i].set_ylim(1e-3,2e3)
        
    for ax in axes[-cols:]:
        ax.set_xlabel('Frequency [cpd]')
    # Dynamically set the ylabel only for left column subplots
    for ax in axes[::cols]:
        ax.set_ylabel('Wave Spectrum')

    plt.tight_layout()  # Adjust layout so titles and labels don't overlap
    # plt.show()   

    return fig, axes
    
def For_list(x,y_list,label,color='r',alpha=0.05):
    for order,i in enumerate(y_list):
        if order == 0:
            plt.plot(x, i,color=color,alpha=alpha, label=label) #just for legend
        else:
            plt.plot(x, i,color=color,alpha=alpha)

def Plot_Density_fit_performance(ds_dict, bins =10):
    #compute the median of the desity fit
    mode_fraction_median_dict = {}
    for i in ds_dict:
        # median  = np.nanmedian(Processing.Cal_density_fit_percentage(ds_dict[i]))
        # median  = np.nanmean(Processing.Cal_density_fit_percentage(ds_dict[i]))
        median  = Processing.Cal_density_fit_percentage(ds_dict[i]).quantile(0.80).values
        if median >= 100:
            print('{}({}%) is overfitting'.format(i,np.round(median,2)))
        elif median <=50:
            print('{}({}%) is underfit'.format(i,np.round(median,2)))
        mode_fraction_median_dict[i] = median
    
    plt.hist(mode_fraction_median_dict.values(), bins = bins, color="skyblue", edgecolor="black")
    plt.xlabel("Density fit performance [%]")
    plt.ylabel("Frequency")
    plt.title('{} Modes'.format(len(ds_dict[i].modes)))
    plt.tight_layout()
    plt.show()

    return mode_fraction_median_dict
  
def Plot_IW_amp_spectrum(time_list,A_list,nmodes):
    row = nmodes
    column = 2
    fig, axes = plt.subplots(row, column,sharex='col',)  # 
    axes = axes.reshape(row, column)
    for mode_number in range(nmodes):
        # Plot time series on the left (first column)
        axes[mode_number, 0].plot(time_list[mode_number], A_list[mode_number])
        axes[mode_number, 0].set_title(f"Mode {mode_number + 1} Time Series")
        axes[mode_number, 0].set_ylabel('Displacement [m]')

        # Compute the spectrum
        Δ = (time_list[mode_number][1]-time_list[mode_number][0]).astype('float')/1e9/86400
        Δ = Δ.values
        F_obs,P_obs = Processing.Cal_periodogram(A_list[mode_number].values,Δ)
        # Plot spectrum on the right (second column)
        axes[mode_number, 1].plot(F_obs, P_obs)
        axes[mode_number, 1].set_title(f"Mode {mode_number + 1} Spectrum")
        axes[mode_number, 1].set_xscale('log')
        axes[mode_number, 1].set_yscale('log')
        axes[mode_number, 1].set_xlabel('Frequency [cpd]')
        axes[mode_number, 1].set_ylabel('PSD [m^2/cpd]')

    # Rotate x-axis labels for better readability
    for ax in axes[:, 0]:  # Time series column
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
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
    plt.xlabel('f_mean[cpd]')
    plt.ylabel('PSD [m²/cpd]')
    # plt.grid(b=True,ls=':')
    plt.legend(loc="lower right") 
    plt.ylim(1e-5, 1e3)
    plt.xlim(0.4,50)

def Plot_each_component_model_fit(df,dict_name,
                                  mode = 3, model_type = 'M1P1'):
    #pre-setup
    n = 600000
    delta = 600/86400
    tt = ut.taus(n, delta)
    #select the df
    selected_df = df[(df['Dict_name'] == dict_name) & (df['Mode'] == mode)]
    if model_type == 'M1P1':
        print(selected_df.loc[:, 'η_c':'γ_D2'])
        params = selected_df.loc[:, 'η_c':'γ_D2'].values[0]
        
        acf_true_model  = Cov.M1P1(tt, params)
        acf_true_LR2    = Cov.LR_2(tt, params[2:],l_cos=1.965)
    elif model_type == 'M1P2': 
        params = selected_df.loc[:, 'η_c':'γ_M2'].values[0]
        acf_true_model = Cov.M1P2_2(tt, params)
        acf_true_LR2 = Cov.LR_2(tt, params[2:], l_cos=1.93) \
                     + Cov.LR_2(tt, params[5:], l_cos=1.99)   #M2+S2
    else:
        raise ValueError("model_type must be 'M1P2' or 'M1P1'")
    acf_true_Matern = Cov.Matern(tt, params,lmbda=3)
    print(selected_df)
    #numerically calculate spectrum from acf
    ff_model, S_bias_model = gary.bochner(acf_true_model, delta, bias=True)
    ff_LR2, S_bias_LR2     = gary.bochner(acf_true_LR2, delta, bias=True)
    ff_Matern, S_bias_Matern = gary.bochner(acf_true_Matern, delta, bias=True)
    #plot
    plt.plot(ff_LR2[ff_LR2>=0], S_bias_LR2[ff_LR2>=0], label="LR2", linestyle="-.",color='blue')  
    plt.plot(ff_Matern[ff_Matern>=0], S_bias_Matern[ff_Matern>=0], label="Matern", linestyle="-.",color='green')   
    plt.plot(ff_model[ff_model>=0], S_bias_model[ff_model>=0], label=model_type,color='black') 
    plt.title(dict_name)
    plt.xlim(0.5,220)
    # plt.ylabel("PSD (K²/cpd)")
    plt.xlabel("Frequency [cpd]")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize = 20)

    return plt

def Plot_parameter_medium_sns(ax, df, parameter, parameter_unit,):
    mode_order = sorted(df['Mode'].unique())
    palette = sns.color_palette("colorblind", len(mode_order))
    # Group by Site and Mode
    grouped = df.groupby(['Site', 'Mode'], observed=True)
    valid_groups = grouped.filter(lambda g: len(g) >= 2)
    invalid_groups = grouped.filter(lambda g: len(g) < 2)

    # Plot valid groups with pointplot
    if not valid_groups.empty:
        sns.pointplot(
            data=valid_groups,
            x="Site",
            y=parameter,
            hue="Mode",
            hue_order=mode_order,
            palette=palette,
            ax=ax,
            estimator="median",
            dodge=0.25,
            markers="o",
            capsize=0.1,
            errorbar=('pi', 50)
        )
    # # Overlay individual points for invalid groups
    # if not invalid_groups.empty:
    #     sns.stripplot(
    #         data=invalid_groups,
    #         x="Site",
    #         y=parameter,
    #         hue="Mode",
    #         hue_order=mode_order,
    #         palette=palette,
    #         ax=ax,
    #         dodge=True,
    #         marker="o",
    #         size=8,
    #     )
     # Format ylabel with LaTeX
    if "_" in parameter:
        # Convert "eta_m2" -> r"$\eta_{M2}$"
        base, sub = parameter.split("_", 1)
        if parameter.lower().startswith("eta"):  
            # Special underline rule for eta
            ylabel = rf"$\underline{{{base}}}_{{{sub}}}$"
        else:
            ylabel = rf"${base}_{{{sub}}}$"
    else:
        ylabel = rf"${parameter}$"
    ax.set_ylabel(f"{ylabel} [{parameter_unit}]")
    ax.grid(True)
    # Remove subplot legends (we'll show one master legend later)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.tick_params(axis='x', rotation=45)


def Plot_parameter_value(ax, df, parameter, xlabel=False):
    # Iterate over sites and plot with offsets
    for i, (site, site_data) in enumerate(df.groupby('Site',observed=True)):
        # Apply an offset based on index 'i' to separate overlapping points
        offset = (i - len(df['Site'].unique()) / 2) * 0.05
        # Scatter plot for individual data points
        ax.scatter(
            site_data['Mode'] + offset,
            site_data[parameter],
            label=site, 
            marker='o',
            alpha=0.8)
        ax.grid(True)
    ax.set_xticks(df['Mode'].unique())
    # Conditionally set x-axis label
    if xlabel:
        ax.set_xlabel('Mode')



def Plot_seasonality_by_mode(df, site_to_plot, parameter_list,
                             parameter_unit,season_list,figsize=(6, 3)):
    # sns.set(style="whitegrid")
    # Filter by site
    df_site = df[df["Site"] == site_to_plot].copy()
    # Map seasons to numbers
    season_map = {s: i for i, s in enumerate(season_list)}
    df_site["season_num"] = df_site["season"].map(season_map)
    # Offset by mode
    mode_list = sorted(df_site["Mode"].unique())
    mode_offset = {m: (i - (len(mode_list)-1) / 2) * 0.1 for i, m in enumerate(mode_list)}
    # Layout in 2 columns
    n_plots = len(parameter_list)
    figure_order = [list(string.ascii_lowercase)[i % 26] for i in range(len(parameter_list))]
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(figsize[0]*ncols, figsize[1]*nrows),
                             sharex=True,)
    axes = axes.flatten()
    # Plot each parameter
    for i, (ax, param) in enumerate(zip(axes, parameter_list)):
        df_site["x"] = df_site.apply(lambda r: r["season_num"] + mode_offset[r["Mode"]], axis=1)
        sns.scatterplot(data=df_site, x="x", y=param, hue="Mode", ax=ax, s=100, palette="tab10")
        # ---- INLINE Y-LABEL FORMATTING ----
        unit = parameter_unit[i] 
        if "_" in param:
            base, sub = param.split("_", 1)
            if base.lower() == "eta":
                ylabel = rf"$\underline{{{base}}}_{{{sub}}}$"
            else:
                ylabel = rf"${base}_{{{sub}}}$"
        else:
            ylabel = rf"${param}$"
        ax.set_ylabel(f"{ylabel} [{unit}]")
        ax.set_title(f"({figure_order[i]})") 
        ax.set_xticks(list(season_map.values()))
        # ax.set_xticklabels(list(season_map.keys()))
        ax.grid(True)
        legend = ax.get_legend()
        if i == 2:
            if legend:
                legend.set_title("Mode")
                legend.set_bbox_to_anchor((1.02, 1))
                legend.set_loc("upper left")
        else:
            if legend:
                legend.remove()
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    # Set shared x-label
    for ax in axes[-ncols:]:
        ax.set_xlabel("")
        ax.tick_params(axis='x',rotation=45)
    return fig, axes
    

#only works for variance parameter
def Plot_η_depth_variablity_with_mode(filtered_temp_dict,η_name,mode,colors,):
    cols = len(η_name)
    rows= 1
    fig, axes = plt.subplots(rows, cols,sharey=True)
    axes = axes.flatten()
    for parameter_order, parameter_name in enumerate(η_name):
        parameter_df_plot_median_all_mode = []
        parameter_df_plot_iqr_all_mode = []  
        for mode_number in mode:
            parameter_dict =  {key: variable for key, variable in filtered_temp_dict.items() if key[2] == parameter_name and key[1] == mode_number}
            # Create a list to hold the data for plotting
            parameter_df_plot = Processing.Transfer_to_plot_df(parameter_dict)
            parameter_df_plot_median = parameter_df_plot.groupby('depth')['values'].median()
            # Calculate the IQR (Q3 - Q1)
            parameter_df_plot_iqr = parameter_df_plot.groupby('depth')['values'].quantile(0.75) - parameter_df_plot.groupby('depth')['values'].quantile(0.25)
            # append to list
            parameter_df_plot_median_all_mode.append(parameter_df_plot_median)
            parameter_df_plot_iqr_all_mode.append(parameter_df_plot_iqr)
            #plot
            y_offset = 2
            axes[parameter_order].errorbar(parameter_df_plot_median,
                                           parameter_df_plot_median.index + mode_number * y_offset,
                                           xerr=parameter_df_plot_iqr, color=colors[mode_number],
                                           capsize=3, alpha=0.8)
            axes[parameter_order].set_xlabel('value (K)')
            
        #calculate the sum
        parameter_df_plot_median_sum = pd.DataFrame(sum(df for df in parameter_df_plot_median_all_mode))
        parameter_df_plot_iqr_sum = pd.DataFrame(sum(df ** 2 for df in parameter_df_plot_iqr_all_mode) ** 0.5)  # Summing variances
        # Plot the sum of modes using IQR for error bars
        axes[parameter_order].errorbar(parameter_df_plot_median_sum['values'],
                                       parameter_df_plot_median.index + (mode_number + 1) * y_offset,
                                       xerr=parameter_df_plot_iqr_sum['values'], color=colors[-1],
                                       capsize=3, label='Sum')
        axes[parameter_order].set_title(f'{parameter_name}')
        axes[parameter_order].grid(True)
        
     # Create a legend for modes
    modes = mode+['Sum']
    modes_legend = {mode_number: color for mode_number, color in zip(modes, colors)}
    handles = [plt.Rectangle((0, 0), 1, 1, color=modes_legend[mode_number]) for mode_number in modes]
    labels = modes
    fig.legend(handles, labels, title='Modes', loc=(0.88, 0.25), bbox_to_anchor=(1, 0.4))  
    
    # Set y-axis label for appropriate subplots
    for i in range(len(axes)):
        if i % 2 == 0:
            axes[i].set_ylabel('Depth [m]')
     
    return fig, axes
  
def Plot_η_depth_variablity_compariation(filtered_temp_dict1,
                                         filtered_temp_dict2,
                                         parameter_name,depth_round_ratio=10):
    #plot set up
    cols = len(parameter_name)
    rows= 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5),sharey=True)
    axes = axes.flatten()
    figure_order = [list(string.ascii_lowercase)[i % 26] for i in range(len(parameter_name))]
    #calculate the sum
    for parameter_order, parameter_name in enumerate(parameter_name):
        # Filter dictionary based on parameter name and mode number
        parameter_dict = {key: variable for key, variable in filtered_temp_dict1.items() 
                              if key[1] == parameter_name}
        parameter_df = Processing.Transfer_to_plot_df(parameter_dict,depth_round_ratio=depth_round_ratio)
        # Calculate median and IQR for each depth level    
        grouped_depth = parameter_df.groupby('depths_round')['values']
        # Calculate median and IQR for each depth level
        median_all_modes = grouped_depth.median()
        upper_error      = grouped_depth.quantile(0.75) - grouped_depth.median()
        lower_error      = grouped_depth.median() - grouped_depth.quantile(0.25)
        #plot
        axes[parameter_order].errorbar(median_all_modes,
                                       median_all_modes.index,
                                       xerr=[lower_error,upper_error], color='black',
                                       capsize=3, label = 'Disp')
        
        #Temp parameters
        Temp_parameter_df_plot_median = filtered_temp_dict2.groupby('depth_round')[parameter_name].median()
        Temp_parameter_df_plot_upper_error    = filtered_temp_dict2.groupby('depth_round')[parameter_name].quantile(0.75)-Temp_parameter_df_plot_median
        Temp_parameter_df_plot_lower_error    = Temp_parameter_df_plot_median - filtered_temp_dict2.groupby('depth_round')[parameter_name].quantile(0.25)
        #plot
        axes[parameter_order].errorbar(Temp_parameter_df_plot_median,
                                       Temp_parameter_df_plot_median.index,
                                       xerr=[Temp_parameter_df_plot_lower_error,Temp_parameter_df_plot_upper_error], color='darkcyan',
                                        capsize=3,label = 'Temp',alpha=0.5)
        
        axes[parameter_order].set_title(f'{parameter_name}')
        axes[parameter_order].grid(True)
   
    # Create a custom legend for the locations
    axes[2].legend(bbox_to_anchor=(1, 0.4))
    axes[0].set_ylabel('Depth [m]')
    axes[1].set_xlabel('value [K]')
        
    return fig,axes

def Plot_Model_type_distribution(df):
    # Unique values for reindexing
    unique_sites = df['Site'].unique()
    unique_model_types = df['model_type'].unique()
    unique_modes = df['Mode'].unique()

    # Table grouped by 'Site'
    site_table = df.pivot_table(index='Site', 
                                columns='model_type', 
                                aggfunc='size', 
                                fill_value=0)
    site_table = site_table.reindex(index=unique_sites, columns=unique_model_types, fill_value=0)
    site_table['Total'] = site_table.sum(axis=1)
    total_column_site = site_table.sum(axis=0)
    total_column_site.name = 'Total'
    site_table = site_table.append(total_column_site)

    # Table grouped by 'Mode'
    mode_table = df.pivot_table(index='Mode', 
                                columns='model_type', 
                                aggfunc='size', 
                                fill_value=0)
    mode_table = mode_table.reindex(index=unique_modes, columns=unique_model_types, fill_value=0)
    mode_table['Total'] = mode_table.sum(axis=1)
    total_column_mode = mode_table.sum(axis=0)
    total_column_mode.name = 'Total'
    mode_table = mode_table.append(total_column_mode)

    # Display the tables
    print("Table grouped by Site:")
    print(site_table)
    print("\nTable grouped by Mode:")
    print(mode_table)




def Plot_map(bathmetry_file,M2tide_SSH_file,site_names,lat_list,lon_list):
    
    #Read the tide SSH
    M2tide_SSH  = xr.open_dataset(M2tide_SSH_file)
    # Define the latitude and longitude ranges (NWS and Timor sea)
    min_lat, max_lat = -22, -8
    min_lon, max_lon = 110, 135
    # Extract variables
    X_SSH = M2tide_SSH['longitude']
    Y_SSH = M2tide_SSH['latitude']
    # M2re_SSH = M2tide_SSH['M2re']
    M2re_SSH = np.sqrt(np.power(M2tide_SSH['M2re'],2)+np.power(M2tide_SSH['M2im'],2))
    # Find the indices corresponding to the latitude and longitude ranges
    idx_X_SSH = np.where((X_SSH>min_lon) & (X_SSH<max_lon))
    X_SSH = X_SSH[(X_SSH>min_lon) & (X_SSH<max_lon)]
    idx_Y_SSH = np.where((Y_SSH>min_lat) & (Y_SSH<max_lat))
    Y_SSH = Y_SSH[(Y_SSH>min_lat) & (Y_SSH<max_lat)]
    M2re = M2re_SSH.sel(longitude = X_SSH,latitude = Y_SSH)

    #Read the bathmetry
    # Load bathymetric data
    bathmetry = xr.open_dataset(bathmetry_file)
    # Extract variables
    X_bath = bathmetry['lon']
    Y_bath = bathmetry['lat']
    idx_X_bath = np.where((X_bath>min_lon) & (X_bath<max_lon))
    X_bath = X_bath[(X_bath>min_lon) & (X_bath<max_lon)]
    idx_Y_bath = np.where((Y_bath>min_lat) & (Y_bath<max_lat))
    Y_bath = Y_bath[(Y_bath>min_lat) & (Y_bath<max_lat)]
    topo = bathmetry['elevation'].sel(lon = X_bath,lat= Y_bath)


    # Location data
    locations = {site: {'latitude': lat, 'longitude': lon} for site, lat, lon in zip(site_names, lat_list, lon_list)}
    #Plot the map
    # Create a map using PlateCarree projection
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(25, 12))
    # Specify the contour levels you want to display
    contour_levels = [-500,-200]
    # Plot contour lines for specified levels
    contour_plot = plt.contour(X_bath, Y_bath, topo, levels=contour_levels,colors='gray',linestyles='solid')
    # Add contour labels
    # plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.0f',colors = 'black')    
     
    # Plot the contour plot on top of the map
    contour_plot = ax.contourf(X_SSH, Y_SSH, M2re*100, cmap='cmo.amp')
    # Add colorbar inside the figure
    cax = ax.inset_axes([0.65, 0.08, 0.3, 0.04])  # [left, bottom, width, height]
    cbar = plt.colorbar(contour_plot, cax=cax,orientation='horizontal')
    cbar.set_label('M2 Amplitude [cm]',fontsize=25)
    
    # Set the extent of the map
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
    # Add Natural Earth land and ocean features
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=cfeature.COLORS['land'])
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor=cfeature.COLORS['water'])
    ax.add_feature(land, zorder=1)
    ax.add_feature(ocean, zorder=0)
    # Add coastlines and gridlines
    ax.coastlines()
    # Use Cartopy's gridlines to control tick sizes
    gridlines = ax.gridlines(draw_labels=True)
    # Add markers and labels for specified locations
    for location, data in locations.items():
        ax.plot(data['longitude'], data['latitude'], 'ro', markersize=10, transform=ccrs.PlateCarree())
        ax.text(data['longitude'] + 0.5, data['latitude'], location, transform=ccrs.PlateCarree(),fontsize=25,fontweight='bold')
    
    ax.set_title('')
    ax.grid(True)
    # Adjust layout
    fig.tight_layout()
    # Show the map
    plt.show()
    return fig, ax