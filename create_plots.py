# Import libraries (requires: pandas, seaborn, matplotlib, xlrd)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

plotFig3 = True
plotFig4 = True
plotFig5 = True
plotFig6 = True

# ******************************
# Inputs
# ******************************

# Results file
filename = "model.xlsx"
sheetname = "Summary"
skiprows = 0
nrows = 35
index_col = 0

# VARIABLES TO CHANGE TO AFFECT GRAPHS
dpi = 1000
context = "notebook"
style = "whitegrid"
colors = sns.color_palette("Paired")
colors_alt = sns.color_palette("colorblind")

captureTypes = ['None', 'Amine-based', 'Ammonia', 'CaL', 'CL', 'Membrane', 'Oxy-fuel', 'Selexol', 'SEWGS']
capture_colors = [colors_alt[0], colors_alt[0], colors_alt[1], colors_alt[2], colors_alt[3], colors_alt[4],
                  colors_alt[5], colors_alt[7], colors_alt[9]]
capture_markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'o', 'X']

label_dict_capture = {'None': 'None', 'Amine-based': 'Amine-based', 'Ammonia': 'Ammonia', 'CaL': 'CaL', 'CL': 'CL',
                      'Membrane': 'Membrane', 'Oxy-fuel': 'Oxy-fuel', 'Selexol': 'Selexol', 'SEWGS': 'SEWGS'}

plantTypes = ['Coal (Steam)', 'Coal (IGCC)', 'NG (NGCC)', 'Biomass (Steam)', 'Biomass (IGCC)', 'Blend (Steam)',
              'Blend (IGCC)']
plant_colors = [colors[5], colors[4], colors[7],
                colors[3], colors[2], colors[1], colors[0]]

label_dict = {'Coal (Steam)': 'Coal w/ CC (Steam)', 'Coal (IGCC)': 'Coal w/ CC (IGCC)', 'NG (NGCC)': 'NG w/ CC (NGCC)',
              'Biomass (Steam)': 'Biomass w/ CC (Steam)', 'Biomass (IGCC)': 'Biomass w/ CC (IGCC)',
              'Blend (Steam)': '50% Blend w/ CC (Steam)', 'Blend (IGCC)': '50% Blend w/ CC (IGCC)'}

label_dict2 = {'Coal (Steam)': 'Coal w/ CC (Steam)', 'Coal (IGCC)': 'Coal w/ CC (IGCC)', 'NG (NGCC)': 'NG w/ CC (NGCC)',
               'Biomass (Steam)': 'Biomass (Steam)', 'Biomass (IGCC)': 'Biomass (IGCC)',
               'Blend (Steam)': '50% Blend (Steam)', 'Blend (IGCC)': '50% Blend (IGCC)'}

dot_size = 100
marker_size = 10
markeredgewidth = 2

# ******************************
# Prepare Data
# ******************************

# Read-in results file
df = pd.read_excel(filename, sheet_name=sheetname, skiprows=skiprows, nrows=nrows, index_col=index_col)
# Drop empty columns

df = df.dropna(axis=1)
df = df.transpose()
df = df.drop(["VariableLabel", "Units"])

df_smry = pd.read_csv('summary.csv')
df_smry2 = pd.read_csv('summary2.csv')

# ******************************
# Plot 3
if plotFig3:
    savename = "Fig3_GWP_vs_EROI_byPowerPlant.png"
    # ******************************

    x_var = 'EROI_mean'
    x_var_low = 'EROI_min'
    x_var_hi = 'EROI_max'
    x_label = 'Energy Return On Investment (-)'
    x_convert = 1.0
    x_lims0 = [0.0, 18.0]
    x_lims1 = [6.0, 12.0]

    y_var = 'GWP_mean'
    y_var_low = 'GWP_min'
    y_var_hi = 'GWP_max'
    y_label = 'Global Warming Potential (kg CO$_2$e/kwh)'
    y_convert = 1.0
    y_lims0 = [-3.200, 0.500]
    y_lims1 = [-1.000, -0.200]

    # Column width guidelines https://www.elsevier.com/authors/author-schemas/artwork-and-media-instructions/artwork-sizing
    # Single column: 90mm = 3.54 in
    # 1.5 column: 140 mm = 5.51 in
    # 2 column: 190 mm = 7.48 i
    width = 7.48  # inches
    height = 7.0  # inches

    # create figure
    f, a = plt.subplots(1, 1)

    # create inset
    # axins = zoomed_inset_axes(a, zoom=1.5, loc='lower right')
    axins = zoomed_inset_axes(a, zoom=1.8, loc='lower right', bbox_to_anchor=(0.975, 0.1), bbox_transform=a.transAxes)

    sns.set_style("white", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})
    sns.set_context("paper")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    PrePostOxys = ['Post', 'Pre', 'Oxy']
    marker_fills = ['Y', 'N', 'Y']
    marker_size = 8
    markeredgewidth = 1.5
    elinewidth = 1.0

    # Iterate through subplots
    for i in range(2):

        if i == 0:
            ax = a
        else:
            ax = axins

        # Iterate through plant types
        for plantType, plant_color in zip(plantTypes, plant_colors):

            # Iterate through capture technologies
            for captureType, capture_marker in zip(captureTypes[1:], capture_markers[1:]):

                for PrePostOxy, marker_fill in zip(PrePostOxys, marker_fills):
                    # Select entries of interest

                    if i == 0:
                        df2 = df_smry[(df_smry.powerPlantType == plantType) & (df_smry.captureType1 == captureType)
                                      & (df_smry.PrePostOxy == PrePostOxy)]
                        # Convert
                        if len(df2) > 0:
                            x = float(df2.loc[:, x_var] * x_convert)
                            x_low = float(df2.loc[:, x_var_low] * x_convert)
                            x_hi = float(df2.loc[:, x_var_hi] * x_convert)

                            y = float(df2.loc[:, y_var] * y_convert)
                            y_low = float(df2.loc[:, y_var_low] * y_convert)
                            y_hi = float(df2.loc[:, y_var_hi] * y_convert)

                            # calculate error bars
                            yerr_low = y - y_low
                            yerr_hi = y_hi - y

                            xerr_hi = x_hi - x
                            xerr_low = x - x_low

                            # Plot Data
                            # ax.plot(x, y, linestyle='', marker=capture_marker, markersize=marker_size,
                            #         markeredgewidth=markeredgewidth, markeredgecolor=plant_color, markerfacecolor='None')
                            if marker_fill == 'Y':
                                ax.errorbar(x, y, xerr=[[xerr_low], [xerr_hi]], yerr=[[yerr_low], [yerr_hi]],
                                            linestyle='',
                                            marker=capture_marker, markersize=marker_size,
                                            markeredgewidth=markeredgewidth,
                                            markeredgecolor=plant_color, markerfacecolor=plant_color,
                                            ecolor=plant_color,
                                            elinewidth=elinewidth)
                            else:
                                ax.errorbar(x, y, xerr=[[xerr_low], [xerr_hi]], yerr=[[yerr_low], [yerr_hi]],
                                            linestyle='',
                                            marker=capture_marker, markersize=marker_size,
                                            markeredgewidth=markeredgewidth,
                                            markeredgecolor=plant_color, markerfacecolor='None', ecolor=plant_color,
                                            elinewidth=elinewidth)
                            # print(str(plant_color))
                            # print(capture_marker)  # print(y)

                    else:
                        # df2 = df_smry[(df_smry.powerPlantType == plantType) & (df_smry.captureType1 == captureType)
                        #               & (df_smry.PrePostOxy == PrePostOxy)]
                        df2 = df[(df.powerPlantType == plantType) & (df.captureType1 == captureType) & (
                                df.PrePostOxy == PrePostOxy)]

                        x = list(df2.loc[:, 'EROI'] * x_convert)
                        y = list(df2.loc[:, 'GWP_total'] * y_convert / 1000.0)

                        # Plot Data
                        if marker_fill == 'Y':
                            ax.plot(x, y, linestyle='', marker=capture_marker, markersize=marker_size,
                                    markeredgewidth=markeredgewidth, markeredgecolor=plant_color,
                                    markerfacecolor=plant_color)
                        else:
                            ax.plot(x, y, linestyle='', marker=capture_marker, markersize=marker_size,
                                    markeredgewidth=markeredgewidth, markeredgecolor=plant_color,
                                    markerfacecolor='None')

        # Despine and remove ticks
        if i == 0:
            sns.despine(ax=ax, )
            ax.tick_params(top=False, right=False)

        # Labels
        if i == 0:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        # Axis Limits
        if i == 0:
            if len(x_lims0) == 2:
                ax.set_xlim(left=x_lims0[0], right=x_lims0[1])
            if len(y_lims0) == 2:
                ax.set_ylim(bottom=y_lims0[0], top=y_lims0[1])
        elif i == 1:
            if len(x_lims1) == 2:
                ax.set_xlim(left=x_lims1[0], right=x_lims1[1])
            if len(x_lims1) == 2:
                ax.set_ylim(bottom=y_lims1[0], top=y_lims1[1])

        # Caption labels
        caption_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        if i == 0:
            ax.text(0.025, 0.975, caption_labels[i], horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize='medium', fontweight='bold')
        elif i == 1:
            ax.text(0.05, 0.9, caption_labels[i], horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize='medium', fontweight='bold')

        if i == 0:
            # Plot vertical reference lines and labels
            EROI_breakeven = 1.0
            # EROI_corn_ethanol_CCS = 1.54
            # EROI_sugarcane_CCS = 3.89
            if len(y_lims0) != 2:
                y_lims0 = [-500, 1000]
            ax.plot([EROI_breakeven, EROI_breakeven], y_lims0, '--', color=[0.5, 0.5, 0.5])
            # ax.plot([EROI_sugarcane_CCS, EROI_sugarcane_CCS], y_lims0, '--', color=[0.5, 0.5, 0.5])
            # ax.plot([EROI_corn_ethanol_CCS, EROI_corn_ethanol_CCS], y_lims0, '--', color=[0.5, 0.5, 0.5])

            v_space = 0.05
            h_space = 0.05
            ax.text(EROI_breakeven - h_space, y_lims0[0] + v_space, 'EROI break even', horizontalalignment='right',
                    verticalalignment='bottom',
                    rotation=90)
            # ax.text(EROI_corn_ethanol_CCS - h_space, y_lims0[0] + v_space, 'Corn ethanol CCS',
            #         horizontalalignment='right',
            #         verticalalignment='bottom',
            #         rotation=90)
            # ax.text(EROI_sugarcane_CCS - h_space, y_lims0[0] + v_space, 'Sugarcane ethanol CCS',
            #         horizontalalignment='right',
            #         verticalalignment='bottom',
            #         rotation=90)

            # Plot horizontal reference line and label
            if len(x_lims0) != 2:
                x_lims0 = [0.0, 20.0]
            ax.plot(x_lims0, [0.0, 0.0], '--', color=[0.5, 0.5, 0.5])

            ax.text(x_lims0[1], 0.0 - v_space, 'Carbon neutral', horizontalalignment='right', verticalalignment='top',
                    rotation=0)

    # Set size
    f = plt.gcf()
    f.set_size_inches(width, height)

    # Add rectangle that represents subplot2
    rect = plt.Rectangle((x_lims1[0], y_lims1[0]), x_lims1[1] - x_lims1[0], y_lims1[1] - y_lims1[0], facecolor="black",
                         alpha=0.05)
    # rect = plt.Rectangle((x_lims1[0], y_lims1[0]), x_lims1[1] - x_lims1[0], y_lims1[1] - y_lims1[0], edgecolor='black',
    #                      facecolor='None', )

    a.add_patch(rect)
    a.text(x_lims1[1], y_lims1[0], 'Extent of B', horizontalalignment='right', verticalalignment='top', rotation=0)

    # Legend
    # Iterate through plant technologies
    patches = []
    for plantType, plant_color in zip(plantTypes, plant_colors):
        patches.append(mpatches.Patch(color=plant_color, label=label_dict[plantType]))

    leg1 = a.legend(handles=patches, bbox_to_anchor=(0.15, -0.11), loc="upper left", title='Power Plants', ncol=1)

    # Iterate through capture technologies
    symbols = []
    for captureType, capture_color, capture_marker in zip(captureTypes[1:], capture_colors[1:], capture_markers[1:]):
        symbols.append(mlines.Line2D([], [], color='black', linestyle='', marker=capture_marker, markersize=9,
                                     markerfacecolor='None', markeredgewidth=1.5,
                                     label=label_dict_capture[captureType]))
    leg2 = a.legend(handles=symbols, bbox_to_anchor=(0.4575, -0.11), loc="upper left", title='Technology', ncol=1)

    # w0 = a.get_window_extent().width
    # w1 = leg1.get_window_extent().width / w0
    # w2 = leg2.get_window_extent().width / w0
    # w3 = leg2.get_window_extent().width / w0
    # w = (0.5 * (w0 - w1 - w2) + w1)/w0

    # Pre/Post
    patches2 = [mpatches.Patch(edgecolor='black', facecolor='None', label='Pre'),
                mpatches.Patch(edgecolor='black', facecolor='black', label='Post')]
    leg3 = a.legend(handles=patches2, bbox_to_anchor=(0.65, -0.11), loc="upper left", title='Capture Type')

    a.add_artist(leg1)
    a.add_artist(leg2)
    a.add_artist(leg3)
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=12)
    # plt.tight_layout()
    plt.subplots_adjust(top=0.95,
                        bottom=0.335,
                        left=0.11,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig(savename, dpi=dpi, bbox_extra_artists=(leg1, leg2, leg3))
    # plt.savefig(savename, dpi=dpi)

# ******************************
# Plot 4
if plotFig4:
    savename = "Fig4_WU_vs_EROI_byPowerPlant.png"
    # ******************************

    x_var = 'EROI_mean'
    x_var_low = 'EROI_min'
    x_var_hi = 'EROI_max'
    x_label = 'EROI (-)'
    x_convert = 1.0
    x_lims = [0.0, 20.0]

    y_var = "WU_mean"
    y_var_low = 'WU_min'
    y_var_hi = 'WU_max'
    y_label = "Water Use (l/kWh)"
    y_convert = 1.0
    y_lims = [0, 4.0]

    # Column width guidelines https://www.elsevier.com/authors/author-schemas/artwork-and-media-instructions/artwork-sizing
    # Single column: 90mm = 3.54 in
    # 1.5 column: 140 mm = 5.51 in
    # 2 column: 190 mm = 7.48 i
    width = 7.48  # inches
    height = 5.5  # inches

    f, axes = plt.subplots(1, 2, sharey=True)  # ,constrained_layout=True)

    sns.set_style("white", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})
    sns.set_context("paper")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    PrePostOxys = ['Post', 'Pre', 'Oxy']
    marker_fills = ['Y', 'N', 'Y']
    marker_size = 7
    markeredgewidth = 1.5

    for i, ax in enumerate(axes):

        if i == 0:
            # Select pre-combustion technologies (including oxy)
            df2 = df_smry[(df_smry.loc[:,
                           "PrePostOxy"] == 'Pre')]  # df_captureTypes = df2.captureType1.unique()  # df_captureTypes = np.sort(df_captureTypes)  # df_powerPlantTypes = df2.powerPlantType.unique()
        else:
            # Select post-combustion technologies (including oxy)
            df2 = df_smry[(df_smry.loc[:,
                           "PrePostOxy"] == 'Post') | (df_smry.loc[:,
                                                       "PrePostOxy"] == 'Oxy')]  # df_captureTypes = df2.captureType1.unique()  # df_captureTypes = np.sort(df_captureTypes)  # df_powerPlantTypes = df2.powerPlantType.unique()

        # Iterate through plant types
        for plantType, plant_color in zip(plantTypes, plant_colors):

            # Iterate through capture technologies
            for captureType, capture_marker in zip(captureTypes[1:], capture_markers[1:]):
                # Select entries of interest
                df3 = df2[(df2.powerPlantType == plantType) & (df2.captureType1 == captureType)]

                if len(df3) > 0:
                    # Convert
                    x = float(df3.loc[:, x_var] * x_convert)
                    x_low = float(df3.loc[:, x_var_low] * x_convert)
                    x_hi = float(df3.loc[:, x_var_hi] * x_convert)

                    y = float(df3.loc[:, y_var] * y_convert)
                    y_low = float(df3.loc[:, y_var_low] * y_convert)
                    y_hi = float(df3.loc[:, y_var_hi] * y_convert)

                    # calculate error bars
                    yerr_low = y - y_low
                    yerr_hi = y_hi - y

                    xerr_hi = x_hi - x
                    xerr_low = x - x_low

                    # Plot Data
                    # ax.plot(x, y, linestyle='', marker=capture_marker, markersize=marker_size,
                    #         markeredgewidth=markeredgewidth, markeredgecolor=plant_color, markerfacecolor='None')
                    ax.errorbar(x, y, xerr=[[xerr_low], [xerr_hi]], yerr=[[yerr_low], [yerr_hi]], linestyle='',
                                marker=capture_marker, markersize=marker_size, markeredgewidth=markeredgewidth,
                                markeredgecolor=plant_color, markerfacecolor='None', ecolor=plant_color)
                    # print(str(plant_color))
                    # print(capture_marker)  # print(y)

        # Despine and remove ticks
        sns.despine(ax=ax, )
        ax.tick_params(top=False, right=False)

        # Plot reference lines
        h_space = 0.055
        v_space = 0.05
        EROI_breakeven = 1.0
        eff = 0.25

        WU_switchgrass = 11.2388 / 1000.0 * 3.6 / eff  # convert from cm^3 / MJ to l/kWh
        # WU_poplar = 5.1948 / 1000.0 * 3.6 / eff
        # WU_corn = 12.6939 / 1000.0 * 3.6 / eff
        # WU_forest_residue = 1.74298 / 1000.0 * 3.6 / eff # convert from cm^3 / MJ to l/kWh
        # WU_willow = 4.42758

        if len(y_lims) != 2:
            y_lims = [0, 4000]

        ax.plot([EROI_breakeven, EROI_breakeven], y_lims, '--', color=[0.5, 0.5, 0.5])
        ax.text(EROI_breakeven - h_space, y_lims[1] - v_space, 'EROI break even', horizontalalignment='right',
                verticalalignment='top',
                rotation=90)

        if len(x_lims) != 2:
            x_lims = [0.0, 20.0]
        rect = plt.Rectangle((x_lims[0], 0.0), x_lims[1] - x_lims[0], WU_switchgrass,
                             facecolor="black",
                             alpha=0.1)
        ax.add_patch(rect)
        ax.text((x_lims[1] - x_lims[0]) / 2.0, WU_switchgrass / 2.0, 'Switchgrass Cultivation Water Use',
                horizontalalignment='center', verticalalignment='center', rotation=0)

        # Labels
        ax.set_xlabel(x_label)
        if i == 0:
            ax.set_ylabel(y_label)

        # Axis Limits
        if len(x_lims) == 2:
            ax.set_xlim(left=x_lims[0], right=x_lims[1])
        if len(y_lims) == 2:
            ax.set_ylim(bottom=y_lims[0], top=y_lims[1])

        # Caption labels
        caption_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        plt.text(0.05, 1.05, caption_labels[i], horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, fontsize='medium', fontweight='bold')

        # Additional Labels
        if i == 0:
            ax.text(0.5, 1.05, 'Pre-combustion', horizontalalignment='center', verticalalignment='center',
                    rotation=0,
                    transform=ax.transAxes)
        elif i == 1:
            ax.text(0.5, 1.05, 'Post-combustion', horizontalalignment='center', verticalalignment='center',
                    rotation=0,
                    transform=ax.transAxes)

    # Set size
    f = plt.gcf()
    f.set_size_inches(width, height)

    # Legend
    # Iterate through plant technologies (present in an alternative order)
    plantTypes2 = ['Coal (IGCC)', 'Biomass (IGCC)', 'Blend (IGCC)', 'NG (NGCC)',
                   'Coal (Steam)', 'Biomass (Steam)', 'Blend (Steam)']
    plant_colors2 = [colors[4], colors[2], colors[0], colors[7],
                     colors[5], colors[3], colors[1]]

    patches = []
    for plantType, plant_color in zip(plantTypes2, plant_colors2):
        patches.append(mpatches.Patch(color=plant_color, label=label_dict[plantType]))

    # leg1 = plt.legend(handles=patches, bbox_to_anchor=(1.0, 0.45), loc="lower left", title='Power Plants') # Side
    leg1 = axes[0].legend(handles=patches, bbox_to_anchor=(0.5, -0.15), loc="upper center", title='Power Plants',
                          ncol=2)  # Bottom

    # Iterate through capture technologies
    symbols = []
    for captureType, capture_color, capture_marker in zip(captureTypes[1:], capture_colors[1:],
                                                          capture_markers[1:]):
        symbols.append(mlines.Line2D([], [], color='black', linestyle='', marker=capture_marker, markersize=9,
                                     markerfacecolor='None', markeredgewidth=1.5,
                                     label=label_dict_capture[captureType]))
    # leg2 = plt.legend(handles=symbols, bbox_to_anchor=(1.0, -0.2), loc="lower left", title='Capture Type') # Side
    leg2 = axes[1].legend(handles=symbols, bbox_to_anchor=(0.5, -0.15), loc="upper center", title='Capture Type',
                          ncol=2)  # Bottom

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(savename, dpi=dpi, bbox_extra_artists=(leg1, leg2))

# ******************************
# Plot 5 - Hot Spot Analysis
if plotFig5:
    savename = "Fig5_Hot_Spot_Analysis.png"
    # ******************************

    # Plot variables
    plt_plantTypes = ['Biomass (Steam)', 'Biomass (IGCC)']
    facets = ['Fuel Production\nand Transport', 'Power Generation', 'Carbon Capture',
              'Solvent Production\nand Transport']
    facet_label = 'Lifecycle Stage (-)'
    x_var = 'captureType1'
    y_labels = ["GWP\n(kg CO$_2$e/kwh)", "Energy Use\n(MJ/kWh)", "Water Use\n(l/kWh)"]
    y_converts = [1.0 / 1000.0, 1.0, 1.0e-3]

    colors = sns.color_palette('colorblind')
    entry_colors = [colors[0], colors[3], colors[2], colors[4], colors[5],
                    colors[0], colors[1], colors[2], colors[6], colors[7]]
    # entry_hatch = ['','','','','/','/','/','/','/']

    # Select data of interest
    df2 = df[
        ((df.loc[:, 'powerPlantType'] == plt_plantTypes[0]) | (df.loc[:, 'powerPlantType'] == plt_plantTypes[1])) & (
                df.loc[:, 'captureType1'] != 'None')]
    df2.loc[:, 'comb_type'] = df2.powerPlantType + ' - ' + df2.captureType1
    entries = np.sort(df2.comb_type.unique())
    entries = ['Biomass (Steam) - Amine-based', 'Biomass (Steam) - Ammonia',
               'Biomass (Steam) - CaL', 'Biomass (Steam) - Membrane',
               'Biomass (Steam) - Oxy-fuel', 'Biomass (IGCC) - Amine-based', 'Biomass (IGCC) - CL',
               'Biomass (IGCC) - CaL', 'Biomass (IGCC) - Selexol', 'Biomass (IGCC) - SEWGS']

    # Column width guidelines https://www.elsevier.com/authors/author-schemas/artwork-and-media-instructions/artwork-sizing
    # Single column: 90mm = 3.54 in
    # 1.5 column: 140 mm = 5.51 in
    # 2 column: 190 mm = 7.48 i
    width = 7.48  # inches
    height = 9.0  # inches

    # Create plot
    f, a = plt.subplots(3, len(facets), sharex='col', sharey='row')

    # Set size
    f.set_size_inches(width, height)

    # Set style and context
    sns.set_style("white", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})
    sns.set_context("paper")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    # iterate through y-variables
    for i, (y_label, y_convert) in enumerate(zip(y_labels, y_converts)):

        if i == 0:
            series = ['GWP_FG', 'GWP_PG', 'GWP_CC', 'GWP_SP', 'GWP_ST']
        elif i == 1:
            series = ['EU_FG', 'EU_PG', 'EU_CC', 'EU_SP', 'EU_ST']
        elif i == 2:
            series = ['WU_FG', 'WU_PG', 'WU_CC', 'WU_SP', 'WU_ST']

        # iterate through facets
        for j, facet in enumerate(facets):

            # Get series variable
            serie = series[j]
            if j == 3:
                serie2 = series[j + 1]

            # Access subplot
            if j == 0:
                ax = a[i, 0]
            elif j == 1:
                ax = a[i, 2]
            elif j == 2:
                ax = a[i, 3]
            elif j == 3:  # Insert Solvent P&T before PG
                ax = a[i, 1]

            # Iterate through entries (columns)
            for k, entry in enumerate(entries):

                # access entry
                # df3 = df2.comb_type

                df3 = df2[(df2.loc[:, 'comb_type'] == entry)]

                # Fill-in values
                if j < 3:
                    mean = df3.loc[:, serie].mean()
                    yerr = np.zeros((2, 1))
                    yerr[0] = df3.loc[:, serie].mean() - df3.loc[:, serie].min()
                    yerr[1] = df3.loc[:, serie].max() - df3.loc[:, serie].mean()
                else:
                    mean = df3.loc[:, serie].mean() + df3.loc[:, serie2].mean()
                    yerr = np.zeros((2, 1))
                    yerr[0] = (df3.loc[:, serie].mean() + df3.loc[:, serie2].mean()) - (
                            df3.loc[:, serie].min() + df3.loc[:, serie2].min())
                    yerr[1] = (df3.loc[:, serie].max() + df3.loc[:, serie2].max()) - (
                            df3.loc[:, serie].mean() + df3.loc[:, serie2].mean())
                # plot
                if k < 5:
                    ax.bar(k, mean * y_convert, 1.0, yerr=yerr * y_convert, color=entry_colors[k], hatch='///')
                else:
                    ax.bar(k, mean * y_convert, 1.0, yerr=yerr * y_convert, color=entry_colors[k])

            # only show y-labels on the left-most panels
            # plt.locator_params(axis='y', nbins=4)
            if j == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel("")

            # Move x-axis crossing to zero for GWP and EU
            if i == 0 or i == 1:
                ax.spines['bottom'].set_position('zero')

            # Customize x-axis
            if i < 2:
                ax.get_xaxis().set_visible(False)
            else:
                ax.set_xlabel(facet)
                ax.get_xaxis().set_ticks([])

            # Reduce number of y axis ticks
            # ax.set_yscale('symlog')
            # if i==0:
            #     ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            #     ax.set_ylim(top=2, bottom=-10.0)
            # elif i == 1:
            #     ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            #     ax.set_ylim(top=11, bottom=-11.0)
            # else:
            #     ax.yaxis.set_major_locator(plt.MaxNLocator(6))
            #     ax.set_ylim(top=10.0, bottom=0.0)

            if i == 0:
                ax.set_yscale('symlog')
                ax.yaxis.set_major_locator(plt.MaxNLocator(20))
                ax.set_ylim(top=10.0, bottom=-10.0)
            elif i == 1:
                ax.set_yscale('symlog')
                ax.yaxis.set_major_locator(plt.MaxNLocator(20))
                ax.set_ylim(top=10, bottom=-10.0)
            else:
                ax.set_yscale('symlog')
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                ax.set_ylim(top=10.0, bottom=0.0)

            # Despine and remove ticks
            if j == 0:
                sns.despine(ax=ax, )
                ax.tick_params(top=False, right=False)
            else:
                sns.despine(ax=ax, left=False)
                ax.tick_params(top=False, right=False)

    # Iterate through plant technologies
    patches = []
    # for serie_label, serie_color in zip(entries, entry_colors):
    for i, entry in enumerate(entries):
        if i < 5:
            patches.append(mpatches.Patch(facecolor=entry_colors[i], label=entry, hatch='///'))
        else:
            patches.append(mpatches.Patch(facecolor=entry_colors[i], label=entry))

    # leg1 = a[2, 0].legend(handles=patches, bbox_to_anchor=(1.0, 0.5), loc="center left", title='Stage')
    leg = a[2, 2].legend(handles=patches, bbox_to_anchor=(0.0, -0.45), ncol=2, loc="upper center")
    # a[2, 2].add_artist(leg)

    # Additional Labels
    ax = a[2, 2]
    ax.text(0.0, -0.35, facet_label, horizontalalignment='center', verticalalignment='center', rotation=0,
            transform=ax.transAxes)

    # Adjust layout
    plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.935, bottom=0.285, left=0.13, right=0.9)

    # Save figure
    f.align_ylabels(a[:, 0])
    plt.savefig(savename, dpi=dpi, bbox_extra_artists=(leg,))

# ******************************
# Plot 6
if plotFig6:
    savename = "Fig6_Sensitivity.png"
    # ******************************

    f, a = plt.subplots(3, 2, sharex="col", sharey="row")
    sns.set_style("whitegrid", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]})
    sns.set_context("paper")
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    # Column width guidelines https://www.elsevier.com/authors/author-schemas/artwork-and-media-instructions/artwork-sizing
    # Single column: 90mm = 3.54 in
    # 1.5 column: 140 mm = 5.51 in
    # 2 column: 190 mm = 7.48 i
    width = 7.48  # inches
    height = 5.5  # inches

    dot_size = 50
    marker_size = 6
    markeredgewidth = 1.5

    PrePostOxys = ['Post', 'Pre', 'Oxy']
    marker_fills = ['Y', 'N', 'N']
    marker_size = 8
    markeredgewidth = 1.5
    elinewidth = 1.0

    for i in range(3):

        if i == 0:
            # Y variable
            y_var = 'GWP_mean'
            y_var_low = 'GWP_min'
            y_var_hi = 'GWP_max'
            y_label = 'GWP (kg CO$_2$e/kwh)'
            y_convert = [1.0]
            y_lims = [-3.500, 1.000]
            y_ticks = []

        elif i == 1:
            # Y variable
            y_var = 'EROI_mean'
            y_var_low = 'EROI_min'
            y_var_hi = 'EROI_max'
            y_label = 'EROI (-)'
            y_convert = [1.0]
            y_lims = [0, 20]
            y_ticks = []

        else:
            # Y variable
            y_var = 'WU_mean'
            y_var_low = 'WU_min'
            y_var_hi = 'WU_max'
            y_label = "Water Use (l/kWh)"
            y_convert = [1.0]
            y_lims = [0, 4.0]
            y_ticks = []

        for j in range(2):

            ax = a[i][j]

            if j == 0:
                # X variable
                x_var = 'effReduction_mean'
                x_var_low = 'effReduction_min'
                x_var_hi = 'effReduction_max'
                x_label = 'Efficiency Reduction (%)'
                x_convert = [1.0]
                x_lims = [0, 20]
                x_ticks = []

            else:
                # X variable
                x_var = 'ccRate_mean'
                x_var_low = 'ccRate_min'
                x_var_hi = 'ccRate_max'
                x_label = 'Capture Rate (%)'
                x_convert = [1.0]
                x_lims = [75, 100]
                x_ticks = []

            # Iterate through plant technologies
            for plantType, plant_color in zip(plantTypes[3:5], plant_colors[3:5]):

                if plantType == 'Biomass (Steam)':
                    marker_fill = 'N'
                elif plantType == 'Biomass (IGCC)':
                    marker_fill = 'Y'

                # Iterate through capture technologies
                for captureType, capture_color, capture_marker in zip(captureTypes[1:], capture_colors[1:],
                                                                      capture_markers[1:]):

                    # Select entries of interest
                    df2 = df_smry2[(df_smry2.powerPlantType == plantType) & (df_smry2.captureType1 == captureType)]
                    # Convert
                    if len(df2) > 0:
                        x = float(df2.loc[:, x_var] * x_convert)
                        x_low = float(df2.loc[:, x_var_low] * x_convert)
                        x_hi = float(df2.loc[:, x_var_hi] * x_convert)

                        y = float(df2.loc[:, y_var] * y_convert)
                        y_low = float(df2.loc[:, y_var_low] * y_convert)
                        y_hi = float(df2.loc[:, y_var_hi] * y_convert)

                        # calculate error bars
                        yerr_low = y - y_low
                        yerr_hi = y_hi - y

                        xerr_hi = x_hi - x
                        xerr_low = x - x_low

                        # Plot Data
                        if marker_fill == 'Y':
                            ax.errorbar(x, y, xerr=[[xerr_low], [xerr_hi]], yerr=[[yerr_low], [yerr_hi]],
                                        linestyle='',
                                        marker=capture_marker, markersize=marker_size,
                                        markeredgewidth=markeredgewidth,
                                        markeredgecolor=capture_color, markerfacecolor=capture_color,
                                        ecolor=capture_color,
                                        elinewidth=elinewidth)
                        else:
                            ax.errorbar(x, y, xerr=[[xerr_low], [xerr_hi]], yerr=[[yerr_low], [yerr_hi]],
                                        linestyle='',
                                        marker=capture_marker, markersize=marker_size,
                                        markeredgewidth=markeredgewidth,
                                        markeredgecolor=capture_color, markerfacecolor='None', ecolor=capture_color,
                                        elinewidth=elinewidth)
                    # print(str(plant_color))
                    # print(capture_marker)  # print(y)

                # # Select entries of interest
                # df2 = df[(df.powerPlantType == plantType) & (df.captureType1 == captureType)]
                #
                # # Convert
                # x = list(df2.loc[:, x_var] * x_convert)
                # y = list(df2.loc[:, y_var] * y_convert)
                #
                # # Plot Data
                # # ax.scatter(x, y, s=dot_size, c=[plant_color], marker=capture_marker)
                #
                # ax.plot(x, y, linestyle='', marker=capture_marker, markersize=marker_size,
                #         markeredgewidth=markeredgewidth, markeredgecolor=plant_color, markerfacecolor='None')

                # Plot reference line
                # if i == 1 and j == 0:
                #     ax.plot([0, 20], [1.0, 1.0], 'k-')
                # if i == 1 and j == 0:
                #     ax.plot([0, 100], [1.0, 1.0],
                #             'k-')  # if len(x_lims) == 2:  #     ax.plot( x_lims, [1.0, 1.0], 'k-')  # else:  #     ax.plot([0, 20], [1.0, 1.0], 'k-')

                # X-axis Labels (Only bottom)
                if i == 2:
                    ax.set_xlabel(x_label)

                # Y-axis Labels (Only bottom)
                if j == 0:
                    ax.set_ylabel(y_label)

                # Axis Limits
                if len(x_lims) == 2:
                    ax.set_xlim(left=x_lims[0], right=x_lims[1])
                if len(y_lims) == 2:
                    ax.set_ylim(bottom=y_lims[0], top=y_lims[1])

                # if j == 0:
                sns.despine(ax=ax, )
                ax.tick_params(top=False,
                               right=False)  # else:  #     sns.despine(ax=ax, left=True)  #     ax.tick_params(top=False, right=False, left=False)

                # plt.tick_params(axis='x',  # changes apply to the x-axis  #                 which='both',  # both major and minor ticks are affected  #                 bottom=True,  # ticks along the bottom edge are off  #                 top=False,  # ticks along the top edge are off  #                 labelbottom=True)

            # Caption labels  # caption_labels = ['A', 'B', 'C', 'D', 'E', 'F']  # plt.text(0.1, 0.9, caption_labels[idx], horizontalalignment='center', verticalalignment='center',  #          transform=ax.transAxes, fontsize='medium', fontweight='bold')

    # Set size
    f = plt.gcf()
    f.set_size_inches(width, height)

    # Legend
    # Iterate through plant technologies
    # patches = []
    # Pre/Post
    patches = [mpatches.Patch(edgecolor='black', facecolor='None', label=label_dict[plantTypes[3]]),
               mpatches.Patch(edgecolor='black', facecolor='black', label=label_dict[plantTypes[4]])]
    # leg3 = a.legend(handles=patches2, bbox_to_anchor=(0.65, -0.11), loc="upper left", title='Capture Type')
    # for plantType, plant_color in zip(plantTypes[3:5], plant_colors[3:5]):
    #     patches.append(mpatches.Patch(color=plant_color, label=label_dict[plantType]))

    leg1 = a[0, 1].legend(handles=patches, bbox_to_anchor=(1.0, 0.0), loc="center left", title='Power Plants')

    # Iterate through capture technologies
    symbols = []
    for captureType, capture_color, capture_marker in zip(captureTypes[1:], capture_colors[1:], capture_markers[1:]):
        # symbols.append(mlines.Line2D([], [], color='black', linestyle='', marker=capture_marker, markersize=10,
        #                              label=label_dict_capture[captureType]))
        symbols.append(mlines.Line2D([], [], color=capture_color, linestyle='', marker=capture_marker, markersize=9,
                                     markerfacecolor='None', markeredgewidth=1.5,
                                     label=label_dict_capture[captureType]))
    leg2 = a[1, 1].legend(handles=symbols, bbox_to_anchor=(1.0, 0.0), loc="center left", title='Capture Type')

    # Reference arrows
    # common
    frac = 0.3
    y = 0.75
    width = 0.08

    # efficiency reduction
    xmin = 0.0
    xmax = 20.0
    xmid = (xmin + xmax) / 2.0
    length = frac * (xmax - xmin)
    a[2, 0].arrow(x=xmid + 0.5 * length, y=y, dx=-length, dy=0.0, width=width, color='black')
    a[2, 0].text(0.5, 0.1, 'Ideal', horizontalalignment='center', verticalalignment='center',
                 rotation=0, transform=a[2, 0].transAxes)

    xmin = 75
    xmax = 100
    xmid = (xmin + xmax) / 2.0
    length = frac * (xmax - xmin)
    a[2, 1].arrow(x=xmid - 0.5 * length, y=y, dx=length, dy=0.0, width=width, color='black')
    a[2, 1].text(0.5, 0.1, 'Ideal', horizontalalignment='center', verticalalignment='center',
                 rotation=0, transform=a[2, 1].transAxes)

    # Save figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    f.align_ylabels(a[:, 0])
    plt.savefig(savename, dpi=dpi, bbox_extra_artists=(leg1, leg2))
