# let's get what we need together
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PARAMS = {

    # parameters for DA modulation of PV firing rate threshold - marmoset
    'midpoint_b_i_marm': 0.36,
    'slope_b_i_marm': 7,
    'height_b_i_marm': -(-49.47 - (-52.13)),

    # parameters for DA modulation of PV firing rate threshold - macaque
    'midpoint_b_i_mac': 0.50,
    'slope_b_i_mac': 7,
    'height_b_i_mac': -(-49.47 - (-52.13)),

    # parameters for DA modulation of NMDARs
    'midpoint_nmda_da': 0.125,
    'slope_nmda_da': 100,
    'height_nmda_da': 132.048812005743 - 97.951187994257,

    # parameters for DA modulation of AMPARs
    'midpoint_ampa_da': 0.125,
    'slope_ampa_da': 100,
    'height_ampa_da': 101.36587467049533 - 78.63412532950467,

}

def sigmoid_DA(height, midpoint, slope):
    return np.exp(slope * (height - midpoint)) / (1 + np.exp(slope * (height - midpoint)))

def step_DA(height, midpoint):
    return np.heaviside(height - midpoint, 0.5)

def plot_all_sigmoids(DA_range, parameters, filename="plot.png"):
    fig = plt.figure(figsize=(8.4, 6.5), dpi=300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 25})

    # NMDA
    da_nmda_mod_factor = step_DA(DA_range, parameters['midpoint_nmda_da'])
    da_nmda_mod = 101 + 39 * da_nmda_mod_factor
    plt.plot(DA_range, da_nmda_mod, color='#990000', label=r'$G^\mathrm{E→E}_\mathrm{NMDA}$', linewidth=10)

    # AMPA
    da_ampa_mod_factor = step_DA(DA_range, parameters['midpoint_ampa_da'])
    da_ampa_mod = 99 - 19 * da_ampa_mod_factor
    plt.plot(DA_range, da_ampa_mod, color='#FC8D59', label=r'$G^\mathrm{E→E}_{\mathrm{AMPA}}, G^\mathrm{E→I}_\mathrm{AMPA}$', linewidth=10)

    axes = plt.gca()

    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    custom_labels_x = ['NO', 'LOW', 'MID', 'HIGH', 'VERY\nHIGH']
    plt.xlabel('D1R occupancy on Delay E cells')
    plt.ylabel('Conductance (%)') #Mean

    axes.set_xticklabels(custom_labels_x, fontsize=24)  # Set the y-tick labels to custom labels

    # Example y-axis tick locations and labels
    y_tick_locations = [80.00, 100.00, 140.00]
    y_tick_labels = ['80.00', '100.00', '140.00']

    axes.set_yticks(y_tick_locations)
    axes.set_yticklabels(y_tick_labels, fontsize=24)

    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.yaxis.set_ticks_position('left')
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    plt.legend(loc='best', fontsize=15) # Show legend in the upper left corner

    # Setting the ylim explicitly to ensure y-axis starts from a certain point
    axes.set_ylim(bottom=np.min([np.min(da_nmda_mod), np.min(da_ampa_mod)]) - 5)  # Adjust bottom limit as needed

    # Drawing dashed lines starting exactly from the x-axis
    y_lower_lim = axes.get_ylim()[0]  # This should now be the explicitly set bottom limit
    y_upper_lim = axes.get_ylim()[1]

    plt.tight_layout()

    # Save the figure
    plt.savefig(filename, dpi=600)
    plt.show()

DA_range = np.arange(-0.05, 1, 0.01)
plot_all_sigmoids(DA_range, PARAMS)

def plot_sigmoids_inh_thresh(DA_range, parameters, filename="plot.png"):
    b_i_marm_mod = sigmoid_DA(DA_range, parameters['midpoint_b_i_marm'], parameters['slope_b_i_marm'])
    b_i_mac_mod = sigmoid_DA(DA_range, parameters['midpoint_b_i_mac'], parameters['slope_b_i_mac'])

    b_i_marm = -49.47 + parameters['height_b_i_marm'] * b_i_marm_mod # modulated threshold for marmoset
    b_i_mac = -49.47 + parameters['height_b_i_mac'] * b_i_mac_mod # modulated threshold for macaque

    fig = plt.figure(figsize=(8.7, 6.5), dpi=300, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 25})

    plt.plot(DA_range, b_i_mac, color='#984EA3', linewidth=10)
    plt.plot(DA_range, b_i_marm, color='#FDBF6F', linewidth=10)

    # Set the x-axis tick positions
    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    custom_labels_x = ['NO', 'LOW', 'MID', 'HIGH', 'VERY\nHIGH']

    plt.legend(['MACAQUE-D1R','MARMOSET-D1R'],loc=('best'),fontsize=15)
    plt.ylabel('Firing Threshold (mV)')
    plt.xlabel('D1R stimulation of PV cells')

    plt.ylim(-52.30, -49.40) # Set y-axis limits

    axes = plt.gca()  # Get the current axes instance

    y_ticks = np.array([-49.55, -52.05])  # Create an array for y-tick positions
    custom_labels_y = ['-49.55', r'V$_{\mathit{th},\!\mathrm{min}}$'] # Create an array for custom y-tick labels

    axes.set_yticks(y_ticks)  # Set the y-tick positions
    axes.set_yticklabels(custom_labels_y)  # Set the y-tick labels to custom labels

    axes.set_xticklabels(custom_labels_x, fontsize=24)

    axes = plt.gca() # Get the current axes instance

    # Set y-axis label format
    #axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Remove top and right spines
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)

    plt.tight_layout()

    # Save the figure
    plt.savefig(filename, dpi=600)
    plt.show()

DA_range = np.arange(-0.05, 1, 0.01)
plot_sigmoids_inh_thresh(DA_range, PARAMS)