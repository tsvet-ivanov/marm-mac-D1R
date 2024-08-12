import numpy
import math
import brian2 as b2
from scipy.special import erf
from numpy.fft import rfft, irfft
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation
from brian2.monitors import SpikeMonitor
from brian2 import start_scope
import sys
from multiprocessing import Pool
import os
from neurodynex3.tools import plot_tools
import matplotlib.pyplot as plt

from brian2 import StateMonitor

N_excitatory = 2048
N_inhibitory_per_pop = 256 #512 #256 #256
weight_scaling_factor = 1
stimuli_width_deg = 30 #20 #60
stimulus_center_deg = 0 #0!!!!
t_stimulus_start = 400 * b2.ms #200 #400
t_stimuli_duration = 250 * b2.ms #250!!!!
sim_time = 5000. * b2.ms #4000.
distractor_center_deg = 180 #180
t_distractor_start = 2700 * b2.ms #2100

N_extern_poisson = 1000
poisson_firing_rate = 1.80 * b2.Hz #1.6 * b2.Hz #1.6 1.80

sigma_weight_profile_E2E = 14.4 #25 #20 #18 #25 #14.4
Jpos_excit2excit = 2.1 #2.1 #1.63#2.1 #7.1 # 1.62 #2 #1.62 #2

sigma_weight_profile_I2E = 14.4 #14.4 #4.5 #20 #32 #4.5 #14.4
Jpos_inhib2excit = 1.6 #1.6 #1.5 #1.22 #2.2 #1.9 #1.63 #1.5 #1.4

sigma_weight_profile_E2I = 14.4 #43.2 #25 #20 #32 #25 #14.4
Jpos_excit2inhib = 1.6 #1.5 #1.22 #2.2 #1.62 #1.63 #0.5

sigma_weight_profile_between_I2I = 14.4 #4.5 #4.5 #14.4
Jpos_between_inhib2inhib = 2.1 #2.1#1.63 #1.63 #1.5 #1.9

sigma_weight_profile_within_I2I = 14.4 #4.5 #4.5 #14.4
Jpos_within_inhib2inhib = 2.1 #2.1 #1 #1.63 #1.5 #1.9

monitored_subset_size_excit = 1024
monitored_subset_size_inhib = 128

# def generate_sim_params(repetitions=1):
#     num_variables = 6
#
#     NMDA_E_scaler_range = numpy.arange(0.85, 1.35 + 0.05, 0.05)
#     AMPA_excit_AND_inhib_scaler_range = numpy.arange(0.75, 1.15 + 0.05, 0.05)
#
#     v_firing_threshold_tuned_inhib_range = numpy.arange(-51.15, -50 + 0.05, 0.05)
#     NMDA_tuned_I_scaler_range = numpy.arange(0.85, 1.35 + 0.05, 0.05)
#
#     v_firing_threshold_opp_tuned_inhib_range = numpy.arange(-51.15, -50 + 0.05, 0.05)
#     NMDA_opp_tuned_I_scaler_range = numpy.arange(0.85, 1.35 + 0.05, 0.05)
#
#     num_NMDA_E_scaler_vals = NMDA_E_scaler_range.shape[0]
#     num_AMPA_excit_AND_inhib_scaler_vals = AMPA_excit_AND_inhib_scaler_range.shape[0]
#
#     num_v_firing_threshold_tuned_inhib_vals = v_firing_threshold_tuned_inhib_range.shape[0]
#     num_v_firing_threshold_opp_tuned_inhib_vals = v_firing_threshold_opp_tuned_inhib_range.shape[0]
#
#     num_simulations_1 = num_v_firing_threshold_tuned_inhib_vals * num_NMDA_E_scaler_vals * num_AMPA_excit_AND_inhib_scaler_vals * repetitions
#     simulation_params_1 = numpy.zeros((num_simulations_1, num_variables))
#     num_simulations_2 = num_v_firing_threshold_opp_tuned_inhib_vals * num_NMDA_E_scaler_vals * num_AMPA_excit_AND_inhib_scaler_vals * repetitions
#     simulation_params_2 = numpy.zeros((num_simulations_2, num_variables))
#
#     # In simulation_params_1, v_firing_threshold_tuned_inhib varies and v_firing_threshold_opp_tuned_inhib is constant
#     v_firing_threshold_opp_tuned_inhib_constant = -50
#     NMDA_opp_tuned_I_scaler_constant = 1.0
#
#     current_sim = 0
#     for current_v_firing_threshold_tuned_inhib_index in numpy.arange(0, num_v_firing_threshold_tuned_inhib_vals, 1):
#         for current_NMDA_scaler_index in numpy.arange(0, num_NMDA_E_scaler_vals, 1):
#             for current_AMPA_excit_AND_inhib_scaler_index in numpy.arange(0, num_AMPA_excit_AND_inhib_scaler_vals, 1):
#                 for _ in range(repetitions):
#                     simulation_params_1[current_sim, :] = [NMDA_E_scaler_range[current_NMDA_scaler_index],
#                                                            v_firing_threshold_tuned_inhib_range[current_v_firing_threshold_tuned_inhib_index],
#                                                            NMDA_tuned_I_scaler_range[current_NMDA_scaler_index],
#                                                            AMPA_excit_AND_inhib_scaler_range[current_AMPA_excit_AND_inhib_scaler_index],
#                                                            v_firing_threshold_opp_tuned_inhib_constant,
#                                                            NMDA_opp_tuned_I_scaler_constant
#                                                            ]
#                     current_sim = current_sim + 1
#
#     # In simulation_params_2, v_firing_threshold_opp_tuned_inhib varies and v_firing_threshold_tuned_inhib is constant
#     v_firing_threshold_tuned_inhib_constant = -50
#     NMDA_tuned_I_scaler_constant = 1.0
#
#     current_sim = 0
#     for current_v_firing_threshold_opp_tuned_index in numpy.arange(0, num_v_firing_threshold_opp_tuned_inhib_vals, 1):
#         for current_NMDA_scaler_index in numpy.arange(0, num_NMDA_E_scaler_vals, 1):
#             for current_AMPA_excit_AND_inhib_scaler_index in numpy.arange(0, num_AMPA_excit_AND_inhib_scaler_vals, 1):
#                 for _ in range(repetitions):
#                     simulation_params_2[current_sim, :] = [NMDA_E_scaler_range[current_NMDA_scaler_index],
#                                                            v_firing_threshold_tuned_inhib_constant,
#                                                            NMDA_tuned_I_scaler_constant,
#                                                            AMPA_excit_AND_inhib_scaler_range[current_AMPA_excit_AND_inhib_scaler_index],
#                                                            v_firing_threshold_opp_tuned_inhib_range[current_v_firing_threshold_opp_tuned_index],
#                                                            NMDA_opp_tuned_I_scaler_range[current_NMDA_scaler_index]
#                                                            ]
#                     current_sim = current_sim + 1
#
#     simulation_params = numpy.concatenate((simulation_params_1, simulation_params_2))
#
#     return simulation_params

#To generate plots, you need to change the directories based on your local directory structure!

def simulate_wm(NMDA_E_scaler=1,
                NMDA_opp_tuned_I_scaler=1,
                NMDA_tuned_I_scaler=1,
                AMPA_scaler_E=1,
                AMPA_opp_tuned_I_scaler=1,
                AMPA_tuned_I_scaler=1,
                ext_AMPA_scaler_I_near=1,
                ext_AMPA_scaler_I_opp=1,
                ext_AMPA_scaler_E=1,
                GABA_scaler_E=1,
                GABA_scaler_I=1,
                v_firing_threshold_tuned_inhib=-50.0 * b2.mV,
                v_firing_threshold_opp_tuned_inhib=-50.0 * b2.mV,
                v_firing_threshold_excit=-50.0 * b2.mV,
                stimuli_strength=0.175 * b2.namp  # 0.175 * b2.namp
                ):
    start_scope()  # new

    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = v_firing_threshold_excit #-50.4 * b2.mV  # spike condition
    v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
    t_abs_refract_excit = 2.0 * b2.ms  # absolute refractory period

    # specify the inhibitory interneurons:
    Cm_inhib = 0.2 * b2.nF
    G_leak_inhib = 20.0 * b2.nS
    E_leak_inhib = -70.0 * b2.mV
    v_reset_inhib = -60.0 * b2.mV
    t_abs_refract_inhib = 1.0 * b2.ms

    # specify the AMPA synapses
    E_AMPA = 0.0 * b2.mV
    tau_AMPA = 2.0 * b2.ms

    # specify the GABA synapses
    E_GABA = -70.0 * b2.mV
    tau_GABA = 10.0 * b2.ms

    # specify the NMDA synapses
    E_NMDA = 0.0 * b2.mV
    tau_NMDA_s = 100.0 * b2.ms
    tau_NMDA_x = 2.0 * b2.ms
    alpha_NMDA = 0.5 * b2.kHz

    # projections from the external population
    G_extern2inhib_near = 2.38 * ext_AMPA_scaler_I_near * b2.nS #2.38, 5.8 * b2.nS, 1.8
    G_extern2inhib_opp = 2.38 * ext_AMPA_scaler_I_opp * b2.nS #2.38, 5.8 * b2.nS, 1.8
    G_extern2excit = 3.1 * ext_AMPA_scaler_E * b2.nS #3.1, 5.915 * b2.nS, 5

    # GABA-mediated projectsions from the inhibitory populations
    G_inhib2inhib = 1.024 * GABA_scaler_I * weight_scaling_factor * b2.nS #0.7413/1.25, 2.2, 1.024 #maybe I can remove the weight scaling factor since it is not needed here
    G_inhib2excit = 1.336 * GABA_scaler_E * weight_scaling_factor * b2.nS #0.9163/1.25, 3.9, 1.336

    # NMDA-mediated projections from the excitatory population
    G_excit2excit = 0.274 * NMDA_E_scaler * weight_scaling_factor * b2.nS #0.42, 1.05, 0.274 #0.274
    G_excit2tuned_inhib = 0.212 * NMDA_tuned_I_scaler * weight_scaling_factor * b2.nS #0.49/1.25, 0.95, 0.242 0.212
    G_excit2opp_tuned_inhib = 0.212 * NMDA_opp_tuned_I_scaler * weight_scaling_factor * b2.nS #0.49/1.25, 0.95, 0.242 0.212

    # recurrent AMPA
    G_excit2excitA = 0.251 * AMPA_scaler_E * weight_scaling_factor * b2.nS #0.251 0.1 # 0.07
    GEEA = G_excit2excitA / G_extern2excit

    G_excit2tuned_inhibA = 0.192 * AMPA_tuned_I_scaler * weight_scaling_factor * b2.nS #0.192/1.25, 0.36, 0.192 #0.192 #G_excit2inhibA
    GEIA = G_excit2tuned_inhibA / G_extern2inhib_near #G_excit2inhibA

    G_excit2opp_tuned_inhibA = 0.192 * AMPA_opp_tuned_I_scaler * weight_scaling_factor * b2.nS #0.192/1.25, 0.36, 0.192 #0.192
    GEoppIA = G_excit2opp_tuned_inhibA / G_extern2inhib_opp

    t_stimulus_end = t_stimulus_start + t_stimuli_duration
    t_distractor_end = t_distractor_start + t_stimuli_duration

    # compute the stimulus index
    stim_center_idx = int(round(N_excitatory / 360. * stimulus_center_deg))
    stim_width_idx = int(round(N_excitatory / 360. * stimuli_width_deg / 2))
    stim_target_idx = [idx % N_excitatory
                       for idx in
                       range(stim_center_idx - stim_width_idx, stim_center_idx + stim_width_idx + 1)]

    # compute the distractor index
    distr_center_idx = int(round(N_excitatory / 360. * distractor_center_deg))
    distr_target_idx = [idx % N_excitatory
                        for idx in
                        range(distr_center_idx - stim_width_idx, distr_center_idx + stim_width_idx + 1)]

    # precompute/specify the weight profile for/in the recurrent EXCITATORY population
    tmp_excit2excit = math.sqrt(2. * math.pi) * sigma_weight_profile_E2E * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_E2E) / 360.
    Jneg_excit2excit = (1. - Jpos_excit2excit * tmp_excit2excit) / (1. - tmp_excit2excit)
    presyn_excit2excit_weight_kernel = \
        [(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) *
          math.exp(-.5 * ((360. * min(j, N_excitatory - j) / N_excitatory) ** 2) / sigma_weight_profile_E2E ** 2))
         for j in range(N_excitatory)]

    # remove E-E autapses
    presyn_excit2excit_weight_kernel[0] = 0

    fft_presyn_excit2excit_weight_kernel = rfft(presyn_excit2excit_weight_kernel)

    # precompute the weight profile for the TUNED INHIBITORY recurrent population
    tmp_tuned_inhib2excit = math.sqrt(2. * math.pi) * sigma_weight_profile_I2E * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_I2E) / 360.
    Jneg_tuned_inhib2excit = (1. - Jpos_inhib2excit * tmp_tuned_inhib2excit) / (1. - tmp_tuned_inhib2excit)

    # precompute the weight profile for the OPPOSITELY TUNED recurrent population
    tmp_opp_tuned_inhib2excit = math.sqrt(2. * math.pi) * sigma_weight_profile_I2E * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_I2E) / 360.
    Jneg_opp_tuned_inhib2excit = (1. - Jpos_inhib2excit * tmp_opp_tuned_inhib2excit) / (1. - tmp_opp_tuned_inhib2excit)

    # precompute the weight profile for the Excitatory to INHIBITORY recurrent populations
    tmp_excit2inhib = math.sqrt(2. * math.pi) * sigma_weight_profile_E2I * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_E2I) / 360.
    Jneg_excit2inhib = (1. - Jpos_excit2inhib * tmp_excit2inhib) / (1. - tmp_excit2inhib)
    presyn_excit2inhib_weight_kernel = \
        [(Jneg_excit2inhib + (Jpos_excit2inhib - Jneg_excit2inhib) *
          math.exp(-.5 * ((360. * min(j, N_excitatory - j) / N_excitatory) ** 2) / sigma_weight_profile_E2I ** 2))
         for j in range(N_excitatory)]

    fft_presyn_excit2inhib_weight_kernel = rfft(presyn_excit2inhib_weight_kernel)

    # precompute the weight profile for the tuned inhibitory to tuned inhibitory (within pops) recurrent populations
    tmp_within_tuned_inhib2tuned_inhib = math.sqrt(2. * math.pi) * sigma_weight_profile_within_I2I * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_within_I2I) / 360.
    Jneg_within_tuned_inhib2tuned_inhib = (1. - Jpos_within_inhib2inhib * tmp_within_tuned_inhib2tuned_inhib) / (
                1. - tmp_within_tuned_inhib2tuned_inhib)

    # precompute the weight profile for the oppositely tuned inhibitory to oppositely tuned inhibitory (within pops) recurrent populations
    tmp_within_opp_tuned_inhib2opp_tuned_inhib = math.sqrt(2. * math.pi) * sigma_weight_profile_within_I2I * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_within_I2I) / 360.
    Jneg_within_opp_tuned_inhib2opp_tuned_inhib = (
                                                              1. - Jpos_within_inhib2inhib * tmp_within_opp_tuned_inhib2opp_tuned_inhib) / (
                                                              1. - tmp_within_opp_tuned_inhib2opp_tuned_inhib)

    # precompute the weight profile for the tuned Inhibitory to opp tuned INHIBITORY (between pops) recurrent populations
    tmp_between_tuned_inhib2opp_tuned_inhib = math.sqrt(2. * math.pi) * sigma_weight_profile_between_I2I * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_between_I2I) / 360.
    Jneg_between_tuned_inhib2opp_tuned_inhib = (
                                                           1. - Jpos_between_inhib2inhib * tmp_between_tuned_inhib2opp_tuned_inhib) / (
                                                           1. - tmp_between_tuned_inhib2opp_tuned_inhib)

    # precompute the weight profile for the opp tuned Inhibitory to tuned INHIBITORY (between pops) recurrent populations
    tmp_between_opp_tuned_inhib2tuned_inhib = math.sqrt(2. * math.pi) * sigma_weight_profile_between_I2I * erf(
        180. / math.sqrt(2.) / sigma_weight_profile_between_I2I) / 360.
    Jneg_between_opp_tuned_inhib2tuned_inhib = (
                                                           1. - Jpos_between_inhib2inhib * tmp_between_opp_tuned_inhib2tuned_inhib) / (
                                                           1. - tmp_between_opp_tuned_inhib2tuned_inhib)

    # define the tuned inhibitory population
    tuned_inhib_lif_dynamics = """
        s_NMDA_total : 1 # the post synaptic sum of s. compare with s_NMDA_presyn
        active : 1  # new parameter
        I_AMPA = G_extern2inhib_near * s_AMPA * (v-E_AMPA): amp
        I_NMDA = G_excit2tuned_inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*1e3*v/volt)/3.57): amp
        dv/dt = active * (
        - G_leak_inhib * (v-E_leak_inhib)
        - I_AMPA
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - I_NMDA
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    eqs_gaba = '''
    w:1
    '''

    # Tuned inhib_pop
    tuned_inhib_pop = NeuronGroup(
        N_inhibitory_per_pop, model=tuned_inhib_lif_dynamics,
        threshold="v>v_firing_threshold_tuned_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    tuned_inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_tuned_inhib / b2.mV,
                                             size=N_inhibitory_per_pop) * b2.mV

    tuned_inhib_pop.active = 1  # can comment out this line while parameter search is running

    # set the connections: extern2inhib
    input_ext2tuned_inhib = PoissonInput(target=tuned_inhib_pop, target_var="s_AMPA",
                                         N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # define the oppositely tuned inhibitory population
    opp_tuned_inhib_lif_dynamics = """
        s_NMDA_total : 1 # the post synaptic sum of s. compare with s_NMDA_presyn
        active : 1  # new parameter
        I_AMPA = G_extern2inhib_opp * s_AMPA * (v-E_AMPA): amp
        I_NMDA = G_excit2opp_tuned_inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*1e3*v/volt)/3.57): amp
        dv/dt = active * (
        - G_leak_inhib * (v-E_leak_inhib)
        - I_AMPA
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - I_NMDA
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    # Oppositely-tuned inhib_pop
    opp_tuned_inhib_pop = NeuronGroup(
        N_inhibitory_per_pop, model=opp_tuned_inhib_lif_dynamics,
        threshold="v>v_firing_threshold_opp_tuned_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib, #t_abs_refract_opp_inhib
        method="rk2")
    # initialize with random voltages:
    opp_tuned_inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_opp_tuned_inhib / b2.mV,
                                                 size=N_inhibitory_per_pop) * b2.mV

    opp_tuned_inhib_pop.active = 1  # can comment out this line while parameter search is running

    # set the connections: extern2inhib #DO I NEED noise input on the inhibitory populations?
    input_ext2opp_tuned_inhib = PoissonInput(target=opp_tuned_inhib_pop, target_var="s_AMPA",
                                             N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population: # defined s_GABA : 1 (shared) BELOW
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1 # the post synaptic sum of s. compare with s_NMDA_presyn      
        I_AMPA = G_extern2excit * s_AMPA * (v - E_AMPA): amp
        I_NMDA = G_excit2excit * s_NMDA_total * (v - E_NMDA) / (1.0 + 1.0 * exp(-0.062 * 1e3 * v / volt) / 3.57): amp
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - I_AMPA
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - I_NMDA
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")

    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV #-65 * b2.mV

    excit_pop.I_stim = 0. * b2.namp

    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # Create synapses and set weights
    eqs_pre = '''
        s_GABA_post += w
        '''

    # define spacing between inhibitory neurons on the ring
    spacing = N_excitatory / N_inhibitory_per_pop

    # set the connections: STRUCTURED tuned inhibitory to excitatory
    syn_tuned_inhib2excit = Synapses(tuned_inhib_pop, excit_pop, eqs_gaba, on_pre=eqs_pre)
    syn_tuned_inhib2excit.connect(p=1.0)
    syn_tuned_inhib2excit.w[
        'abs((spacing * i)-j)<N_excitatory/2'] = '(Jneg_tuned_inhib2excit + (Jpos_inhib2excit - Jneg_tuned_inhib2excit) * exp(-.5 * ((360. * abs((spacing * i)-j) / N_excitatory)) ** 2 / sigma_weight_profile_I2E ** 2))'  # 'i * 0.001'
    syn_tuned_inhib2excit.w[
        'abs((spacing * i)-j)>=N_excitatory/2'] = '(Jneg_tuned_inhib2excit + (Jpos_inhib2excit - Jneg_tuned_inhib2excit) * exp(-.5 * ((360. * (N_excitatory - abs((spacing * i)-j)) / N_excitatory)) ** 2 / sigma_weight_profile_I2E ** 2))'

    # set the connections: STRUCTURED oppositely-tuned inhibitory to excitatory
    syn_opp_tuned_inhib2excit = Synapses(opp_tuned_inhib_pop, excit_pop, eqs_gaba, on_pre=eqs_pre)
    syn_opp_tuned_inhib2excit.connect(p=1.0)
    syn_opp_tuned_inhib2excit.w[
        'abs((spacing * i)-j)<N_excitatory/2'] = '(Jneg_opp_tuned_inhib2excit + (Jpos_inhib2excit - Jneg_opp_tuned_inhib2excit) * exp(-.5 * ((360. * abs(((spacing * i)-j) / N_excitatory) - 180.)) ** 2 / sigma_weight_profile_I2E ** 2))'
    syn_opp_tuned_inhib2excit.w[
        'abs((spacing * i)-j)>=N_excitatory/2'] = '(Jneg_opp_tuned_inhib2excit + (Jpos_inhib2excit - Jneg_opp_tuned_inhib2excit) * exp(-.5 * (((360. * (N_excitatory - abs((spacing * i)-j)) / N_excitatory) - 180)) ** 2 / sigma_weight_profile_I2E ** 2))'

    # set the connections: STRUCTURED tuned inhibitory to tuned inhibitory
    syn_tuned_inhib2tuned_inhib = Synapses(tuned_inhib_pop, tuned_inhib_pop, eqs_gaba, on_pre=eqs_pre,
                                           delay=0.0 * b2.ms)
    syn_tuned_inhib2tuned_inhib.connect(condition="i!=j", p=1.0)
    syn_tuned_inhib2tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))<N_excitatory/2'] = '(Jneg_within_tuned_inhib2tuned_inhib + (Jpos_within_inhib2inhib - Jneg_within_tuned_inhib2tuned_inhib) * exp(-.5 * ((360. * abs((spacing * i) - (spacing * j)) / N_excitatory)) ** 2 / sigma_weight_profile_within_I2I ** 2))'
    syn_tuned_inhib2tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))>=N_excitatory/2'] = '(Jneg_within_tuned_inhib2tuned_inhib + (Jpos_within_inhib2inhib - Jneg_within_tuned_inhib2tuned_inhib) * exp(-.5 * ((360. * (N_excitatory - abs((spacing * i) - (spacing * j))) / N_excitatory)) ** 2 / sigma_weight_profile_within_I2I ** 2))'

    # set the connections: STRUCTURED oppositely-tuned inhibitory to oppositely-tuned inhibitory
    syn_opp_tuned_inhib2opp_tuned_inhib = Synapses(opp_tuned_inhib_pop, opp_tuned_inhib_pop, eqs_gaba, on_pre=eqs_pre,
                                                   delay=0.0 * b2.ms)
    syn_opp_tuned_inhib2opp_tuned_inhib.connect(condition="i!=j", p=1.0)
    syn_opp_tuned_inhib2opp_tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))<N_excitatory/2'] = '(Jneg_within_opp_tuned_inhib2opp_tuned_inhib + (Jpos_within_inhib2inhib - Jneg_within_opp_tuned_inhib2opp_tuned_inhib) * exp(-.5 * ((360. * abs((spacing * i) - (spacing * j)) / N_excitatory) - 180.) ** 2 / sigma_weight_profile_within_I2I ** 2))'  # not sure this is correct: 'abs((8 * i) - (8 * j))<N_excitatory/2' - why not swap with N_inhibitory - well because conceptually the rings for the two inhibtiry populations are expanded to match the ring for the excitatory population (but of course they are sparser) so using N_excitatory is warranted!!!!
    syn_opp_tuned_inhib2opp_tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))>=N_excitatory/2'] = '(Jneg_within_opp_tuned_inhib2opp_tuned_inhib + (Jpos_within_inhib2inhib - Jneg_within_opp_tuned_inhib2opp_tuned_inhib) * exp(-.5 * ((360. * (N_excitatory - abs((spacing * i) - (spacing * j))) / N_excitatory) - 180.) ** 2 / sigma_weight_profile_within_I2I ** 2))'  # above I understood the condition statement but why use (8 * i) - (8 * j), though? - maybe for the same reason - we are expanding the inhibitory population rings to the size of the excitatory population ring... so... both i and j have to be scaled by 8? - but why do we expand the ring - I am not a 100% sure about this

    # set the connections: STRUCTURED tuned inhibitory to oppositely-tuned inhibitory
    syn_tuned_inhib2opp_tuned_inhib = Synapses(tuned_inhib_pop, opp_tuned_inhib_pop, eqs_gaba, on_pre=eqs_pre,
                                               delay=0.0 * b2.ms)
    syn_tuned_inhib2opp_tuned_inhib.connect(p=1.0)
    syn_tuned_inhib2opp_tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))<N_excitatory/2'] = '(Jneg_between_tuned_inhib2opp_tuned_inhib + (Jpos_between_inhib2inhib - Jneg_between_tuned_inhib2opp_tuned_inhib) * exp(-.5 * (((360. * abs((spacing * i) - (spacing * j)) / N_excitatory))) ** 2 / sigma_weight_profile_between_I2I ** 2))'
    syn_tuned_inhib2opp_tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))>=N_excitatory/2'] = '(Jneg_between_tuned_inhib2opp_tuned_inhib + (Jpos_between_inhib2inhib - Jneg_between_tuned_inhib2opp_tuned_inhib) * exp(-.5 * (((360. * (N_excitatory - abs((spacing * i) - (spacing * j))) / N_excitatory))) ** 2 / sigma_weight_profile_between_I2I ** 2))'

    # set the connections: STRUCTURED oppositely-tuned inhibitory to tuned inhibitory
    syn_opp_tuned_inhib2tuned_inhib = Synapses(opp_tuned_inhib_pop, tuned_inhib_pop, eqs_gaba, on_pre=eqs_pre,
                                               delay=0.0 * b2.ms)
    syn_opp_tuned_inhib2tuned_inhib.connect(p=1.0)
    syn_opp_tuned_inhib2tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))<N_excitatory/2'] = '(Jneg_between_opp_tuned_inhib2tuned_inhib + (Jpos_between_inhib2inhib - Jneg_between_opp_tuned_inhib2tuned_inhib) * exp(-.5 * ((360. * abs((spacing * i) - (spacing * j)) / N_excitatory) - 180.) ** 2 / sigma_weight_profile_between_I2I ** 2))'
    syn_opp_tuned_inhib2tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))>=N_excitatory/2'] = '(Jneg_between_opp_tuned_inhib2tuned_inhib + (Jpos_between_inhib2inhib - Jneg_between_opp_tuned_inhib2tuned_inhib) * exp(-.5 * ((360. * (N_excitatory - abs((spacing * i) - (spacing * j))) / N_excitatory) - 180.) ** 2 / sigma_weight_profile_between_I2I ** 2))'

    # equation for presynaptic AMPA-mediated spike
    eqs_AMPA_pre = '''
     s_AMPA_post += w
     '''

    eqs_AMPA = '''
     w : 1
     '''

    syn_AMPA_excit2tuned_inhib = Synapses(excit_pop, tuned_inhib_pop, model=eqs_AMPA, on_pre=eqs_AMPA_pre)
    syn_AMPA_excit2tuned_inhib.connect(p=1.0)
    syn_AMPA_excit2tuned_inhib.w[
        'abs(i-(spacing*j))<N_excitatory/2'] = 'GEIA * (Jneg_excit2inhib + (Jpos_excit2inhib - Jneg_excit2inhib) * exp(-.5 * (360. * abs(i-(spacing*j)) / N_excitatory) ** 2 / sigma_weight_profile_E2I ** 2))'
    syn_AMPA_excit2tuned_inhib.w[
        'abs(i-(spacing*j))>=N_excitatory/2'] = 'GEIA * (Jneg_excit2inhib + (Jpos_excit2inhib - Jneg_excit2inhib) * exp(-.5 * (360. * (N_excitatory - abs(i-(spacing*j))) / N_excitatory) ** 2 / sigma_weight_profile_E2I ** 2))'

    syn_AMPA_excit2opp_tuned_inhib = Synapses(excit_pop, opp_tuned_inhib_pop, model=eqs_AMPA,
                                              on_pre=eqs_AMPA_pre)  # Synapses(excit_pop, opp_tuned_inhib_pop, 's_ampa', weight=lambda i, j: wrec_i * (Jm_ei + (Jp_ei - Jm_ei) * exp(-0.5 * (360. * min(abs(i - 4 * j), NE - abs(i - 4 * j)) / NE) ** 2 / sigma_ei ** 2)))
    syn_AMPA_excit2opp_tuned_inhib.connect(p=1.0)
    syn_AMPA_excit2opp_tuned_inhib.w[
        'abs(i-(spacing*j))<N_excitatory/2'] = 'GEoppIA * (Jneg_excit2inhib + (Jpos_excit2inhib - Jneg_excit2inhib) * exp(-.5 * (360. * abs(i-(spacing*j)) / N_excitatory) ** 2 / sigma_weight_profile_E2I ** 2))'
    syn_AMPA_excit2opp_tuned_inhib.w[
        'abs(i-(spacing*j))>=N_excitatory/2'] = 'GEoppIA * (Jneg_excit2inhib + (Jpos_excit2inhib - Jneg_excit2inhib) * exp(-.5 * (360. * (N_excitatory - abs(i-(spacing*j))) / N_excitatory) ** 2 / sigma_weight_profile_E2I ** 2))'

    syn_AMPA_excit2excit = Synapses(excit_pop, excit_pop, model=eqs_AMPA, on_pre=eqs_AMPA_pre)
    syn_AMPA_excit2excit.connect(condition="i!=j", p=1.0)
    syn_AMPA_excit2excit.w[
        'abs(i-j)<N_excitatory/2'] = 'GEEA * (Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * abs(i-j) / N_excitatory) ** 2 / sigma_weight_profile_E2E ** 2))'
    syn_AMPA_excit2excit.w[
        'abs(i-j)>=N_excitatory/2'] = 'GEEA * (Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * (N_excitatory - abs(i-j)) / N_excitatory) ** 2 / sigma_weight_profile_E2E ** 2))'

    # set the connections: STRUCTURED recurrent excitatory to excitatory and (tuned and opp-tuned) inhibitory - seems to be faster than the Synapses approach
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_E2E_total = numpy.multiply(fft_presyn_excit2excit_weight_kernel, fft_s_NMDA)
        fft_s_NMDA_E2I_total = numpy.multiply(fft_presyn_excit2inhib_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_E2E_total)
        s_NMDA_EI_tot = irfft(fft_s_NMDA_E2I_total)
        excit_pop.s_NMDA_total_ = s_NMDA_tot

        # set the connections: STRUCTURED NMDA-mediated excitatory to tuned inhibitory
        tuned_inhib_pop.s_NMDA_total_ = s_NMDA_EI_tot[0: -1: int(spacing)]

        # set the connections: STRUCTURED NMDA-mediated excitatory to oppositely-tuned inhibitory
        opp_tuned_inhib_pop.s_NMDA_total = s_NMDA_EI_tot[0: -1: int(spacing)]

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        if t >= t_stimulus_start and t < t_stimulus_end:
            excit_pop.I_stim[stim_target_idx] = stimuli_strength
        else:
            excit_pop.I_stim = 0. * b2.namp
        # add distractor
        if t >= t_distractor_start and t < t_distractor_end:
            excit_pop.I_stim[distr_target_idx] = stimuli_strength

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = \
            [int(math.ceil(k))
             for k in numpy.linspace(0, N - 1, nr_monitored + 2)][1:-1]
        spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
        return spike_monitor, idx_monitored_neurons

    # collect data of a subset of neurons: #can remove some parts of the below while parameter search is running
    spike_monitor_excit, idx_monitored_neurons_excit = \
        get_monitors(excit_pop, monitored_subset_size_excit, N_excitatory)

    spike_monitor_near_target_inhib, idx_monitored_neurons_near_target_inhib = \
        get_monitors(tuned_inhib_pop, 256, 256) # 256 neurons total

    spike_monitor_opp_target_inhib, idx_monitored_neurons_opp_target_inhib = \
        get_monitors(opp_tuned_inhib_pop, 256, 256) # 256 neurons total

    s_AMPA_excit_monitor = StateMonitor(excit_pop, 's_AMPA', record=True)
    s_NMDA_excit_monitor = StateMonitor(excit_pop, 's_NMDA', record=True)

    neuron_0_degress_index = 0  # for example, the nth neuron (indexing starts at 0)
    excit_neuron_180_degrees_index = 1023
    inhib_neuron_180_degrees_index = 127

    M_excit = StateMonitor(excit_pop, ('s_AMPA', 's_NMDA', 'I_AMPA', 'I_NMDA'), record=True)
    M_tuned_inhib = StateMonitor(tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=True)
    M_opp_tuned_inhib = StateMonitor(opp_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=True)

    M_excit_neuron_0_degress_index = StateMonitor(excit_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    M_excit_neuron_180_degress_index = StateMonitor(excit_pop, ('I_AMPA', 'I_NMDA'), record=excit_neuron_180_degrees_index)

    M_tuned_inhib_neuron_0_degress_index = StateMonitor(tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    M_tuned_inhib_neuron_180_degress_index = StateMonitor(tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=inhib_neuron_180_degrees_index)

    M_opp_tuned_inhib_neuron_0_degress_index = StateMonitor(opp_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    M_opp_tuned_inhib_neuron_180_degress_index = StateMonitor(opp_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=inhib_neuron_180_degrees_index)

    b2.run(sim_time)

    spike_indices_excit = spike_monitor_excit.i[:]  # This is a NumPy array with neuron indices
    spike_times_excit = spike_monitor_excit.t[:]

    spike_indices_near_target_inhib = spike_monitor_near_target_inhib.i[:]  # This is a NumPy array with neuron indices
    spike_times_near_target_inhib = spike_monitor_near_target_inhib.t[:]

    spike_indices_opp_target_inhib = spike_monitor_opp_target_inhib.i[:]  # This is a NumPy array with neuron indices
    spike_times_opp_target_inhib = spike_monitor_opp_target_inhib.t[:]

    return spike_indices_excit, spike_times_excit, s_AMPA_excit_monitor, s_NMDA_excit_monitor, spike_indices_near_target_inhib, spike_times_near_target_inhib, spike_indices_opp_target_inhib, spike_times_opp_target_inhib, M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index #added spike_indices_near_target_inhib, spike_times_near_target_inhib, spike_indices_opp_target_inhib, spike_times_opp_target_inhib

def getting_started(): #simulation_params, simulation_id
    # Unpack simulation_params
    # NMDA_E_scaler, v_firing_threshold_tuned_inhib, NMDA_tuned_I_scaler, AMPA_excit_AND_inhib_scaler, v_firing_threshold_opp_tuned_inhib, NMDA_opp_tuned_I_scaler = simulation_params

    b2.defaultclock.dt = 0.1 * b2.ms
    spike_indices_excit, spike_times_excit, s_AMPA_excit_monitor, s_NMDA_excit_monitor, spike_indices_near_target_inhib, spike_times_near_target_inhib, spike_indices_opp_target_inhib, spike_times_opp_target_inhib, M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index = \
                                                         simulate_wm(NMDA_E_scaler=1.1245, #0.8032 #0.865 #0.8333 #1.1245, #0.76, #1.1 #0.785 #0.75 #0.9425, 0.74, #0.54, 0.76 #new model: low 0.85 1.1245 0.865
                                                                     NMDA_opp_tuned_I_scaler=1, #1.1 #0.785, 0.74 #0.54; 0.76; 0.9425; 0.756
                                                                     NMDA_tuned_I_scaler=1, #1.05 #1.1; 0.785 #0.9425, 0.74, #0.54, 0.76
                                                                     AMPA_scaler_E=0.9, #0.9 #1.13 #1.015 #new model: low 1.00 0.9
                                                                     AMPA_opp_tuned_I_scaler=0.9, #1.00 0.96 0.9
                                                                     AMPA_tuned_I_scaler=0.9, #1.00 0.9
                                                                     ext_AMPA_scaler_I_near=1,
                                                                     ext_AMPA_scaler_I_opp=1,
                                                                     ext_AMPA_scaler_E=1.00,
                                                                     GABA_scaler_E=1,
                                                                     GABA_scaler_I=1,
                                                                     v_firing_threshold_tuned_inhib=-50.90 * b2.mV, #51.25 #51.90 -49.55
                                                                     v_firing_threshold_opp_tuned_inhib=-50.00 * b2.mV, #51.90
                                                                     v_firing_threshold_excit=-50.00 * b2.mV, #50.30
                                                                     stimuli_strength=0.2 * b2.namp  # 0.175 * b2.namp #0.25 0.2
                                                                     )

    # Calculate the Local Field Potential

    LFP = numpy.mean(s_AMPA_excit_monitor.s_AMPA, axis=0)
    #newer
    LFP = LFP - numpy.mean(LFP)
    #newer
    # 1. Compute the FFT of the LFP
    # Assuming a time step of 1ms for the simulation. Please adjust if different.
    time_step = b2.defaultclock.dt # in seconds
    frequencies = numpy.fft.fftfreq(len(LFP), d=time_step)
    LFP_fft = numpy.fft.fft(LFP)

    # 2. Compute the power spectral density and consider only positive frequencies
    positive_freq_mask = frequencies > 0
    frequencies = frequencies[positive_freq_mask]
    power_spectral_density = numpy.abs(LFP_fft[positive_freq_mask]) ** 2

    # Optional: Normalize power values (uncomment if needed)
    #power_spectral_density /= numpy.max(power_spectral_density)

    # 3. Plot the power spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, power_spectral_density, '-o', markersize=2)
    plt.xlim([0, 100])
    plt.xticks(numpy.arange(0, 101, 25))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectrum of Local Field Potential')
    plt.grid(True)
    plt.show()

    filename = f'/Users/nd23721/Documents/spike_monitor_data_simulation_0.npz'
    numpy.savez(filename,
                indices=spike_indices_excit,
                times=spike_times_excit,
                )

    filename = f'/Users/nd23721/Documents/spike_monitor_data_simulation_near_targetting.npz'
    numpy.savez(filename,
                indices_near_target=spike_indices_near_target_inhib,
                times_near_target=spike_times_near_target_inhib,
                )

    filename = f'/Users/nd23721/Documents/spike_monitor_data_simulation_opp_targetting.npz'
    numpy.savez(filename,
                indices_opp_target=spike_indices_opp_target_inhib,
                times_opp_target=spike_times_opp_target_inhib,
                )

    #simulation_id=simulation_id, simulation_params=simulation_params

    # add the code that reduces the data here
    return M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index

if __name__ == "__main__":
    M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index = getting_started()

# 0 deg. E cell: I_AMPA and I_NMDA over Time

time = M_excit.t / b2.ms
I_AMPA = M_excit.I_AMPA
I_NMDA = M_excit.I_NMDA
plt.figure(figsize=(12, 6))
plt.plot(time, M_excit.I_AMPA[(0), :] / b2.nA, label='I_AMPA', color='#ff7f0e') #M_excit.I_AMPA[0, :] numpy.mean(M_excit.I_AMPA, axis=0) / b2.nA
plt.plot(time, M_excit.I_NMDA[(0), :] / b2.nA, label='I_NMDA', color='#1f77b4') # numpy.mean(M_excit.I_NMDA, axis=0)
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Current (nA)', fontsize=20)
plt.title('0 deg. E cell: I_AMPA and I_NMDA over Time', fontsize=20)
# Adjusting tick marks
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick mark label size
plt.legend(loc='best', fontsize=18)
plt.show()

# 180 deg. E cell: I_AMPA and I_NMDA over Time

time = M_excit.t / b2.ms
I_AMPA = M_excit.I_AMPA
I_NMDA = M_excit.I_NMDA
plt.figure(figsize=(12, 6))
plt.plot(time, M_excit.I_AMPA[(1023), :] / b2.nA, label='I_AMPA', color='#ff7f0e') #M_excit.I_AMPA[0, :] numpy.mean(M_excit.I_AMPA, axis=0) / b2.nA
plt.plot(time, M_excit.I_NMDA[(1023), :] / b2.nA, label='I_NMDA', color='#1f77b4') # numpy.mean(M_excit.I_NMDA, axis=0)
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Current (nA)', fontsize=20)
plt.title('180 deg. E cell: I_AMPA and I_NMDA over Time', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick mark label size
plt.legend(loc='best', fontsize=18)
plt.show()

# 0 deg. PV_near cell: I_AMPA and I_NMDA over Time

time = M_tuned_inhib.t / b2.ms
I_AMPA = M_tuned_inhib.I_AMPA
I_NMDA = M_tuned_inhib.I_NMDA
plt.figure(figsize=(12, 6))
plt.plot(time, M_tuned_inhib.I_AMPA[(0), :] / b2.nA, label='I_AMPA', color='#ff7f0e') # numpy.mean(M_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_tuned_inhib.I_NMDA[(0), :] / b2.nA, label='I_NMDA', color='#1f77b4') #M_tuned_inhib.I_NMDA[0, :] numpy.mean(M_tuned_inhib.I_NMDA, axis=0)
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Current (nA)', fontsize=20)
plt.title('0 deg. PV_near cell: I_AMPA and I_NMDA over Time', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick mark label size
plt.legend(loc='best', fontsize=18)
plt.show()

# 180 deg. PV_near cell: I_AMPA and I_NMDA over Time

time = M_tuned_inhib.t / b2.ms
I_AMPA = M_tuned_inhib.I_AMPA
I_NMDA = M_tuned_inhib.I_NMDA
plt.figure(figsize=(12, 6))
plt.plot(time, M_tuned_inhib.I_AMPA[(127), :] / b2.nA, label='I_AMPA', color='#ff7f0e') # numpy.mean(M_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_tuned_inhib.I_NMDA[(127), :] / b2.nA, label='I_NMDA', color='#1f77b4') #M_tuned_inhib.I_NMDA[0, :] numpy.mean(M_tuned_inhib.I_NMDA, axis=0)
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Current (nA)', fontsize=20)
plt.title('180 deg. PV_near cell: I_AMPA and I_NMDA over Time', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick mark label size
plt.legend(loc='best', fontsize=18)
plt.show()

# 0 deg. PV_opp cell: I_AMPA and I_NMDA over Time

time = M_opp_tuned_inhib.t / b2.ms
I_AMPA = M_opp_tuned_inhib.I_AMPA
I_NMDA = M_opp_tuned_inhib.I_NMDA
plt.figure(figsize=(12, 6))
plt.plot(time, M_opp_tuned_inhib.I_AMPA[(0), :] / b2.nA, label='I_AMPA', color='#ff7f0e') #M_opp_tuned_inhib.I_AMPA[0, :] numpy.mean(M_opp_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_opp_tuned_inhib.I_NMDA[(0), :] / b2.nA, label='I_NMDA', color='#1f77b4') # numpy.mean(M_opp_tuned_inhib.I_NMDA, axis=0)
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Current (nA)', fontsize=20)
plt.title('0 deg. PV_opp cell: I_AMPA and I_NMDA over Time', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick mark label size
plt.legend(loc='best', fontsize=18)
plt.show()

# 180 deg. PV_opp cell: I_AMPA and I_NMDA over Time

time = M_opp_tuned_inhib.t / b2.ms
I_AMPA = M_opp_tuned_inhib.I_AMPA
I_NMDA = M_opp_tuned_inhib.I_NMDA
plt.figure(figsize=(12, 6))
plt.plot(time, M_opp_tuned_inhib.I_AMPA[(127), :] / b2.nA, label='I_AMPA', color='#ff7f0e') #M_opp_tuned_inhib.I_AMPA[0, :] numpy.mean(M_opp_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_opp_tuned_inhib.I_NMDA[(127), :] / b2.nA, label='I_NMDA', color='#1f77b4') # numpy.mean(M_opp_tuned_inhib.I_NMDA, axis=0)
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Current (nA)', fontsize=20)
plt.title('180 deg. PV_opp cell: I_AMPA and I_NMDA over Time', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick mark label size
plt.legend(loc='best', fontsize=18)
plt.show()

#EXCITATORY population rastergram

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.colors import Normalize

#NEW
# Before creating your figure, adjust the default DPI setting for plots
plt.rcParams['figure.dpi'] = 300  # or any other DPI value you prefer
#NEW

# Load the data
data = np.load("/Users/nd23721/Documents/spike_monitor_data_simulation_0.npz")

# Extract the arrays
neuron_indices = data['indices']
spike_times = data['times']

# Total number of neurons
num_neurons = neuron_indices.max() + 1  # assuming indices start from 0

# Convert neuron indices to degrees
neuron_degrees = neuron_indices * (360.0 / num_neurons)

total_time_length = max(spike_times) - min(spike_times)

adjustment_val = 5 - total_time_length

# Calculate histogram
bins_time = np.linspace(min(spike_times), max(spike_times) + adjustment_val, 100)
bins_neuron = np.linspace(0, 360, 100)
counts, _, _ = np.histogram2d(spike_times, neuron_degrees, bins=[bins_time, bins_neuron])

# Convert counts to spikes per second per neuron
time_bin_duration = (bins_time[1] - bins_time[0])
neurons_per_bin = num_neurons / 100
counts_per_second_per_neuron = counts / (time_bin_duration * neurons_per_bin)

# Reorder the counts using advanced indexing
num_bins = counts_per_second_per_neuron.shape[1]
new_order = list(range(3 * num_bins // 4, num_bins)) + list(range(3 * num_bins // 4))
counts_per_second_per_neuron = counts_per_second_per_neuron[:, new_order]

# Define Hz as 1.0 for our calculations
Hz = 1.0

# Plot a heatmap on the bottom axes with power-law color scale
# Let's assume the maximum firing rate is 100 Hz
vmax = 50 * Hz  # If you want the value in Hz

# Clip counts to match the maximum firing rate
clipped_counts = np.minimum(counts_per_second_per_neuron / Hz, vmax / Hz)

# Create a figure and axes with shared x-axis
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)

img = ax.imshow(clipped_counts.T, extent=[min(spike_times), max(spike_times), 270, 630],
                aspect='auto', origin='lower', cmap='viridis', norm=Normalize(vmin=0, vmax=vmax))

#ax.set_xlabel('Time (s)', fontsize=18)
ax.set_ylabel('Neuron index', fontsize=18)

# Set fixed locations for y-axis
ax.yaxis.set_major_locator(FixedLocator([270, 360, 450, 540, 630]))

# Create custom formatter
def format_fn(tick_val, tick_pos):
    if tick_val == 630:
        return '270°'
    elif tick_val == 540:
        return '180°'
    elif tick_val == 450:
        return '90°'
    elif tick_val == 360:
        return '0°'
    else:
        return '270°'

# Set custom formatter
ax.yaxis.set_major_formatter(FuncFormatter(format_fn))

# Change the color of specific y-tick labels
for label in ax.get_yticklabels():
    if label.get_text() == '0°':
        label.set_color('red')
    elif label.get_text() == '180°':
        label.set_color('blue')

# Extend x-axis limit to 5000 ms
ax.set_xlim([min(spike_times), 5])

# Add a tickmark at 5000 ms
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))  # Time is already in seconds
ax.tick_params(axis='both', which='major', labelsize=16)

cbar = fig.colorbar(img, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Spikes per second', size=16)

# Add red and blue strips to indicate the duration of target and distractor stimulus
plt.axvspan(0.4, 0.65, ymin=1.03, ymax=1.06, color='red', transform=ax.get_xaxis_transform(), clip_on=False)
plt.axvspan(2.7, 2.95, ymin=1.03, ymax=1.06, color='blue', transform=ax.get_xaxis_transform(), clip_on=False)

plt.tight_layout()
plt.savefig('/Users/nd23721/Documents/marm-mac/excit_net_3_marm_high_D1R.png', dpi=300)
plt.show()

######################################################################################

#NEAR FEATURE-SELECTIVE PV rastergram

#NEW
# Before creating your figure, adjust the default DPI setting for plots
plt.rcParams['figure.dpi'] = 300  # or any other DPI value you prefer
#NEW

# Load the data
data = np.load("/Users/nd23721/Documents/spike_monitor_data_simulation_near_targetting.npz")

# Extract the arrays
neuron_indices_near_target = data['indices_near_target']
spike_times_near_target = data['times_near_target']

# Total number of neurons
num_neurons = neuron_indices_near_target.max() + 1  # assuming indices start from 0

# Convert neuron indices to degrees
neuron_degrees = neuron_indices_near_target * (360.0 / num_neurons)

total_time_length = max(spike_times_near_target) - min(spike_times_near_target)

adjustment_val = 5 - total_time_length

# Calculate histogram
bins_time = np.linspace(min(spike_times_near_target), max(spike_times_near_target) + adjustment_val, 100)
bins_neuron = np.linspace(0, 360, 100)
counts, _, _ = np.histogram2d(spike_times_near_target, neuron_degrees, bins=[bins_time, bins_neuron])

# Convert counts to spikes per second per neuron
time_bin_duration = (bins_time[1] - bins_time[0])
neurons_per_bin = num_neurons / 100
counts_per_second_per_neuron = counts / (time_bin_duration * neurons_per_bin)

# Reorder the counts using advanced indexing
num_bins = counts_per_second_per_neuron.shape[1]
new_order = list(range(3 * num_bins // 4, num_bins)) + list(range(3 * num_bins // 4))
counts_per_second_per_neuron = counts_per_second_per_neuron[:, new_order]

# Define Hz as 1.0 for our calculations
Hz = 1.0

# Plot a heatmap on the bottom axes with power-law color scale
# Let's assume the maximum firing rate is 100 Hz
vmax = 50 * Hz  # If you want the value in Hz

# Clip counts to match the maximum firing rate
clipped_counts = np.minimum(counts_per_second_per_neuron / Hz, vmax / Hz)

# Create a figure and axes with shared x-axis
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)

img = ax.imshow(clipped_counts.T, extent=[min(spike_times_near_target), max(spike_times_near_target), 270, 630],
                aspect='auto', origin='lower', cmap='viridis', norm=Normalize(vmin=0, vmax=vmax))

#ax.set_xlabel('Time (s)', fontsize=18)
ax.set_ylabel('Neuron index', fontsize=18)

# Set fixed locations for y-axis
ax.yaxis.set_major_locator(FixedLocator([270, 360, 450, 540, 630]))

# Create custom formatter
def format_fn(tick_val, tick_pos):
    if tick_val == 630:
        return '270°'
    elif tick_val == 540:
        return '180°'
    elif tick_val == 450:
        return '90°'
    elif tick_val == 360:
        return '0°'
    else:
        return '270°'

# Set custom formatter
ax.yaxis.set_major_formatter(FuncFormatter(format_fn))

# Change the color of specific y-tick labels
for label in ax.get_yticklabels():
    if label.get_text() == '0°':
        label.set_color('red')
    elif label.get_text() == '180°':
        label.set_color('blue')

# Extend x-axis limit to 5000 ms
ax.set_xlim([min(spike_times_near_target), 5])

# Add a tickmark at 5000 ms
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))  # Time is already in seconds
ax.tick_params(axis='both', which='major', labelsize=16)

cbar = fig.colorbar(img, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Spikes per second', size=16)

# Add red and blue strips to indicate the duration of target and distractor stimulus
plt.axvspan(0.4, 0.65, ymin=1.03, ymax=1.06, color='red', transform=ax.get_xaxis_transform(), clip_on=False)
plt.axvspan(2.7, 2.95, ymin=1.03, ymax=1.06, color='blue', transform=ax.get_xaxis_transform(), clip_on=False)

plt.tight_layout()
plt.savefig('/Users/nd23721/Documents/marm-mac/near_net_3_marm_high_D1R.png', dpi=300)
plt.show()

######################################################################################

#OPPOSITE FEATURE-SELECTIVE PV rastergram

#NEW
# Before creating your figure, adjust the default DPI setting for plots
plt.rcParams['figure.dpi'] = 300  # or any other DPI value you prefer
#NEW

# Load the data
data = np.load("/Users/nd23721/Documents/spike_monitor_data_simulation_opp_targetting.npz")

# Extract the arrays
neuron_indices_opp_target = data['indices_opp_target']
spike_times_opp_target = data['times_opp_target']

# Total number of neurons
num_neurons = neuron_indices_opp_target.max() + 1  # assuming indices start from 0

# Convert neuron indices to degrees
neuron_degrees = neuron_indices_opp_target * (360.0 / num_neurons)

# Calculate histogram
bins_time = np.linspace(0, sim_time.item(), 100)
bins_neuron = np.linspace(0, 360, 100)
counts, _, _ = np.histogram2d(spike_times_opp_target, neuron_degrees, bins=[bins_time, bins_neuron])

# Convert counts to spikes per second per neuron
time_bin_duration = (bins_time[1] - bins_time[0])
neurons_per_bin = num_neurons / 100
counts_per_second_per_neuron = counts / (time_bin_duration * neurons_per_bin)

# Reorder the counts using advanced indexing
num_bins = counts_per_second_per_neuron.shape[1]
new_order = list(range(3 * num_bins // 4, num_bins)) + list(range(3 * num_bins // 4))
counts_per_second_per_neuron = counts_per_second_per_neuron[:, new_order]

# Define Hz as 1.0 for our calculations
Hz = 1.0

# Plot a heatmap on the bottom axes with power-law color scale
# Let's assume the maximum firing rate is 100 Hz
vmax = 50 * Hz  # If you want the value in Hz

# Clip counts to match the maximum firing rate
clipped_counts = np.minimum(counts_per_second_per_neuron / Hz, vmax / Hz)

# Create a figure and axes with shared x-axis
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)

img = ax.imshow(clipped_counts.T, extent=[0, sim_time.item(), 270, 630],
                aspect='auto', origin='lower', cmap='viridis', norm=Normalize(vmin=0, vmax=vmax))

#ax.set_xlabel('Time (s)', fontsize=18)
ax.set_ylabel('Neuron index', fontsize=18)

# Set fixed locations for y-axis
ax.yaxis.set_major_locator(FixedLocator([270, 360, 450, 540, 630]))

# Create custom formatter
def format_fn(tick_val, tick_pos):
    if tick_val == 630:
        return '270°'
    elif tick_val == 540:
        return '180°'
    elif tick_val == 450:
        return '90°'
    elif tick_val == 360:
        return '0°'
    else:
        return '270°'

# Set custom formatter
ax.yaxis.set_major_formatter(FuncFormatter(format_fn))

# Change the color of specific y-tick labels
for label in ax.get_yticklabels():
    if label.get_text() == '0°':
        label.set_color('red')
    elif label.get_text() == '180°':
        label.set_color('blue')

# Extend x-axis limit to 5000 ms
ax.set_xlim([0, sim_time.item()])

# Add a tickmark at 5000 ms
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))  # Time is already in seconds
ax.tick_params(axis='both', which='major', labelsize=16)

cbar = fig.colorbar(img, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Spikes per second', size=16)

# Add red and blue strips to indicate the duration of target and distractor stimulus
plt.axvspan(0.4, 0.65, ymin=1.03, ymax=1.06, color='red', transform=ax.get_xaxis_transform(), clip_on=False)
plt.axvspan(2.7, 2.95, ymin=1.03, ymax=1.06, color='blue', transform=ax.get_xaxis_transform(), clip_on=False)

plt.tight_layout()
plt.savefig('/Users/nd23721/Documents/marm-mac/opp_net_3_marm_high_D1R.png', dpi=300)
plt.show()
