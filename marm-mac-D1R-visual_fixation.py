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
# new
import subprocess

os.environ['SDKROOT'] = subprocess.check_output('xcrun --show-sdk-path', shell=True).decode().strip()
# new
from neurodynex3.tools import plot_tools
import matplotlib.pyplot as plt

from brian2 import StateMonitor

# new
# Specify the compiler
os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'

# Set compiler flags for Brian2
from brian2 import prefs

prefs['codegen.cpp.extra_compile_args'] = ['-std=c++11', '-stdlib=libc++']
prefs['codegen.cpp.extra_link_args'] = ['-stdlib=libc++']
# new

N_excitatory = 2048
N_inhibitory_per_pop = 512  # 256 #512 #256 #256

# new
N_fix_excitatory = 256  # 128 #160 #170 #256 #512 #256 #512 #256
N_fix_inhibitory_per_pop = 32  # 32 #20 #21 #32 #64 # 32 #64 #128 #64
# new

weight_scaling_factor = 1
stimuli_width_deg = 30  # 30 #20 #60
sim_time = 3400. * b2.ms  # 4000.

# new
t_fixation_start = 400 * b2.ms  # 1000

angles = [0, 45, 90, 135, 180, 225, 270, 315]

num_stimuli = 175  # Number of stimuli
distractor_center_deg = numpy.random.choice(angles, size=175)  # 55
t_distractor_start = 600 * b2.ms  # 2100 2700
t_distractor_duration = 16 * b2.ms  # 250

# Compute indices corresponding to the random angles and their time windows
stim_width_idx = int(round(N_excitatory / 360. * stimuli_width_deg / 2))
stimuli = []

for i_angle, angle in enumerate(distractor_center_deg):
    center_idx = int(round(N_excitatory / 360. * angle))
    indices = [idx % N_excitatory for idx in range(center_idx - stim_width_idx, center_idx + stim_width_idx + 1)]
    start_time = t_distractor_start + i_angle * t_distractor_duration
    end_time = start_time + t_distractor_duration
    stimuli.append((start_time, end_time, indices))
# new

N_extern_poisson = 1000
poisson_firing_rate = 1.80 * b2.Hz  # 1.6 * b2.Hz #1.6 1.80

sigma_weight_profile_E2E = 14.4  # 25 #20 #18 #25 #14.4
Jpos_excit2excit = 2.1  # 2.1 #1.63#2.1 #7.1 # 1.62 #2 #1.62 #2

sigma_weight_profile_I2E = 14.4  # 14.4 #4.5 #20 #32 #4.5 #14.4
Jpos_inhib2excit = 1.6  # 1.6 #1.5 #1.22 #2.2 #1.9 #1.63 #1.5 #1.4

sigma_weight_profile_E2I = 14.4  # 43.2 #25 #20 #32 #25 #14.4
Jpos_excit2inhib = 1.6  # 1.5 #1.22 #2.2 #1.62 #1.63 #0.5

sigma_weight_profile_between_I2I = 14.4  # 4.5 #4.5 #14.4
Jpos_between_inhib2inhib = 2.1  # 2.1#1.63 #1.63 #1.5 #1.9

sigma_weight_profile_within_I2I = 14.4  # 4.5 #4.5 #14.4
Jpos_within_inhib2inhib = 2.1  # 2.1 #1 #1.63 #1.5 #1.9

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

# To generate plots, you need to change the directories based on your local directory structure!

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
                v_firing_threshold_fix_tuned_inhib=-50.90 * b2.mV,
                v_firing_threshold_opp_tuned_inhib=-50.0 * b2.mV,
                v_firing_threshold_excit=-50.0 * b2.mV,
                stimuli_strength=0.175 * b2.namp,  # 0.175 * b2.namp
                fix_NMDA_E_scaler=1,
                fix_NMDA_tuned_I_scaler=1,
                fix_NMDA_opp_tuned_I_scaler=1,
                fix_AMPA_scaler_E=1,
                fix_AMPA_tuned_I_scaler=1,
                fix_AMPA_opp_tuned_I_scaler=1,
                fix_near_GABA_scaler_fix_E=1,  # 1.2
                fix_opp_GABA_scaler_E=1,
                ):
    start_scope()  # new

    # specify the excitatory pyramidal cells:
    Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
    G_leak_excit = 25.0 * b2.nS  # leak conductance
    E_leak_excit = -70.0 * b2.mV  # reversal potential
    v_firing_threshold_excit = v_firing_threshold_excit  # -50.4 * b2.mV  # spike condition
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
    G_extern2inhib_near = 2.38 * ext_AMPA_scaler_I_near * b2.nS  # 2.38, 5.8 * b2.nS, 1.8
    G_extern2inhib_opp = 2.38 * ext_AMPA_scaler_I_opp * b2.nS  # 2.38, 5.8 * b2.nS, 1.8
    G_extern2excit = 3.1 * ext_AMPA_scaler_E * b2.nS  # 3.1, 5.915 * b2.nS, 5

    # GABA-mediated projectsions from the inhibitory populations
    G_inhib2inhib = 1.024 * GABA_scaler_I * weight_scaling_factor * b2.nS  # 0.7413/1.25, 2.2, 1.024 #maybe I can remove the weight scaling factor since it is not needed here
    G_inhib2excit = 1.336 * GABA_scaler_E * weight_scaling_factor * b2.nS  # 0.9163/1.25, 3.9, 1.336

    # new
    G_fix_near_inhib2fix_excit = 1.336 * fix_near_GABA_scaler_fix_E * weight_scaling_factor * b2.nS  # 0.9163/1.25, 3.9, 1.336
    G_fix_opp_inhib2excit = 1.336 * fix_opp_GABA_scaler_E * weight_scaling_factor * b2.nS  # 0.9163/1.25, 3.9, 1.336
    # new

    # NMDA-mediated projections from the excitatory population
    G_excit2excit = 0.274 * NMDA_E_scaler * weight_scaling_factor * b2.nS  # 0.42, 1.05, 0.274 #0.274
    G_excit2tuned_inhib = 0.212 * NMDA_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.49/1.25, 0.95, 0.242 0.212
    G_excit2opp_tuned_inhib = 0.212 * NMDA_opp_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.49/1.25, 0.95, 0.242 0.212

    # new
    # NMDA-mediated projections from the FIXATION excitatory population
    G_fix_excit2fix_excit = 0.274 * fix_NMDA_E_scaler * weight_scaling_factor * b2.nS  # 0.42, 1.05, 0.274 #0.274
    # G_fix_excit2tuned_inhib = 0.212 * fix_NMDA_tuned_I_scaler * weight_scaling_factor * b2.nS #0.49/1.25, 0.95, 0.242 0.212
    G_fix_excit2fix_tuned_inhib = 0.212 * fix_NMDA_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.49/1.25, 0.95, 0.242 0.212
    G_fix_excit2fix_opp_tuned_inhib = 0.212 * fix_NMDA_opp_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.49/1.25, 0.95, 0.242 0.212
    # new

    # recurrent AMPA
    G_excit2excitA = 0.251 * AMPA_scaler_E * weight_scaling_factor * b2.nS  # 0.251 0.1 # 0.07
    GEEA = G_excit2excitA / G_extern2excit

    G_excit2tuned_inhibA = 0.192 * AMPA_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.192/1.25, 0.36, 0.192 #0.192 #G_excit2inhibA
    GEIA = G_excit2tuned_inhibA / G_extern2inhib_near  # G_excit2inhibA

    G_excit2opp_tuned_inhibA = 0.192 * AMPA_opp_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.192/1.25, 0.36, 0.192 #0.192
    GEoppIA = G_excit2opp_tuned_inhibA / G_extern2inhib_opp

    # new
    # recurrent AMPA FIXATION
    G_fix_excit2fix_excitA = 0.251 * fix_AMPA_scaler_E * weight_scaling_factor * b2.nS  # 0.251 0.1 # 0.07
    fix_GEEA = G_fix_excit2fix_excitA / G_extern2excit

    G_fix_excit2tuned_inhibA = 0.192 * fix_AMPA_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.192/1.25, 0.36, 0.192 #0.192 #G_excit2inhibA
    fix_GEIA = G_fix_excit2tuned_inhibA / G_extern2inhib_near  # G_excit2inhibA

    G_fix_excit2opp_tuned_inhibA = 0.192 * fix_AMPA_opp_tuned_I_scaler * weight_scaling_factor * b2.nS  # 0.192/1.25, 0.36, 0.192 #0.192
    fix_GEoppIA = G_fix_excit2opp_tuned_inhibA / G_extern2inhib_opp

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

    # define the FIXATION tuned inhibitory population # 2 * G_inhib2inhib * s_GABA * (v-E_GABA)
    fix_tuned_inhib_lif_dynamics = """
        s_NMDA_total : 1 # the post synaptic sum of s. compare with s_NMDA_presyn
        active : 1  # new parameter
        I_AMPA = G_extern2inhib_near * s_AMPA * (v-E_AMPA): amp
        I_NMDA = G_fix_excit2fix_tuned_inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*1e3*v/volt)/3.57): amp
        dv/dt = active * (
        - G_leak_inhib * (v-E_leak_inhib)
        - I_AMPA
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - I_NMDA
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    # Tuned inhib_pop
    fix_tuned_inhib_pop = NeuronGroup(
        N_fix_inhibitory_per_pop, model=fix_tuned_inhib_lif_dynamics,
        threshold="v>v_firing_threshold_fix_tuned_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    fix_tuned_inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_fix_tuned_inhib / b2.mV,
                                                 size=N_fix_inhibitory_per_pop) * b2.mV

    fix_tuned_inhib_pop.active = 1  # can comment out this line while parameter search is running

    # set the connections: extern2inhib
    input_ext2fix_tuned_inhib = PoissonInput(target=fix_tuned_inhib_pop, target_var="s_AMPA",
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
        threshold="v>v_firing_threshold_opp_tuned_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,  # t_abs_refract_opp_inhib
        method="rk2")
    # initialize with random voltages:
    opp_tuned_inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_opp_tuned_inhib / b2.mV,
                                                 size=N_inhibitory_per_pop) * b2.mV

    opp_tuned_inhib_pop.active = 0  # can comment out this line while parameter search is running

    # set the connections: extern2inhib #DO I NEED noise input on the inhibitory populations?
    input_ext2opp_tuned_inhib = PoissonInput(target=opp_tuned_inhib_pop, target_var="s_AMPA",
                                             N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # define the FIXATION oppositely tuned inhibitory population #2 * G_inhib2inhib * s_GABA * (v-E_GABA)
    fix_opp_tuned_inhib_lif_dynamics = """
        s_NMDA_total : 1 # the post synaptic sum of s. compare with s_NMDA_presyn
        active : 1  # new parameter
        I_AMPA = G_extern2inhib_opp * s_AMPA * (v-E_AMPA): amp
        I_NMDA = G_fix_excit2fix_opp_tuned_inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*1e3*v/volt)/3.57): amp
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
    fix_opp_tuned_inhib_pop = NeuronGroup(
        N_fix_inhibitory_per_pop, model=fix_opp_tuned_inhib_lif_dynamics,
        threshold="v>v_firing_threshold_opp_tuned_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,  # t_abs_refract_opp_inhib
        method="rk2")
    # initialize with random voltages:
    fix_opp_tuned_inhib_pop.v = numpy.random.uniform(v_reset_inhib / b2.mV, high=v_firing_threshold_opp_tuned_inhib / b2.mV,
                                                     size=N_fix_inhibitory_per_pop) * b2.mV

    fix_opp_tuned_inhib_pop.active = 1  # can comment out this line while parameter search is running

    # set the connections: extern2inhib #DO I NEED noise input on the inhibitory populations?
    input_ext2fix_opp_tuned_inhib = PoissonInput(target=fix_opp_tuned_inhib_pop, target_var="s_AMPA",
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
        - G_fix_opp_inhib2excit * s_fix_opp_GABA_to_E * (v-E_GABA)
        - I_NMDA
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_fix_opp_GABA_to_E/dt = -s_fix_opp_GABA_to_E/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                            refractory=t_abs_refract_excit, method="rk2")

    # initialize with random voltages:
    excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                       size=N_excitatory) * b2.mV  # -65 * b2.mV

    excit_pop.I_stim = 0. * b2.namp

    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the FIXATIOn excitatory population: # defined s_GABA : 1 (shared) BELOW #        - 2 * G_inhib2excit * s_GABA * (v-E_GABA)
    fix_excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1 # the post synaptic sum of s. compare with s_NMDA_presyn
        I_AMPA = G_extern2excit * s_AMPA * (v - E_AMPA): amp
        I_NMDA = G_fix_excit2fix_excit * s_NMDA_total * (v - E_NMDA) / (1.0 + 1.0 * exp(-0.062 * 1e3 * v / volt) / 3.57): amp
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - I_AMPA
        - G_fix_near_inhib2fix_excit * s_fix_near_GABA_to_fix_E * (v-E_GABA)
        - I_NMDA
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_fix_near_GABA_to_fix_E/dt = -s_fix_near_GABA_to_fix_E/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    fix_excit_pop = NeuronGroup(N_fix_excitatory, model=fix_excit_lif_dynamics,
                                threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit; x+=1.0",
                                refractory=t_abs_refract_excit, method="rk2")

    # initialize with random voltages:
    fix_excit_pop.v = numpy.random.uniform(v_reset_excit / b2.mV, high=v_firing_threshold_excit / b2.mV,
                                           size=N_fix_excitatory) * b2.mV  # -65 * b2.mV

    fix_excit_pop.I_stim = 0. * b2.namp

    # set the connections: extern2excit
    input_ext2fix_excit = PoissonInput(target=fix_excit_pop, target_var="s_AMPA",
                                       N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # Create synapses and set weights
    eqs_pre = '''
        s_GABA_post += w
        '''

    # Create synapses and set weights
    eqs_fix_pre = '''
        s_GABA_post += 1.0
        '''

    eqs_fix_opp_to_E_pre = '''
        s_fix_opp_GABA_to_E_post += 1.0
        '''

    eqs_fix_near_to_fix_E_pre = '''
        s_fix_near_GABA_to_fix_E_post += 1.0
        '''

    eqs_fix_opp_to_fix_near_pre = '''
    s_GABA_opp_to_near_post += 1.0
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

    # set the connections: tuned inhibitory (FIXATION) to excitatory (FIXATION)
    syn_FIX_tuned_inhib2FIX_excit = Synapses(fix_tuned_inhib_pop, fix_excit_pop, on_pre=eqs_fix_near_to_fix_E_pre)
    syn_FIX_tuned_inhib2FIX_excit.connect(p=1.0)

    # set the connections: oppositely-tuned inhibitory (FIXATION) to excitatory
    syn_FIX_opp_tuned_inhib2excit = Synapses(fix_opp_tuned_inhib_pop, excit_pop, eqs_gaba, on_pre=eqs_fix_opp_to_E_pre)
    syn_FIX_opp_tuned_inhib2excit.connect(p=1.0)

    # set the connections: STRUCTURED tuned inhibitory to tuned inhibitory
    syn_tuned_inhib2tuned_inhib = Synapses(tuned_inhib_pop, tuned_inhib_pop, eqs_gaba, on_pre=eqs_pre,
                                           delay=0.0 * b2.ms)
    syn_tuned_inhib2tuned_inhib.connect(condition="i!=j", p=1.0)
    syn_tuned_inhib2tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))<N_excitatory/2'] = '(Jneg_within_tuned_inhib2tuned_inhib + (Jpos_within_inhib2inhib - Jneg_within_tuned_inhib2tuned_inhib) * exp(-.5 * ((360. * abs((spacing * i) - (spacing * j)) / N_excitatory)) ** 2 / sigma_weight_profile_within_I2I ** 2))'
    syn_tuned_inhib2tuned_inhib.w[
        'abs((spacing * i) - (spacing * j))>=N_excitatory/2'] = '(Jneg_within_tuned_inhib2tuned_inhib + (Jpos_within_inhib2inhib - Jneg_within_tuned_inhib2tuned_inhib) * exp(-.5 * ((360. * (N_excitatory - abs((spacing * i) - (spacing * j))) / N_excitatory)) ** 2 / sigma_weight_profile_within_I2I ** 2))'

    # set the connections: tuned inhibitory (FIXATION) to tuned inhibitory (FIXATION)
    syn_fix_tuned_inhib2fix_tuned_inhib = Synapses(fix_tuned_inhib_pop, fix_tuned_inhib_pop, eqs_gaba, on_pre=eqs_fix_pre,
                                                   delay=0.0 * b2.ms)
    syn_fix_tuned_inhib2fix_tuned_inhib.connect(condition="i!=j", p=1.0)

    # set the connections: tuned inhibitory (FIXATION) to oppositely-tuned inhibitory (FIXATION)
    syn_fix_tuned_inhib2fix_opp_tuned_inhib = Synapses(fix_tuned_inhib_pop, fix_opp_tuned_inhib_pop, eqs_gaba, on_pre=eqs_fix_pre,
                                               delay=0.0 * b2.ms)
    syn_fix_tuned_inhib2fix_opp_tuned_inhib.connect(p=1.0)

    # # set the connections: oppositely-tuned inhibitory (FIXATION) to tuned inhibitory (FIXATION)
    # syn_fix_opp_tuned_inhib2fix_tuned_inhib = Synapses(fix_opp_tuned_inhib_pop, fix_tuned_inhib_pop, eqs_gaba, on_pre=eqs_fix_opp_to_fix_near_pre,
    #                                            delay=0.0 * b2.ms)
    # syn_fix_opp_tuned_inhib2fix_tuned_inhib.connect(p=1.0)

    # equation for presynaptic AMPA-mediated spike
    eqs_AMPA_pre = '''
     s_AMPA_post += w
     '''

    # equation for presynaptic AMPA-mediated spike for FIXATION population
    eqs_fix_AMPA_pre = '''
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

    syn_AMPA_excit2fix_tuned_inhib = Synapses(excit_pop, fix_tuned_inhib_pop, model=eqs_AMPA, on_pre=eqs_AMPA_pre)
    syn_AMPA_excit2fix_tuned_inhib.connect(p=1.0)
    syn_AMPA_excit2fix_tuned_inhib.w = GEIA

    syn_AMPA_fix_excit2fix_tuned_inhib = Synapses(fix_excit_pop, fix_tuned_inhib_pop, model=eqs_AMPA, on_pre=eqs_fix_AMPA_pre)
    syn_AMPA_fix_excit2fix_tuned_inhib.connect(p=1.0)
    syn_AMPA_fix_excit2fix_tuned_inhib.w = fix_GEIA

    syn_AMPA_fix_excit2fix_opp_tuned_inhib = Synapses(fix_excit_pop, fix_opp_tuned_inhib_pop, model=eqs_AMPA,
                                              on_pre=eqs_fix_AMPA_pre)  # Synapses(excit_pop, opp_tuned_inhib_pop, 's_ampa', weight=lambda i, j: wrec_i * (Jm_ei + (Jp_ei - Jm_ei) * exp(-0.5 * (360. * min(abs(i - 4 * j), NE - abs(i - 4 * j)) / NE) ** 2 / sigma_ei ** 2)))
    syn_AMPA_fix_excit2fix_opp_tuned_inhib.connect(p=1.0)
    syn_AMPA_fix_excit2fix_opp_tuned_inhib.w = fix_GEoppIA

    syn_AMPA_excit2excit = Synapses(excit_pop, excit_pop, model=eqs_AMPA, on_pre=eqs_AMPA_pre)
    syn_AMPA_excit2excit.connect(condition="i!=j", p=1.0)
    syn_AMPA_excit2excit.w[
        'abs(i-j)<N_excitatory/2'] = 'GEEA * (Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * abs(i-j) / N_excitatory) ** 2 / sigma_weight_profile_E2E ** 2))'
    syn_AMPA_excit2excit.w[
        'abs(i-j)>=N_excitatory/2'] = 'GEEA * (Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * (N_excitatory - abs(i-j)) / N_excitatory) ** 2 / sigma_weight_profile_E2E ** 2))'

    syn_AMPA_fix_excit2fix_excit = Synapses(fix_excit_pop, fix_excit_pop, model=eqs_AMPA, on_pre=eqs_fix_AMPA_pre)
    syn_AMPA_fix_excit2fix_excit.connect(condition="i!=j", p=1.0)
    syn_AMPA_fix_excit2fix_excit.w = fix_GEEA

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

        # new
        s_NMDA_tot_fix = numpy.sum(fix_excit_pop.s_NMDA)
        s_NMDA_tot_fix_ex = s_NMDA_tot_fix - fix_excit_pop.s_NMDA #exlude autapses

        # # Compute NMDA totals excluding autapses
        # N = len(fix_excit_pop.s_NMDA)
        # s_NMDA_tot_fix_ex = numpy.zeros(N)  # Create an array to store results
        # for i in range(N):
        #     # Total NMDA contribution to neuron `i` from all others
        #     s_NMDA_tot_fix_ex[i] = numpy.sum(fix_excit_pop.s_NMDA) - fix_excit_pop.s_NMDA[i]

        s_NMDA_tot_cue = numpy.sum(excit_pop.s_NMDA)
        fix_excit_pop.s_NMDA_total_ = s_NMDA_tot_fix_ex
        fix_tuned_inhib_pop.s_NMDA_total_ = s_NMDA_tot_fix + s_NMDA_tot_cue
        fix_opp_tuned_inhib_pop.s_NMDA_total_ = s_NMDA_tot_fix
        # new

    @network_operation(dt=1 * b2.ms)
    def stimulate_network(t):
        # new
        if (t >= 400 * b2.ms and t < 3400 * b2.ms):  # (t >= (t_stimulus_start + (850 * b2.ms)) and t < (t_stimulus_end + (1100 * b2.ms))
            fix_excit_pop.I_stim = stimuli_strength
        else:
            fix_excit_pop.I_stim = 0. * b2.namp
        #
        # if t >= t_fixation_start:
        #     fix_excit_pop.I_stim = stimuli_strength
        # else:
        #     fix_excit_pop.I_stim = 0 * b2.namp

        if t >= t_distractor_start and t < t_distractor_start + num_stimuli * t_distractor_duration:
            elapsed_time = t - t_distractor_start
            stimulus_index = int(elapsed_time / t_distractor_duration)
            if stimulus_index < num_stimuli:
                # Reset the input current before applying the new stimulus
                excit_pop.I_stim = 0. * b2.namp
                indices = stimuli[stimulus_index][2]
                excit_pop.I_stim[indices] = stimuli_strength
        else:
            # Ensure that no stimulus is applied outside the stimulus periods
            excit_pop.I_stim = 0. * b2.namp

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
        get_monitors(tuned_inhib_pop, 256, 256)  # 256 neurons total

    spike_monitor_opp_target_inhib, idx_monitored_neurons_opp_target_inhib = \
        get_monitors(opp_tuned_inhib_pop, 256, 256)  # 256 neurons total

    # new
    spike_monitor_fix_excit, idx_monitored_neurons_fix_excit = \
        get_monitors(fix_excit_pop, N_fix_excitatory, N_fix_excitatory)

    spike_monitor_fix_near_target_inhib, idx_monitored_neurons_fix_near_target_inhib = \
        get_monitors(fix_tuned_inhib_pop, N_fix_inhibitory_per_pop, N_fix_inhibitory_per_pop)  # 256 neurons total

    # spike_monitor_fix_opp_target_inhib, idx_monitored_neurons_fix_opp_target_inhib = \
    #     get_monitors(fix_opp_tuned_inhib_pop, N_fix_inhibitory_per_pop, N_fix_inhibitory_per_pop) # 256 neurons total
    # new

    s_AMPA_excit_monitor = StateMonitor(excit_pop, 's_AMPA', record=True)
    s_NMDA_excit_monitor = StateMonitor(excit_pop, 's_NMDA', record=True)

    neuron_0_degress_index = 0  # for example, the nth neuron (indexing starts at 0)
    excit_neuron_180_degrees_index = 1023
    inhib_neuron_180_degrees_index = 127

    M_excit = StateMonitor(excit_pop, ('s_AMPA', 's_NMDA', 'I_AMPA', 'I_NMDA'), record=True)
    M_tuned_inhib = StateMonitor(tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=True)
    M_opp_tuned_inhib = StateMonitor(opp_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=True)

    # new
    M_fix_excit = StateMonitor(fix_excit_pop, ('s_AMPA', 's_NMDA', 'I_AMPA', 'I_NMDA'), record=True)
    M_fix_tuned_inhib = StateMonitor(fix_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=True)
    # M_fix_opp_tuned_inhib = StateMonitor(fix_opp_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=True)
    # new

    M_excit_neuron_0_degress_index = StateMonitor(excit_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    M_excit_neuron_180_degress_index = StateMonitor(excit_pop, ('I_AMPA', 'I_NMDA'), record=excit_neuron_180_degrees_index)

    M_tuned_inhib_neuron_0_degress_index = StateMonitor(tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    M_tuned_inhib_neuron_180_degress_index = StateMonitor(tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=inhib_neuron_180_degrees_index)

    M_opp_tuned_inhib_neuron_0_degress_index = StateMonitor(opp_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    M_opp_tuned_inhib_neuron_180_degress_index = StateMonitor(opp_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=inhib_neuron_180_degrees_index)

    # new
    # M_fix_excit_neuron_0_degress_index = StateMonitor(fix_excit_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    # M_fix_excit_neuron_180_degress_index = StateMonitor(fix_excit_pop, ('I_AMPA', 'I_NMDA'), record=excit_neuron_180_degrees_index)
    #
    # M_fix_tuned_inhib_neuron_0_degress_index = StateMonitor(fix_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=neuron_0_degress_index)
    # M_fix_tuned_inhib_neuron_180_degress_index = StateMonitor(fix_tuned_inhib_pop, ('I_AMPA', 'I_NMDA'), record=inhib_neuron_180_degrees_index)

    b2.run(sim_time)

    spike_indices_excit = spike_monitor_excit.i[:]  # This is a NumPy array with neuron indices
    spike_times_excit = spike_monitor_excit.t[:]

    spike_indices_near_target_inhib = spike_monitor_near_target_inhib.i[:]  # This is a NumPy array with neuron indices
    spike_times_near_target_inhib = spike_monitor_near_target_inhib.t[:]

    spike_indices_opp_target_inhib = spike_monitor_opp_target_inhib.i[:]  # This is a NumPy array with neuron indices
    spike_times_opp_target_inhib = spike_monitor_opp_target_inhib.t[:]

    # new
    spike_indices_fix_excit = spike_monitor_fix_excit.i[:]  # This is a NumPy array with neuron indices
    spike_times_fix_excit = spike_monitor_fix_excit.t[:]

    spike_indices_fix_tuned_inhib = spike_monitor_fix_near_target_inhib.i[:]  # This is a NumPy array with neuron indices
    spike_times_fix_tuned_inhib = spike_monitor_fix_near_target_inhib.t[:]

    # spike_indices_fix_opp_tuned_inhib = spike_monitor_fix_opp_target_inhib.i[:]  # This is a NumPy array with neuron indices
    # spike_times_fix_opp_tuned_inhib = spike_monitor_fix_opp_target_inhib.t[:]
    # new

    return spike_indices_excit, spike_times_excit, s_AMPA_excit_monitor, s_NMDA_excit_monitor, spike_indices_near_target_inhib, spike_times_near_target_inhib, spike_indices_opp_target_inhib, spike_times_opp_target_inhib, M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index, spike_indices_fix_excit, spike_times_fix_excit, spike_indices_fix_tuned_inhib, spike_times_fix_tuned_inhib  # , spike_indices_fix_opp_tuned_inhib, spike_times_fix_opp_tuned_inhib #added spike_indices_near_target_inhib, spike_times_near_target_inhib, spike_indices_opp_target_inhib, spike_times_opp_target_inhib


def getting_started():  # simulation_params, simulation_id
    # Unpack simulation_params
    # NMDA_E_scaler, v_firing_threshold_tuned_inhib, NMDA_tuned_I_scaler, AMPA_excit_AND_inhib_scaler, v_firing_threshold_opp_tuned_inhib, NMDA_opp_tuned_I_scaler = simulation_params

    b2.defaultclock.dt = 0.1 * b2.ms  # NEED TO CHANGE BACK TO 0.1ms once things are working for the cluster!!!
    spike_indices_excit, spike_times_excit, s_AMPA_excit_monitor, s_NMDA_excit_monitor, spike_indices_near_target_inhib, spike_times_near_target_inhib, spike_indices_opp_target_inhib, spike_times_opp_target_inhib, M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index, spike_indices_fix_excit, spike_times_fix_excit, spike_indices_fix_tuned_inhib, spike_times_fix_tuned_inhib = \
        simulate_wm(NMDA_E_scaler=0.8032,#0.865,
                    # 0.8032 #0.865 #0.8333 #1.1245, #0.76, #1.1 #0.785 #0.75 #0.9425, 0.74, #0.54, 0.76 #new model: low 0.85 1.1245 0.865 #0.775
                    NMDA_opp_tuned_I_scaler=1,  # 1.1 #0.785, 0.74 #0.54; 0.76; 0.9425; 0.756
                    NMDA_tuned_I_scaler=1,  # 1.05 #1.1; 0.785 #0.9425, 0.74, #0.54, 0.76
                    AMPA_scaler_E=1.13,  # 0.9 #1.13 #1.015 #new model: low 1.00 0.9 1.13
                    AMPA_opp_tuned_I_scaler=1.13,  # 1.00 0.96 0.9 1.13
                    AMPA_tuned_I_scaler=1.13,  # 1.00 0.9 1.13
                    ext_AMPA_scaler_I_near=1,
                    ext_AMPA_scaler_I_opp=1,
                    ext_AMPA_scaler_E=1,
                    GABA_scaler_E=1.125, #1.075, #1.1 #1.125
                    GABA_scaler_I=1.0,
                    v_firing_threshold_tuned_inhib=-50.00 * b2.mV,  # 51.25 #51.90 -49.55 50.53
                    v_firing_threshold_fix_tuned_inhib=-50.80 * b2.mV,  # 51.40 51.25 #51.90 -49.55 50.53 #-50.60 -- this last one seems to work quite well 51.40 50.80
                    v_firing_threshold_opp_tuned_inhib=-50.00 * b2.mV,  # 51.90 50.00
                    v_firing_threshold_excit=-50.00 * b2.mV,  # 50.30
                    stimuli_strength=0.2 * b2.namp,  #0.175 * b2.namp #0.25 0.2
                    fix_NMDA_E_scaler=1.1245, #1.1245 #1.1245 #1.2
                    fix_NMDA_tuned_I_scaler=1,
                    fix_NMDA_opp_tuned_I_scaler=1, #0.1
                    fix_AMPA_scaler_E=1.13, #1.2 #1.13 #1.2
                    fix_AMPA_tuned_I_scaler=1.13, #1.2 #1.13 #1.2
                    fix_AMPA_opp_tuned_I_scaler=1.13, #1.8 #1.2 #1.13 #1.2
                    fix_near_GABA_scaler_fix_E=1.275, #125, #1.2 #0.85 #1
                    fix_opp_GABA_scaler_E=0.25 #0.075 #1.25 #0.4
                    )

    # Calculate the Local Field Potential

    LFP = numpy.mean(s_AMPA_excit_monitor.s_AMPA, axis=0)
    # newer
    LFP = LFP - numpy.mean(LFP)
    # newer
    # 1. Compute the FFT of the LFP
    # Assuming a time step of 1ms for the simulation. Please adjust if different.
    time_step = b2.defaultclock.dt  # in seconds
    frequencies = numpy.fft.fftfreq(len(LFP), d=time_step)
    LFP_fft = numpy.fft.fft(LFP)

    # 2. Compute the power spectral density and consider only positive frequencies
    positive_freq_mask = frequencies > 0
    frequencies = frequencies[positive_freq_mask]
    power_spectral_density = numpy.abs(LFP_fft[positive_freq_mask]) ** 2

    # Optional: Normalize power values (uncomment if needed)
    # power_spectral_density /= numpy.max(power_spectral_density)

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

    # new
    filename = f'/Users/nd23721/Documents/spike_monitor_fix_data_simulation_0.npz'
    numpy.savez(filename,
                indices_fix=spike_indices_fix_excit,
                times_fix=spike_times_fix_excit,
                )

    filename = f'/Users/nd23721/Documents/spike_monitor_fix_data_simulation_near_targetting.npz'
    numpy.savez(filename,
                indices_fix_near_target=spike_indices_fix_tuned_inhib,
                times_fix_near_target=spike_times_fix_tuned_inhib,
                )

    # filename = f'/Users/nd23721/Documents/spike_monitor_fix_data_simulation_opp_targetting.npz'
    # numpy.savez(filename,
    #             indices_fix_opp_target=spike_indices_fix_opp_tuned_inhib,
    #             times_fix_opp_target=spike_times_fix_opp_tuned_inhib,
    #             )

    # simulation_id=simulation_id, simulation_params=simulation_params

    # add the code that reduces the data here
    return M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index


if __name__ == "__main__":
    M_excit, M_tuned_inhib, M_opp_tuned_inhib, M_excit_neuron_0_degress_index, M_excit_neuron_180_degress_index, M_tuned_inhib_neuron_0_degress_index, M_tuned_inhib_neuron_180_degress_index, M_opp_tuned_inhib_neuron_0_degress_index, M_opp_tuned_inhib_neuron_180_degress_index = getting_started()

# 0 deg. E cell: I_AMPA and I_NMDA over Time

time = M_excit.t / b2.ms
I_AMPA = M_excit.I_AMPA
I_NMDA = M_excit.I_NMDA
plt.figure(figsize=(12, 6))
plt.plot(time, M_excit.I_AMPA[(0), :] / b2.nA, label='I_AMPA', color='#ff7f0e')  # M_excit.I_AMPA[0, :] numpy.mean(M_excit.I_AMPA, axis=0) / b2.nA
plt.plot(time, M_excit.I_NMDA[(0), :] / b2.nA, label='I_NMDA', color='#1f77b4')  # numpy.mean(M_excit.I_NMDA, axis=0)
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
plt.plot(time, M_excit.I_AMPA[(1023), :] / b2.nA, label='I_AMPA', color='#ff7f0e')  # M_excit.I_AMPA[0, :] numpy.mean(M_excit.I_AMPA, axis=0) / b2.nA
plt.plot(time, M_excit.I_NMDA[(1023), :] / b2.nA, label='I_NMDA', color='#1f77b4')  # numpy.mean(M_excit.I_NMDA, axis=0)
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
plt.plot(time, M_tuned_inhib.I_AMPA[(0), :] / b2.nA, label='I_AMPA', color='#ff7f0e')  # numpy.mean(M_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_tuned_inhib.I_NMDA[(0), :] / b2.nA, label='I_NMDA',
         color='#1f77b4')  # M_tuned_inhib.I_NMDA[0, :] numpy.mean(M_tuned_inhib.I_NMDA, axis=0)
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
plt.plot(time, M_tuned_inhib.I_AMPA[(127), :] / b2.nA, label='I_AMPA', color='#ff7f0e')  # numpy.mean(M_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_tuned_inhib.I_NMDA[(127), :] / b2.nA, label='I_NMDA',
         color='#1f77b4')  # M_tuned_inhib.I_NMDA[0, :] numpy.mean(M_tuned_inhib.I_NMDA, axis=0)
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
plt.plot(time, M_opp_tuned_inhib.I_AMPA[(0), :] / b2.nA, label='I_AMPA',
         color='#ff7f0e')  # M_opp_tuned_inhib.I_AMPA[0, :] numpy.mean(M_opp_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_opp_tuned_inhib.I_NMDA[(0), :] / b2.nA, label='I_NMDA', color='#1f77b4')  # numpy.mean(M_opp_tuned_inhib.I_NMDA, axis=0)
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
plt.plot(time, M_opp_tuned_inhib.I_AMPA[(127), :] / b2.nA, label='I_AMPA',
         color='#ff7f0e')  # M_opp_tuned_inhib.I_AMPA[0, :] numpy.mean(M_opp_tuned_inhib.I_AMPA, axis=0)
plt.plot(time, M_opp_tuned_inhib.I_NMDA[(127), :] / b2.nA, label='I_NMDA', color='#1f77b4')  # numpy.mean(M_opp_tuned_inhib.I_NMDA, axis=0)
plt.xlabel('Time (ms)', fontsize=20)
plt.ylabel('Current (nA)', fontsize=20)
plt.title('180 deg. PV_opp cell: I_AMPA and I_NMDA over Time', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=18)  # Increase tick mark label size
plt.legend(loc='best', fontsize=18)
plt.show()

# EXCITATORY population rastergram

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.colors import Normalize

# NEW
# Before creating your figure, adjust the default DPI setting for plots
plt.rcParams['figure.dpi'] = 300  # or any other DPI value you prefer
# NEW

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

adjustment_val = 3.4 - total_time_length # 5000

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

# ax.set_xlabel('Time (s)', fontsize=18)
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
ax.set_xlim([min(spike_times), 3.4])

# Add a tickmark at 5000 ms
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 3.4]))
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

# EXCITATORY population rastergram

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.colors import Normalize

# NEW
# Before creating your figure, adjust the default DPI setting for plots
plt.rcParams['figure.dpi'] = 300  # or any other DPI value you prefer
# NEW

# FIXATION EXCITATORY POPULATION RASTERGRAM

# Load the data
data = np.load("/Users/nd23721/Documents/spike_monitor_fix_data_simulation_0.npz")

# Extract the arrays
neuron_indices = data['indices_fix']
spike_times = data['times_fix']

# Total number of neurons
num_neurons = neuron_indices.max() + 1  # assuming indices start from 0

total_time_length = max(spike_times) - min(spike_times)

adjustment_val = 3.4 - total_time_length # 5000

# Calculate histogram
bins_time = np.linspace(min(spike_times), max(spike_times) + adjustment_val, 100)
bins_neuron = np.linspace(0, num_neurons, 100)
counts, _, _ = np.histogram2d(spike_times, neuron_indices, bins=[bins_time, bins_neuron])

# Convert counts to spikes per second per neuron
time_bin_duration = bins_time[1] - bins_time[0]
neurons_per_bin = num_neurons / 100  # assuming 100 bins in neurons
counts_per_second_per_neuron = counts / (time_bin_duration * neurons_per_bin)

# Remove reordering since data is no longer circular
# counts_per_second_per_neuron remains unchanged

# Define Hz as 1.0 for our calculations
Hz = 1.0

# Plot a heatmap with linear color scale
# Assume the maximum firing rate is 50 Hz
vmax = 50 * Hz

# Clip counts to match the maximum firing rate
clipped_counts = np.minimum(counts_per_second_per_neuron / Hz, vmax / Hz)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)

img = ax.imshow(clipped_counts.T, extent=[min(spike_times), max(spike_times) + adjustment_val, 0, num_neurons],
                aspect='auto', origin='lower', cmap='viridis', norm=Normalize(vmin=0, vmax=vmax))

ax.set_xlabel('Time (s)', fontsize=18)
ax.set_ylabel('Neuron index', fontsize=18)

# Set fixed locations for y-axis ticks
y_ticks = [0, num_neurons / 4, num_neurons / 2, 3 * num_neurons / 4, num_neurons]
ax.yaxis.set_major_locator(FixedLocator(y_ticks))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))

# Adjust x-axis limits
ax.set_xlim([min(spike_times), 3.4]) # 5000

# Set x-axis ticks at every second 5000 ms
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 3.4]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))

ax.tick_params(axis='both', which='major', labelsize=16)

# Add colorbar
cbar = fig.colorbar(img, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Spikes per second', size=16)

# Add red and blue strips to indicate stimulus durations
plt.axvspan(0.4, 0.65, ymin=1.03, ymax=1.06, color='red', transform=ax.get_xaxis_transform(), clip_on=False)
plt.axvspan(2.7, 2.95, ymin=1.03, ymax=1.06, color='blue', transform=ax.get_xaxis_transform(), clip_on=False)

plt.tight_layout()
plt.savefig('/Users/nd23721/Documents/marm-mac/fix_excit_net_3_marm_high_D1R.png', dpi=300)
plt.show()

######################################################################################

# NEAR FEATURE-SELECTIVE PV rastergram

# NEW
# Before creating your figure, adjust the default DPI setting for plots
plt.rcParams['figure.dpi'] = 300  # or any other DPI value you prefer
# NEW

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

adjustment_val = 3.4 - total_time_length # 5000

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

# ax.set_xlabel('Time (s)', fontsize=18)
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
ax.set_xlim([min(spike_times_near_target), 3.4])

# Add a tickmark at 5000 ms
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 3.4]))
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

# FIXATION NEAR FEATURE-SELECTIVE PV rastergram

# Set the default DPI setting for plots
plt.rcParams['figure.dpi'] = 300

# Load the data
data = np.load("/Users/nd23721/Documents/spike_monitor_fix_data_simulation_near_targetting.npz")

# Extract the arrays
neuron_indices_near_target = data['indices_fix_near_target']
spike_times_near_target = data['times_fix_near_target']

# Total number of neurons
num_neurons = neuron_indices_near_target.max() + 1  # assuming indices start from 0

total_time_length = max(spike_times_near_target) - min(spike_times_near_target)

adjustment_val = 3.4 - total_time_length # 5000

# Calculate histogram
bins_time = np.linspace(min(spike_times_near_target), max(spike_times_near_target) + adjustment_val, 100)
bins_neuron = np.linspace(0, num_neurons, 100)
counts, _, _ = np.histogram2d(spike_times_near_target, neuron_indices_near_target, bins=[bins_time, bins_neuron])

# Convert counts to spikes per second per neuron
time_bin_duration = bins_time[1] - bins_time[0]
neurons_per_bin = num_neurons / 100  # assuming 100 bins in neurons
counts_per_second_per_neuron = counts / (time_bin_duration * neurons_per_bin)

# Remove reordering since data is no longer circular
# counts_per_second_per_neuron remains unchanged

# Define Hz as 1.0 for our calculations
Hz = 1.0

# Plot a heatmap with linear color scale
# Assume the maximum firing rate is 50 Hz
vmax = 50 * Hz

# Clip counts to match the maximum firing rate
clipped_counts = np.minimum(counts_per_second_per_neuron / Hz, vmax / Hz)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 3), dpi=300)

img = ax.imshow(clipped_counts.T, extent=[min(spike_times_near_target), max(spike_times_near_target) + adjustment_val, 0, num_neurons],
                aspect='auto', origin='lower', cmap='viridis', norm=Normalize(vmin=0, vmax=vmax))

ax.set_ylabel('Neuron index', fontsize=18)

# Set fixed locations for y-axis ticks
y_ticks = [0, num_neurons / 4, num_neurons / 2, 3 * num_neurons / 4, num_neurons]
ax.yaxis.set_major_locator(FixedLocator(y_ticks))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))

# Optionally, highlight specific neurons by changing the color of their tick labels
for label in ax.get_yticklabels():
    if int(label.get_text()) == 0:
        label.set_color('red')
    elif int(label.get_text()) == int(num_neurons / 2):
        label.set_color('blue')

# Extend x-axis limit to 5 seconds
ax.set_xlim([min(spike_times_near_target), 3.4]) # 5000

# Set x-axis ticks at every second 5000 ms
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 3.4]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))
ax.tick_params(axis='both', which='major', labelsize=16)

cbar = fig.colorbar(img, ax=ax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Spikes per second', size=16)

# Add red and blue strips to indicate the duration of target and distractor stimulus
plt.axvspan(0.4, 0.65, ymin=1.03, ymax=1.06, color='red', transform=ax.get_xaxis_transform(), clip_on=False)
plt.axvspan(2.7, 2.95, ymin=1.03, ymax=1.06, color='blue', transform=ax.get_xaxis_transform(), clip_on=False)

plt.tight_layout()
plt.savefig('/Users/nd23721/Documents/marm-mac/fix_near_net_3_marm_high_D1R.png', dpi=300)
plt.show()

# #FIXATION OPPOSITE FEATURE-SELECTIVE PV rastergram
#
# #NEW
# # Before creating your figure, adjust the default DPI setting for plots
# plt.rcParams['figure.dpi'] = 300  # or any other DPI value you prefer
# #NEW
#
# # Load the data
# data = np.load("/Users/nd23721/Documents/spike_monitor_fix_data_simulation_opp_targetting.npz")
#
# # Extract the arrays
# neuron_indices_opp_target = data['indices_fix_opp_target']
# spike_times_opp_target = data['times_fix_opp_target']
#
# # Total number of neurons
# num_neurons = neuron_indices_opp_target.max() + 1  # assuming indices start from 0
#
# total_time_length = max(spike_times_opp_target) - min(spike_times_opp_target)
#
# adjustment_val = 5 - total_time_length
#
# # Calculate histogram
# bins_time = np.linspace(min(spike_times_opp_target), max(spike_times_opp_target) + adjustment_val, 100)
# bins_neuron = np.linspace(0, num_neurons, 100)
# counts, _, _ = np.histogram2d(spike_times_opp_target, neuron_indices_opp_target, bins=[bins_time, bins_neuron])
#
# # Convert counts to spikes per second per neuron
# time_bin_duration = bins_time[1] - bins_time[0]
# neurons_per_bin = num_neurons / 100  # assuming 100 bins in neurons
# counts_per_second_per_neuron = counts / (time_bin_duration * neurons_per_bin)
#
# # Remove reordering since data is no longer circular
# # counts_per_second_per_neuron remains unchanged
#
# # Define Hz as 1.0 for our calculations
# Hz = 1.0
#
# # Plot a heatmap with linear color scale
# # Assume the maximum firing rate is 50 Hz
# vmax = 50 * Hz
#
# # Clip counts to match the maximum firing rate
# clipped_counts = np.minimum(counts_per_second_per_neuron / Hz, vmax / Hz)
#
# # Create a figure and axes
# fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
#
# img = ax.imshow(clipped_counts.T, extent=[min(spike_times_opp_target), max(spike_times_opp_target) + adjustment_val, 0, num_neurons],
#                 aspect='auto', origin='lower', cmap='viridis', norm=Normalize(vmin=0, vmax=vmax))
#
# ax.set_ylabel('Neuron index', fontsize=18)
#
# # Set fixed locations for y-axis ticks
# y_ticks = [0, num_neurons / 4, num_neurons / 2, 3 * num_neurons / 4, num_neurons]
# ax.yaxis.set_major_locator(FixedLocator(y_ticks))
# ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
#
# # Optionally, highlight specific neurons by changing the color of their tick labels
# for label in ax.get_yticklabels():
#     if int(label.get_text()) == 0:
#         label.set_color('red')
#     elif int(label.get_text()) == int(num_neurons / 2):
#         label.set_color('blue')
#
# # Extend x-axis limit to 5 seconds
# ax.set_xlim([min(spike_times_opp_target), 5])
#
# # Set x-axis ticks at every second
# ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 4, 5]))
# ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))
# ax.tick_params(axis='both', which='major', labelsize=16)
#
# cbar = fig.colorbar(img, ax=ax)
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label('Spikes per second', size=16)
#
# # Add red and blue strips to indicate the duration of target and distractor stimulus
# plt.axvspan(0.4, 0.65, ymin=1.03, ymax=1.06, color='red', transform=ax.get_xaxis_transform(), clip_on=False)
# plt.axvspan(2.7, 2.95, ymin=1.03, ymax=1.06, color='blue', transform=ax.get_xaxis_transform(), clip_on=False)
#
# plt.tight_layout()
# plt.savefig('/Users/nd23721/Documents/marm-mac/fix_opp_net_3_marm_high_D1R.png', dpi=300)
# plt.show()

## TEST

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FixedLocator, FuncFormatter
# from matplotlib.colors import Normalize
# from matplotlib.gridspec import GridSpec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# # Set the default DPI for high-resolution plots
# plt.rcParams['figure.dpi'] = 300
#
# # Load data for the first plot (EXCITATORY population rastergram)
# data1 = np.load("/Users/nd23721/Documents/spike_monitor_data_simulation_0.npz")
# neuron_indices1 = data1['indices']
# spike_times1 = data1['times']
# num_neurons1 = neuron_indices1.max() + 1  # Total number of neurons in the first population
#
# # Convert neuron indices to degrees for the first plot
# neuron_degrees1 = neuron_indices1 * (360.0 / num_neurons1)
#
# # Load data for the second plot (FIXATION EXCITATORY population rastergram)
# data2 = np.load("/Users/nd23721/Documents/spike_monitor_fix_data_simulation_0.npz")
# neuron_indices2 = data2['indices_fix']
# spike_times2 = data2['times_fix']
# num_neurons2 = neuron_indices2.max() + 1  # Total number of neurons in the second population
#
# # Define common time bins for both plots
# time_start = 0
# time_end = 3.4 #5000
# bins_time = np.linspace(time_start, time_end, 100)
#
# # Histogram calculations for the first plot
# bins_neuron1 = np.linspace(0, 360, 100)
# counts1, _, _ = np.histogram2d(spike_times1, neuron_degrees1, bins=[bins_time, bins_neuron1])
# time_bin_duration = bins_time[1] - bins_time[0]
# neurons_per_bin1 = num_neurons1 / 100
# counts_per_second_per_neuron1 = counts1 / (time_bin_duration * neurons_per_bin1)
#
# # Reorder the counts for the first plot using advanced indexing
# num_bins1 = counts_per_second_per_neuron1.shape[1]
# new_order = list(range(3 * num_bins1 // 4, num_bins1)) + list(range(3 * num_bins1 // 4))
# counts_per_second_per_neuron1 = counts_per_second_per_neuron1[:, new_order]
#
# # Histogram calculations for the second plot
# bins_neuron2 = np.linspace(0, num_neurons2, 100)
# counts2, _, _ = np.histogram2d(spike_times2, neuron_indices2, bins=[bins_time, bins_neuron2])
# neurons_per_bin2 = num_neurons2 / 100
# counts_per_second_per_neuron2 = counts2 / (time_bin_duration * neurons_per_bin2)
#
# # Create the figure and GridSpec layout
# fig = plt.figure(figsize=(10, 6), dpi=300)
# gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[num_neurons2, num_neurons1])
#
# # Create subplots for the two rastergrams
# ax_top = fig.add_subplot(gs[0, 0])
# ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
#
# # Define the maximum firing rate for color scaling
# vmax = 50.0  # Assuming Hz = 1.0
#
# # Plot the second rastergram (top plot)
# clipped_counts2 = np.minimum(counts_per_second_per_neuron2, vmax)
# img2 = ax_top.imshow(
#     clipped_counts2.T,
#     extent=[time_start, time_end, 0, num_neurons2],
#     aspect='auto',
#     origin='lower',
#     cmap='viridis',
#     norm=Normalize(vmin=0, vmax=vmax)
# )
#
# # Remove y-axis labels and ticks for the second plot
# ax_top.set_ylabel('')  # Remove the y-axis label
# ax_top.set_yticks([])  # Remove the y-axis ticks
#
# # ax_top.set_ylabel('Neuron index', fontsize=18)
# # y_ticks = [0, num_neurons2 / 4, num_neurons2 / 2, 3 * num_neurons2 / 4, num_neurons2]
# # ax_top.yaxis.set_major_locator(FixedLocator(y_ticks))
# # ax_top.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
# ax_top.tick_params(axis='both', which='major', labelsize=16)
# ax_top.set_xlim([time_start, time_end])
# #ax_top.axvspan(0.4, 5, ymin=1.03, ymax=1.06, color='red', transform=ax_top.get_xaxis_transform(), clip_on=False)
# # ax_top.axvspan(2.7, 2.95, ymin=1.03, ymax=1.06, color='blue', transform=ax_top.get_xaxis_transform(), clip_on=False)
# plt.setp(ax_top.get_xticklabels(), visible=False)  # Hide x-axis labels on the top plot
#
# # Plot the first rastergram (bottom plot)
# clipped_counts1 = np.minimum(counts_per_second_per_neuron1, vmax)
# img1 = ax_bottom.imshow(
#     clipped_counts1.T,
#     extent=[time_start, time_end, 270, 630],
#     aspect='auto',
#     origin='lower',
#     cmap='viridis',
#     norm=Normalize(vmin=0, vmax=vmax)
# )
# ax_bottom.set_xlabel('Time (s)', fontsize=18)
# ax_bottom.set_ylabel('Neuron index', fontsize=18)
# ax_bottom.yaxis.set_major_locator(FixedLocator([270, 360, 450, 540, 630]))
#
# def format_fn(tick_val, tick_pos):
#     if tick_val == 630:
#         return '270°'
#     elif tick_val == 540:
#         return '180°'
#     elif tick_val == 450:
#         return '90°'
#     elif tick_val == 360:
#         return '0°'
#     else:
#         return '270°'
#
# ax_bottom.yaxis.set_major_formatter(FuncFormatter(format_fn))
# # for label in ax_bottom.get_yticklabels():
# #     if label.get_text() == '0°':
# #         label.set_color('red')
# #     elif label.get_text() == '180°':
# #         label.set_color('blue')
# ax_bottom.tick_params(axis='both', which='major', labelsize=16)
# ax_bottom.set_xlim([time_start, time_end])
# ax_bottom.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3, 3.4])) #5000
# ax_bottom.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))
# # ax_bottom.axvspan(0.4, 0.65, ymin=1.03, ymax=1.06, color='red', transform=ax_bottom.get_xaxis_transform(), clip_on=False)
# ax_bottom.axvspan(0.6, 3.4, ymin=1.03, ymax=1.06, color='blue', transform=ax_bottom.get_xaxis_transform(), clip_on=False) # 5000
#
# # Adjust the spacing between subplots
# plt.subplots_adjust(hspace=0.05)
#
# # # Add a single colorbar for both plots
# # cax = fig.add_subplot(gs[:, 1])
# # cbar = fig.colorbar(img1, cax=cax)
# # cbar.ax.tick_params(labelsize=12)
# # cbar.set_label('Spikes per second', size=16)
# #
# # # Save and display the combined figure
# # plt.tight_layout()
# # plt.savefig('/Users/nd23721/Documents/marm-mac/combined_plot.png', dpi=300)
# # plt.show()
#
# # Get positions of the axes in figure coordinates
# pos_top = ax_top.get_position()
# pos_bottom = ax_bottom.get_position()
#
# # Compute the vertical offset and height of the axvspan in the bottom plot (in figure coordinates)
# offset_bottom = (1.03 - 1.0) * pos_bottom.height  # Offset from top of bottom axes to start of axvspan
# height_axvspan_bottom = (1.06 - 1.03) * pos_bottom.height  # Height of the axvspan
#
# # Compute the position of the axvspan in the top plot (in figure coordinates)
# axvspan_ymin_top = pos_top.y1 + offset_bottom
# axvspan_ymax_top = axvspan_ymin_top + height_axvspan_bottom
#
# # Convert back to axes fraction units for the top plot
# axes_fraction_ymin_top = (axvspan_ymin_top - pos_top.y0) / pos_top.height
# axes_fraction_ymax_top = (axvspan_ymax_top - pos_top.y0) / pos_top.height
#
# # Update the axvspan in the top plot with adjusted ymin and ymax
# ax_top.axvspan(
#     0.4, 3.4, # 5000
#     ymin=axes_fraction_ymin_top,
#     ymax=axes_fraction_ymax_top,
#     color='red',
#     transform=ax_top.get_xaxis_transform(),
#     clip_on=False
# )
#
# # Add a single colorbar for both plots
# cax = fig.add_subplot(gs[:, 1])
# cbar = fig.colorbar(img1, cax=cax)
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label('Spikes per second', size=16)
#
# # Save and display the combined figure
# plt.tight_layout()
# plt.savefig('/Users/nd23721/Documents/marm-mac/combined_plot.png', dpi=300)
# plt.show()

###################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set the default DPI for high-resolution plots
plt.rcParams['figure.dpi'] = 300

# Load data for the first plot (EXCITATORY population rastergram)
data1 = np.load("/Users/nd23721/Documents/spike_monitor_data_simulation_0.npz")
neuron_indices1 = data1['indices']
spike_times1 = data1['times']
num_neurons1 = neuron_indices1.max() + 1  # Total number of neurons in the first population

# Convert neuron indices to degrees for the first plot
neuron_degrees1 = neuron_indices1 * (360.0 / num_neurons1)

# Load data for the second plot (FIXATION EXCITATORY population rastergram)
data2 = np.load("/Users/nd23721/Documents/spike_monitor_fix_data_simulation_0.npz")
neuron_indices2 = data2['indices_fix']
spike_times2 = data2['times_fix']
num_neurons2 = neuron_indices2.max() + 1  # Total number of neurons in the second population

# Define common time bins for both plots
time_start = 0
time_end = 3.4 #5000
bins_time = np.linspace(time_start, time_end, 100)

# Define the time offset to shift the x-axis
time_offset = 0.4
shifted_time_start = time_start - time_offset
shifted_time_end = time_end - time_offset

# Histogram calculations for the first plot
bins_neuron1 = np.linspace(0, 360, 100)
counts1, _, _ = np.histogram2d(spike_times1, neuron_degrees1, bins=[bins_time, bins_neuron1])
time_bin_duration = bins_time[1] - bins_time[0]
neurons_per_bin1 = num_neurons1 / 100
counts_per_second_per_neuron1 = counts1 / (time_bin_duration * neurons_per_bin1)

# Reorder the counts for the first plot using advanced indexing
num_bins1 = counts_per_second_per_neuron1.shape[1]
new_order = list(range(3 * num_bins1 // 4, num_bins1)) + list(range(3 * num_bins1 // 4))
counts_per_second_per_neuron1 = counts_per_second_per_neuron1[:, new_order]

# Histogram calculations for the second plot
bins_neuron2 = np.linspace(0, num_neurons2, 100)
counts2, _, _ = np.histogram2d(spike_times2, neuron_indices2, bins=[bins_time, bins_neuron2])
neurons_per_bin2 = num_neurons2 / 100
counts_per_second_per_neuron2 = counts2 / (time_bin_duration * neurons_per_bin2)

# Create the figure and GridSpec layout
fig = plt.figure(figsize=(10, 6), dpi=300)
gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[num_neurons2, num_neurons1])

# Create subplots for the two rastergrams
ax_top = fig.add_subplot(gs[0, 0])
ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

# Define the maximum firing rate for color scaling
vmax = 50.0  # Assuming Hz = 1.0

# Plot the second rastergram (top plot)
clipped_counts2 = np.minimum(counts_per_second_per_neuron2, vmax)
img2 = ax_top.imshow(
    clipped_counts2.T,
    extent=[shifted_time_start, shifted_time_end, 0, num_neurons2],
    aspect='auto',
    origin='lower',
    cmap='viridis',
    norm=Normalize(vmin=0, vmax=vmax)
)

# Remove y-axis labels and ticks for the second plot
#ax_top.set_ylabel('')  # Remove the y-axis label
#ax_top.set_ylabel('Fixation Rule\nE cells', fontsize=13)
ax_top.set_ylabel('Fixation Rule\nE cells', fontsize=15, rotation=0, labelpad=20)
ax_top.yaxis.set_label_coords(-0.1, 0.1)  # Adjust the x and y coordinates as needed
ax_top.set_yticks([])  # Remove the y-axis ticks

# ax_top.set_ylabel('Neuron index', fontsize=18)
# y_ticks = [0, num_neurons2 / 4, num_neurons2 / 2, 3 * num_neurons2 / 4, num_neurons2]
# ax_top.yaxis.set_major_locator(FixedLocator(y_ticks))
# ax_top.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
ax_top.tick_params(axis='both', which='major', labelsize=16)
ax_top.set_xlim([shifted_time_start, shifted_time_end])
plt.setp(ax_top.get_xticklabels(), visible=False)  # Hide x-axis labels on the top plot

# Plot the first rastergram (bottom plot)
clipped_counts1 = np.minimum(counts_per_second_per_neuron1, vmax)
img1 = ax_bottom.imshow(
    clipped_counts1.T,
    extent=[shifted_time_start, shifted_time_end, 270, 630],
    aspect='auto',
    origin='lower',
    cmap='viridis',
    norm=Normalize(vmin=0, vmax=vmax)
)
ax_bottom.set_xlabel('Time (s)', fontsize=18)
ax_bottom.set_ylabel('Cue E cell index', fontsize=18)
ax_bottom.yaxis.set_major_locator(FixedLocator([270, 360, 450, 540, 630]))

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

ax_bottom.yaxis.set_major_formatter(FuncFormatter(format_fn))
ax_bottom.tick_params(axis='both', which='major', labelsize=16)
ax_bottom.set_xlim([shifted_time_start, shifted_time_end])
ax_bottom.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3]))
ax_bottom.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x}'))
ax_bottom.axvspan(0.6 - time_offset, 3.4 - time_offset, ymin=1.03, ymax=1.06, color='blue', transform=ax_bottom.get_xaxis_transform(), clip_on=False)

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.05)

# Get positions of the axes in figure coordinates
pos_top = ax_top.get_position()
pos_bottom = ax_bottom.get_position()

# Compute the vertical offset and height of the axvspan in the bottom plot (in figure coordinates)
offset_bottom = (1.03 - 1.0) * pos_bottom.height  # Offset from top of bottom axes to start of axvspan
height_axvspan_bottom = (1.06 - 1.03) * pos_bottom.height  # Height of the axvspan

# Compute the position of the axvspan in the top plot (in figure coordinates)
axvspan_ymin_top = pos_top.y1 + offset_bottom
axvspan_ymax_top = axvspan_ymin_top + height_axvspan_bottom

# Convert back to axes fraction units for the top plot
axes_fraction_ymin_top = (axvspan_ymin_top - pos_top.y0) / pos_top.height
axes_fraction_ymax_top = (axvspan_ymax_top - pos_top.y0) / pos_top.height

# Update the axvspan in the top plot with adjusted ymin and ymax
ax_top.axvspan(
    0.4 - time_offset, 3.4 - time_offset,
    ymin=axes_fraction_ymin_top,
    ymax=axes_fraction_ymax_top,
    color='red',
    transform=ax_top.get_xaxis_transform(),
    clip_on=False
)

#new
tick_positions = np.arange(0.0, 3.00001, 0.5)  # Tick marks corresponding to 0 to 3.0 adjusted
plt.xticks(tick_positions, [f"{t - 0.0:.1f}" for t in tick_positions], fontsize=15)  # Correct labeling
#new
# Add a single colorbar for both plots
cax = fig.add_subplot(gs[:, 1])
cbar = fig.colorbar(img1, cax=cax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Spikes per second', size=16)

# Save and display the combined figure
plt.tight_layout()
plt.savefig('/Users/nd23721/Documents/marm-mac/combined_plot_adjusted.png', dpi=300)
plt.show()

###################################################################

import numpy as np
import matplotlib.pyplot as plt

# Load the data file
file_path = '/Users/nd23721/Documents/spike_monitor_fix_data_simulation_0.npz'
data = np.load(file_path)

# Extract spike times and neuron indices
spike_times = data['times_fix']  # spike times in seconds
spike_indices = data['indices_fix']  # neuron indices

# Define the parameters for the computation
num_neurons = len(np.unique(spike_indices))  # total number of neurons
adjustment_val = 0.001  # small adjustment to the max time for binning
bins_time = np.linspace(spike_times.min(), spike_times.max() + adjustment_val, 100)
bins_neuron = np.linspace(0, num_neurons, 100)

# Calculate 2D histogram for time and neurons
counts, _, _ = np.histogram2d(spike_times, spike_indices, bins=[bins_time, bins_neuron])

# Convert counts to spikes per second per neuron
time_bin_duration = bins_time[1] - bins_time[0]  # duration of each time bin
neurons_per_bin = num_neurons / 100  # number of neurons in each neuron bin
counts_per_second_per_neuron = counts / (time_bin_duration * neurons_per_bin)

# Average firing rate across all neurons
average_firing_rate = counts_per_second_per_neuron.mean(axis=1)

# Convert time bins to centers for plotting
time_centers = bins_time[:-1] + time_bin_duration / 2

# Find the first point where the average firing rate drops below 10 Hz after 0.6 seconds
valid_indices = np.where(time_centers > 0.6)[0]
below_10_idx = np.where(average_firing_rate[valid_indices] < 10)[0]

if len(below_10_idx) > 0:
    first_below_10_idx = valid_indices[below_10_idx[0]]
    star_x = time_centers[first_below_10_idx]
    star_y = average_firing_rate[first_below_10_idx]

# Plotting the average firing rate
plt.figure(figsize=(10, 6))
plt.plot(time_centers, average_firing_rate, color='black')
plt.xlabel('Time (s)', fontsize=24)
plt.ylabel('Firing Rate (Hz)', fontsize=24)
plt.title('Average Firing Rate of Fixation Rule E cells', fontsize=24)

# Add the yellow star if the point was found
if len(below_10_idx) > 0:
    plt.scatter(star_x + 0.01, star_y, color='yellow', s=200, marker='*', edgecolor='black', zorder=10, label='Fixation Break')

# Adjust tick positions and set specific tick marks
tick_positions = np.arange(0.4, 3.40001, 0.5)  # Tick marks corresponding to 0 to 3.0 adjusted
plt.xticks(tick_positions, [f"{t - 0.4:.1f}" for t in tick_positions], fontsize=15)  # Correct labeling
plt.yticks(fontsize=15)

# Set x-axis limits from -0.4 to 3.0 seconds explicitly
plt.xlim(0.0, 3.4)
plt.ylim(0, 65)  # Cap y-axis at 60 Hz

# Add shaded regions
plt.axvspan(0.0, 0.4, color='white', alpha=0.3, label="Baseline")
plt.axvspan(0.4, 0.6, color='red', alpha=0.4, label="Fixation without Distractors")
plt.axvspan(0.6, 3.4, color='blue', alpha=0.4, label="Fixation with Distractors")

# Add legend
plt.legend(fontsize=15)

# Add gridlines
plt.grid(visible=True, which='both', linestyle='--', linewidth=2, alpha=0.7)

# Save and display the combined figure
plt.tight_layout()
plt.savefig('/Users/nd23721/Documents/marm-mac/fixation_average_FR.png', dpi=300)
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Load the most recent uploaded data
data_path_latest_update = '/Users/nd23721/Documents/spike_monitor_data_simulation_0.npz'
data_latest_update = np.load(data_path_latest_update)

# Extract the arrays from the updated dataset
neuron_indices_updated = data_latest_update['indices']
spike_times_updated = data_latest_update['times']

# Recalculate the parameters with the updated data
num_neurons_updated = neuron_indices_updated.max() + 1  # assuming indices start from 0
neuron_degrees_updated = neuron_indices_updated * (360.0 / num_neurons_updated)
total_time_length_updated = spike_times_updated.max() - spike_times_updated.min()
adjustment_val_updated = 3.4 - total_time_length_updated

# Define histogram bins
bins_time_updated = np.linspace(spike_times_updated.min(), spike_times_updated.max() + adjustment_val_updated, 100)
bins_neuron_updated = np.linspace(0, 360, 100)

# Calculate 2D histogram (time vs. neuron degree bins)
counts_updated, _, _ = np.histogram2d(spike_times_updated, neuron_degrees_updated, bins=[bins_time_updated, bins_neuron_updated])

# Normalize the histogram counts to spikes/sec/neuron
time_bin_duration_updated = bins_time_updated[1] - bins_time_updated[0]
neurons_per_bin_updated = num_neurons_updated / 100  # Assuming 100 bins for neuron degrees
counts_per_second_per_neuron_updated = counts_updated / (time_bin_duration_updated * neurons_per_bin_updated)

# Define angles and ±15° window
angles = [0, 45, 90, 135, 180, 225, 270, 315]
angle_window = 15

# Extract firing rates for each angle within ±15 degrees
firing_rates_over_time_normalized_updated = {}
for angle in angles:
    lower_bound = (angle - angle_window) % 360
    upper_bound = (angle + angle_window) % 360
    if lower_bound < upper_bound:
        relevant_bins = (bins_neuron_updated[:-1] >= lower_bound) & (bins_neuron_updated[:-1] <= upper_bound)
    else:  # Handle wrapping around 360 degrees
        relevant_bins = (bins_neuron_updated[:-1] >= lower_bound) | (bins_neuron_updated[:-1] <= upper_bound)
    average_rate = counts_per_second_per_neuron_updated[:, relevant_bins].mean(axis=1)  # Mean rate over relevant bins
    firing_rates_over_time_normalized_updated[angle] = average_rate

# Plot the normalized firing rates over time for each angle with the updated data
plt.figure(figsize=(10, 6))
for angle, rates in firing_rates_over_time_normalized_updated.items():
    plt.plot(bins_time_updated[:-1], rates, label=f'{angle}°', linewidth=2)

plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Firing Rate (Hz)', fontsize=14)
plt.title('Normalized Firing Rates Over Time for Each Central Angle (Updated Data)', fontsize=16)
plt.legend(title='Central Angle', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# Load the data files
data1 = np.load("/Users/nd23721/Documents/spike_monitor_data_simulation_0.npz")
data2 = np.load("/Users/nd23721/Documents/spike_monitor_fix_data_simulation_0.npz")

# Extract data for the first plot (EXCITATORY population rastergram)
neuron_indices1 = data1['indices']
spike_times1 = data1['times']
num_neurons1 = neuron_indices1.max() + 1
neuron_degrees1 = neuron_indices1 * (360.0 / num_neurons1)

# Extract data for the second plot (FIXATION EXCITATORY population rastergram)
neuron_indices2 = data2['indices_fix']
spike_times2 = data2['times_fix']
num_neurons2 = neuron_indices2.max() + 1

# Define common time bins for both plots
time_start = 0
time_end = 3.4
bins_time = np.linspace(time_start, time_end, 100)

# Histogram calculations for the first plot
bins_neuron1 = np.linspace(0, 360, 100)
counts1, _, _ = np.histogram2d(spike_times1, neuron_degrees1, bins=[bins_time, bins_neuron1])
time_bin_duration = bins_time[1] - bins_time[0]
neurons_per_bin1 = num_neurons1 / 100
counts_per_second_per_neuron1 = counts1 / (time_bin_duration * neurons_per_bin1)

# Histogram calculations for the second plot
bins_neuron2 = np.linspace(0, num_neurons2, 100)
counts2, _, _ = np.histogram2d(spike_times2, neuron_indices2, bins=[bins_time, bins_neuron2])
neurons_per_bin2 = num_neurons2 / 100
counts_per_second_per_neuron2 = counts2 / (time_bin_duration * neurons_per_bin2)

# Average firing rate for the top plot
average_firing_rate = counts_per_second_per_neuron2.mean(axis=1)
time_centers_fix = bins_time[:-1] + time_bin_duration / 2

#new

# Find the first point where the average firing rate drops below 10 Hz after 0.6 seconds
valid_indices = np.where(time_centers > 0.6)[0]
below_10_idx = np.where(average_firing_rate[valid_indices] < 10)[0]

if len(below_10_idx) > 0:
    first_below_10_idx = valid_indices[below_10_idx[0]]
    star_x = time_centers[first_below_10_idx]
    star_y = average_firing_rate[first_below_10_idx]

#new

# Create the figure and GridSpec layout
fig = plt.figure(figsize=(10, 10), dpi=300)
gs = GridSpec(3, 2, height_ratios=[1, 0.125, 1], width_ratios=[1, 0.05])

# Top plot (Average Firing Rate, now first)
ax_top = fig.add_subplot(gs[0, 0])
ax_top.plot(time_centers_fix, average_firing_rate, color="black")
ax_top.set_ylabel("Firing Rate (Hz)", fontsize=18, labelpad=10)
ax_top.yaxis.set_label_coords(-0.12, 0.5)  # Align with other plots
ax_top.set_xlim(time_start, time_end)
ax_top.set_ylim(0, 65)
ax_top.tick_params(axis="both", which="major", labelsize=15, bottom=False, labelbottom=False)  # Remove tick labels
ax_top.set_xticks(np.arange(0.4, 3.5, 0.5))
# tick_labels = [f"{t - 0.4:.1f}" for t in np.arange(0.4, 3.5, 0.5)]
# ax_top.set_xticklabels(tick_labels)
ax_top.grid(visible=True, which="both", linestyle="--", linewidth=1, alpha=0.7)

#new

# Add the yellow star if the point was found
if len(below_10_idx) > 0:
    ax_top.scatter(star_x + 0.01, star_y, color='yellow', s=200, marker='*', edgecolor='black', zorder=10, label='Fixation Break')

#new

# Vertical lines and updated legend for top plot
ax_top.axvline(x=0.4, color="red", linestyle="--", linewidth=2) #, label="Fixation Onset"
ax_top.axvline(x=0.6, color="blue", linestyle="--", linewidth=2) #, label="Distractors Onset"
ax_top.legend(fontsize=15)

# Middle plot (Fixation Rule E cells, now second)
ax_middle = fig.add_subplot(gs[1, 0])
img2 = ax_middle.imshow(
    counts_per_second_per_neuron2.T,
    extent=[time_start, time_end, 0, num_neurons2],
    aspect="auto",
    origin="lower",
    cmap="viridis",
    norm=Normalize(vmin=0, vmax=50)
)
ax_middle.set_ylabel("Fixation Rule\nE cells", fontsize=15, rotation=0, labelpad=20)
ax_middle.yaxis.set_label_coords(-0.12, 0.1)  # Move the label further to the left
ax_middle.set_yticks([])
ax_middle.tick_params(axis="both", which="major", labelsize=16, bottom=True, labelbottom=False)
ax_middle.set_xlim([time_start, time_end])
ax_middle.set_xticks(np.arange(0.4, 3.5, 0.5))
plt.setp(ax_middle.get_xticklabels(), visible=False)

# Bottom plot (Cue E cells, now third)
ax_bottom = fig.add_subplot(gs[2, 0], sharex=ax_middle)
img1 = ax_bottom.imshow(
    counts_per_second_per_neuron1.T,
    extent=[time_start, time_end, 270, 630],
    aspect="auto",
    origin="lower",
    cmap="viridis",
    norm=Normalize(vmin=0, vmax=50)
)
ax_bottom.set_ylabel("Cue E cell index", fontsize=18, labelpad=15)
ax_bottom.yaxis.set_major_locator(FixedLocator([270, 360, 450, 540, 630]))
ax_bottom.yaxis.set_major_formatter(
    FuncFormatter(lambda val, pos: f"{val - 360:.0f}°" if val >= 360 else f"{val:.0f}°")
)
ax_bottom.tick_params(axis="both", which="major", labelsize=16, bottom=True, labelbottom=True)  # Add tick labels
ax_bottom.set_xlim([time_start, time_end])
ax_bottom.set_xticks(np.arange(0.4, 3.5, 0.5))
tick_labels = [f"{t - 0.4:.1f}" for t in np.arange(0.4, 3.5, 0.5)]
ax_bottom.set_xticklabels(tick_labels)
ax_bottom.set_xlabel("Time (s)", fontsize=18)  # Add x-axis label to the bottom plot

# Adding the blue line above the bottom plot (Cue E cells)
ax_bottom.axvspan(
    0.6, 3.4,
    ymin=1.02, ymax=1.04,
    color="blue",
    transform=ax_bottom.get_xaxis_transform(),
    clip_on=False
)

# Adjusting red bar's position above the middle plot
pos_middle = ax_middle.get_position()
pos_bottom = ax_bottom.get_position()

# Compute the offset and height of the axvspan for the middle plot
offset_middle = (1.02 - 1.0) * pos_bottom.height  # Offset based on bottom height
height_axvspan_middle = (1.04 - 1.02) * pos_bottom.height  # Height based on bottom height

# Compute red bar's position for the middle plot
axvspan_ymin_middle = pos_middle.y1 + offset_middle
axvspan_ymax_middle = axvspan_ymin_middle + height_axvspan_middle

# Adjust to axes fraction coordinates for ax_middle
axes_fraction_ymin_middle = (axvspan_ymin_middle - pos_middle.y0) / pos_middle.height
axes_fraction_ymax_middle = (axvspan_ymax_middle - pos_middle.y0) / pos_middle.height

# Adding the red line to the middle plot
ax_middle.axvspan(
    0.4, 3.4,
    ymin=axes_fraction_ymin_middle,
    ymax=axes_fraction_ymax_middle,
    color="red",
    transform=ax_middle.get_xaxis_transform(),
    clip_on=False
)

# Add a shared colorbar only for the last two plots (middle and bottom)
cax = fig.add_subplot(gs[1:, 1])  # Limited to rows for middle and bottom plots
cbar = fig.colorbar(img1, cax=cax)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("Spikes per second", size=16)

plt.tight_layout()
plt.show()
