import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import save_utilities

def g(V,A,B):
    """
    Asymptotic value of gating variable.

    :param V: Voltage (mV); dim=(# neurons,)
    :param A: Inflection voltage (where g(V)=0) (mV); dim=(gating variables,)
    :param B: Length scale of tanh (mV);  dim=(gating variables,)
    :return: g (muS); dim=(# neurons, # gating variables)
    """
    g = 0.5*(1.0+np.tanh(np.divide(float(V[:,None]-A[None,:]),float(B[None,:]))))
    print(f"dim(g)={g.shape}")
    return g

def S0(V_presyn,V0,dV0):
    # TODO: Calculate this dimension first
    """
    Asymptotic value of synapse variable (analogous to gating variable).

    :param V_presyn: Voltage (mV) of presynaptic neuron
    :param V0: Inflection voltage (where S0(V)=0) (mV)
    :param dV0: Length scale of tanh (mV)
    :return: S0 (unitless (?)) TODO: Check return dimension
    """
    S0 = 0.5*(1.0+np.tanh(np.divide(float(V_presyn[:,None]-V0[None,:]),float(dV0[None,:]))))
    print(f"dim(S0)={S0.shape}")
    return S0

def tau(V,t0,t1,A,B):
    """
    Time constant of gating variable ODEs.

    :param V: Voltage (mV)
    :param t0: Constant (ms)
    :param t1: Constant (ms)
    :param A: Location of g(V)=0 (mV)
    :param B: Length scale of tanh (mV)
    :return: tau (Voltage-dependent time constant) (ms) TODO: Check return dimension
    """
    tau = t0+t1*(1.0-np.power(np.tanh(np.divide(float(V[:,None]-A[None,:]),float(B[None,:]))),2))
    print(f"dim(tau)={tau.shape}")
    return tau


def calc_dAdt(V,A,vA,dvA,tA0,tA1):
    # TODO: Need tau() dimension before this dim can be calculated
    """
    Time derivative of ion channel gating variable "A" (example: m, h, n)

    :param V: (mV)
    :param A: gating variable (unitless); dim=(# gating variables,# neurons)
    :param vA: Inflection voltage (where g(V)=0) (mV)
    :param dvA: Length scale of tanh (mV)
    :param tA0: Constant (ms)
    :param tA1: Constant (ms)
    :return: dAdt (unitless) TODO: Check return dimension
    """
    dAdt = np.divide((g(V,vA,dvA) - A), tau(V,tA0,tA1,vA,dvA))
    print(f"dim(dAdt)={dAdt.shape}")
    return dAdt

def calc_dSdt(V_presyn, S, V0, dV0, tau_syn, S_syn):
    # TODO: Need S0 dimension before this dim can be calculated
    """
    Time derivative of synapse "gating variable" "S" (analogous to ion channel gating variables: m, h, n)

    :param V_presyn: Presynaptic neuron voltage (mV); dim = (# neurons)
    :param S: Synapse variable (unitless); # dim index =(postsyn neuron index, presyn neuron index)
    :param V0: Inflection voltage (where S0(V)=0) (mV)
    :param dV0: Length scale of tanh (mV)
    :param tau_syn: tau_1 = excitatory, tau_2 = inhibitory; dim = (2,)
    :param S_syn: S_1=excitatory, S_2 = inhibitory; dim=(2,)
    :return: dSdt (unitless) TODO: Check return dimension
    """
    S0_value = S0(V_presyn,V0,dV0)
    dSdt = np.divide( (S0_value-S), (np.multiply(tau_syn,(S_syn-S0_value))) )
    print(f"dim(dSdt)={dSdt.shape}")
    return dSdt


def calc_dVdt(V, A_modulation, g_internal_gating, E_rev_gate_matrix):
    """
    Voltage calculation. Accepts vectors and matrices. See documentation for necessary dimension.

    :param V: V (mV) dim: (# neurons,)
    :param A_modulation: dim: (# neurons, # ion channels)
                            [[m1,   h1,    n1]
                            [m2,    h2,    n2]
                             ....
                            [mLast, hLast, nLast]]
    :param g_internal_gating: dim: (# ion channels)
    :param E_rev_gate_matrix: dim: (# neurons, # ion channels)
    :param E_rev_syn_matrix: dim: (# neurons, # neurons)
    :return: dVdt: dim (# neurons)
    """
    dVdt = np.dot(np.multiply(A_modulation, E_rev_gate_matrix[0] - V), g_internal_gating)
    return dVdt


# capacitance (units: nF)
C = 1.0

# conductances and ion channel reversal potentials
gN = 120.0
vNa = 50.0
gK = 20.0
vK = -77.0
gL = 0.3
vL = -54.4

# n gating variable constants
vn = -55.0 # inflection voltage for n gating variable
dvn = 30.0
tn0 = 1.0
tn1 = 5.0

# m gating variable constants
vm = -40.0 # inflection voltage for m gating variable
dvm = 15.0
tm0 = 0.1
tm1 = 0.4

# h gating variable constants
vh = -60.0 # inflection voltage for h gating variable
dvh = -15.0
th0 = 1.0
th1 = 7.0

# synaptic constants
g12 = 0.35
g23 = 0.27
g32 = 0.215
g31 = 0.203
Ereve = 0.0 # excitatory reversal potential
Erevi = -80.0 # inhibitory reversal potential
tau1 = 1.0 # excitatory S time constant
tau2 = 3.0 # inhibitory S time constant
S1 = 3.0 / 2.0
S2 = 5.0 / 3.0

V0  = -5.0
dV0 = 5.0

# Vector equivalents to use later
vA = np.array([vm,
               vh,
               vn])  # dim=(gating variables,)
dvA = np.array([dvm,
                dvh,
                dvn])  # dim=(gating variables,)
tA0 = np.array([tm0,
                th0,
                tn0])  # dim=(gating variables,)
tA1 = np.array([tm1,
                th1,
                tn1])  # dim=(gating variables,)

# dictionary needed to pass constants into ODEInt:
constants_dict = {"vA":vA,
                  "dvA":dvA,
                  "tA0":tA0,
                  "tA1":tA1}
params = [constants_dict]

# Internal ion channels
g_internal_gating = np.array([gN,
                              gK,
                              gL]) #(# ion channels,)
E_rev_gate_matrix =   np.array([[vNa    , vK,      vL],
                                [vNa    , vK,      vL],
                                [vNa    , vK,      vL]]) # (# neurons, # ion channels)

# Synaptic
g_syn = np.array([[0    , g12, 0],
                  [0    , 0, g23],
                  [g31,   g32, 0]]) # (# neurons, # neurons)
E_rev_syn_matrix = np.array([[0    ,   Ereve,       0],
                             [0    ,   0,       Ereve],
                             [Erevi,   Erevi,       0]]) # (# neurons, # neurons)



def dStatedt(state, params):
    # V1, V2, V3 = state[0:3]
    # m1, m2, m2 = state[3:6]
    # h1, h2, h2 = state[6:9]
    # n1, n2, n2 = state[9:12]
    vA = params[0]["vA"]
    dvA = params[0]["dvA"]
    tA0 = params[0]["tA0"]
    tA1 = params[0]["tA1"]

    # Prepare for matrix multiplication later in function
    V = state[0:3]
    m = state[3:6]
    h = state[6:9]
    n = state[9:12]
    # S = state[12:16] # S12, S23, S31, S32 = state[12:16]
    S12, S23, S31, S32 = state[12:16]
    synapse_arr = np.array([[0, S12, 0],
                            [0, 0, S23],
                            [S31, S32, 0]]) # dim index =(postsyn neuron index, presyn neuron index)

    A = np.array([m,
                  h,
                  n]) # dim=(# gating variables,# neurons)

    # Perform matrix multiplications
    m3h = np.multiply(np.power(m, 3), h)
    A_modulation = np.array([m3h,
                            np.power(n,4),
                            1.0]).transpose() # dim(3,3) or (# neurons, # ion channels)

    # Voltages: dim(dVdt)=(# neurons,)
    dVdt = calc_dVdt(V, A_modulation, g_internal_gating, E_rev_gate_matrix)
    # Gating variables: dim(dAdt)=(3,)
    dAdt = calc_dAdt(V,A,vA,dvA,tA0,tA1)
    # Synapses: dim(dSdt)=(3,3)
    tau_syn = np.array([tau1, tau2])
    S_syn   = np.array([S1, S2])
    dSdt = calc_dSdt(V, synapse_arr, V0, dV0, tau_syn, S_syn) # TODO: Arguments are not correctly set yet

    dStatedt = np.ravel(np.array([dVdt, dAdt, dSdt])) # flatten array
    return dStatedt

# What do I want the end result to look like?
t_start = 0.0 # ms
t_stop = 100.0 # ms
dt = 0.02     # ms
times_array = np.arange(start= t_start,stop= t_stop, step=dt)

state_initial = np.zeros((16))

# Solve system
sol = odeint(dStatedt, times_array, state_initial, args=(params,))



# Plotting
fig = plt.figure(figsize=(20, 10))
axs =  fig.add_subplot(111)
axs.plot(times_array,sol.transpose())
axs.set_xlabel("Time (ms)")
axs.set_ylabel("State variable")
plt.title("Simulated 3 NaKL Network")
fig.show()

save_utilities.save_text(sol,"save","simulated_data/3NN/solution.txt")
save_utilities.save_fig_with_makedir(figure=fig,
                                     save_location="simulated_data/3NN/solution.png")
plt.close("all")
