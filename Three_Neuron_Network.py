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
    g = 0.5*(1.0+np.tanh(np.divide(V[:,None]-A[None,:],B[None,:])))
    return g

def S0(V_presyn,V0,dV0):
    """
    Asymptotic value of synapse variable (analogous to gating variable).

    :param V_presyn: Voltage (mV) of presynaptic neuron; dim = (# neurons)
    :param V0: Inflection voltage (where S0(V)=0) (mV) of presynaptic neuron; (scalar)
    :param dV0: Length scale of tanh (mV) for presynaptic neuron; (scalar)
    :return: S0; dim = (# neurons)
    """
    S0 = 0.5*(1.0+np.tanh(np.divide(V_presyn-V0,dV0)))
    return S0

def tau(V,t0,t1,A,B):
    """
    Time constant of gating variable ODEs.

    :param V: Voltage (mV); dim=(# neurons)
    :param t0: Constant (ms)
    :param t1: Constant (ms)
    :param A: Location of g(V)=0 (mV);   dim=(gating variables,)
    :param B: Length scale of tanh (mV);   dim=(gating variables,)
    :return: tau (Voltage-dependent time constant) (ms); dim=(# neurons, # gating variables)
    """
    epsilon = 0.0000000000000001
    tau = t0+t1*(1.0-np.power(np.tanh(np.divide(V[:,None]-A[None,:],B[None,:]+epsilon)),2))
    return tau


def calc_dAdt(V,A,vA,dvA,tA0,tA1):
    """
    Time derivative of ion channel gating variable "A" (example: m, h, n)

    :param V: (mV)
    :param A: gating variable (unitless); dim=(# gating variables,# neurons)
    :param vA: Inflection voltage (where g(V)=0) (mV)
    :param dvA: Length scale of tanh (mV)
    :param tA0: Constant (ms)
    :param tA1: Constant (ms)
    :return: dAdt; dim=(# neurons, # gating variables)
    """
    epsilon = 0.0000000000000001
    dAdt = np.divide(g(V,vA,dvA) - A, tau(V,tA0,tA1,vA,dvA) +epsilon)
    return dAdt

def calc_dSdt(V_presyn, S, V0, dV0, tau_syn, S_syn):
    """
    Time derivative of synapse "gating variable" "S" (analogous to ion channel gating variables: m, h, n)

    :param V_presyn: Presynaptic neuron voltage (mV); dim = (# neurons)
    :param S: Synapse variable (unitless); # dim index =(postsyn neuron index, presyn neuron index)
    :param V0: Inflection voltage (where S0(V)=0) (mV) of presynaptic neuron; (scalar)
    :param dV0: Length scale of tanh (mV) of presynaptic neuron; (scalar)
    :param tau_syn: tau_1 = excitatory, tau_2 = inhibitory; dim = (postsyn neuron index, presyn neuron index)
    :param S_syn: S_1=excitatory, S_2 = inhibitory; dim = (postsyn neuron index, presyn neuron index)
    :return: dSdt; dim index =(postsyn neuron index, presyn neuron index)
    """
    epsilon = 0.0000000000000001
    S0_value = S0(V_presyn,V0,dV0) # dim = (# postsynaptic neurons)
    dSdt = np.divide( (S0_value[None,:]-S), (np.multiply(tau_syn,(S_syn-S0_value[None,:]))) +epsilon)
    return dSdt


def calc_dVdt(V, A_modulation, g_internal_gating, E_rev_gate_matrix, S_modulation, E_rev_syn_matrix, g_syn):
    """
    Voltage calculation. Accepts vectors and matrices. See documentation for necessary dimension.

    :param V: V (mV) dim: (# neurons,); Note: if using V[None,:], this represents PREsynaptic neurons
    :param A_modulation: dim: (# neurons, # ion channels)
                            [[m1,   h1,    n1]
                            [m2,    h2,    n2]
                             ....
                            [mLast, hLast, nLast]]
    :param g_internal_gating: dim: (# ion channels)
    :param E_rev_gate_matrix: dim: (# neurons, # ion channels)
    :param S_modulation: dim: (# postsyn neurons, # presyn neurons)
                        [[S11,   S12,       S13]
                        [S21,    S22,       S23]
                         ....
                        [SLast1, SLast2, SLast3]]
    :param E_rev_syn_matrix: dim: (# postsyn neurons, # presyn neurons)
    :param g_syn: dim: (# postsyn neurons, # presyn neurons)
    :return: dVdt: dim (# neurons)
    """
    dVdt_internal = np.dot(np.multiply(A_modulation, E_rev_gate_matrix - V[:,None]), g_internal_gating)
    # dVdt_internal = np.sum(np.multiply(np.multiply(A_modulation, E_rev_gate_matrix - V[:,None]), g_internal_gating[None,:]),axis=1)
    # test_multiplication_a =  np.multiply(S_modulation, E_rev_syn_matrix)
    # test_multiplication_b =  np.multiply(S_modulation, V[None,:])
    #
    # test_multiplication =  np.multiply(S_modulation, E_rev_syn_matrix - V[None,:])
    # test_2_multi = np.multiply(g_syn, test_multiplication)
    dVdt_synapse = np.sum(np.multiply(g_syn, np.multiply(S_modulation, E_rev_syn_matrix - V[None,:])),axis=1)
    return dVdt_internal + dVdt_synapse # TODO: Remove the zero from this line

def dStatedt(t, state, params):
    """
    Run this inside ODEInt or SciPy's integrate. Contains vectorized ODEs for 3 NaKL neurons.
    
    :param t:
    :param state:
    :param params:
    :return:
    """
    # V1, V2, V3 = state[0:3]
    # m1, m2, m2 = state[3:6]
    # h1, h2, h2 = state[6:9]
    # n1, n2, n2 = state[9:12]
    vA = params[0]["vA"]
    dvA = params[0]["dvA"]
    tA0 = params[0]["tA0"]
    tA1 = params[0]["tA1"]
    tau_syn_arr = params[0]["tau_syn_arr"] # dim = (postsyn neuron index, presyn neuron index)
    S_syn_arr = params[0]["S_syn_arr"] # dim = (postsyn neuron index, presyn neuron index)
    g_internal_gating = params[0]["g_internal_gating"]
    E_rev_gate_matrix = params[0]["E_rev_gate_matrix"]
    E_rev_syn_matrix = params[0]["E_rev_syn_matrix"]
    g_syn = params[0]["g_synapse_gating"]
    I_ext = params[0]["I_ext"]

    # Prepare for matrix multiplication later in function
    V = state[0:3]
    m = state[3:6]
    h = state[6:9]
    n = state[9:12]
    # S = state[12:16] # S12, S23, S31, S32 = state[12:16]
    S12, S23, S31, S32 = state[12:16]
    S_arr = np.array([[0, S12, 0],
                      [0, 0, S23],
                      [S31, S32, 0]]) # dim index =(postsyn neuron index, presyn neuron index)

    A = np.array([m,
                  h,
                  n]).transpose() # dim=(# neurons, # gating variables)
    m3h = np.multiply(np.power(m, 3), h)
    A_modulation = np.array([m3h,
                            np.power(n,4),
                            np.ones(V.size)]).transpose() # dim(3,3) or (# neurons, # ion channels)

    # Perform matrix multiplications
    # Voltages: dim(dVdt)=(# neurons,)
    dVdt = calc_dVdt(V, A_modulation, g_internal_gating, E_rev_gate_matrix,
                     S_arr, E_rev_syn_matrix, g_syn) + I_ext
    # Gating variables: dim(dAdt)=(#gating variables,# neurons)
    dAdt = calc_dAdt(V,A,vA,dvA,tA0,tA1).transpose()
    # Synapses: dim(dSdt)=(# neurons,# neurons); dim index =(postsyn neuron index, presyn neuron index)
    dSdt = calc_dSdt(V, S_arr, V0, dV0, tau_syn_arr, S_syn_arr)
    dStatedt = np.empty((16))
    dStatedt[0:3] = dVdt
    dStatedt[3:12] = np.ravel(dAdt)
    dStatedt[12:16] = np.array([dSdt[0][1],dSdt[1][2],dSdt[2][0],dSdt[2][1]])
    return dStatedt


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
g12 = 0.35#80*0.35
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

I = 40.0
I_ext = np.array([0,
                  I,
                  I/4])# External stimulus (nA)

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



tau_syn_arr = np.array([[0    , tau1, 0],
                        [0    , 0, tau1],
                        [tau2,   tau2, 0]]) # (# neurons, # neurons)
S_syn_arr = np.array([[0    , S1, 0],
                        [0    , 0, S1],
                        [S2,   S2, 0]]) # (# neurons, # neurons)

# dictionary needed to pass constants into ODEInt:
constants_dict = {"vA":vA,
                  "dvA":dvA,
                  "tA0":tA0,
                  "tA1":tA1,
                  "tau_syn_arr":tau_syn_arr,
                  "S_syn_arr":S_syn_arr,
                  "g_internal_gating" : g_internal_gating,
                  "E_rev_gate_matrix" : E_rev_gate_matrix,
                  "E_rev_syn_matrix" : E_rev_syn_matrix,
                  "g_synapse_gating" : g_syn,
                  "I_ext":I_ext}
params = [constants_dict]

# What do I want the end result to look like?
t_start = 0.0 # ms
t_stop = 200.0 # ms
dt = 0.02     # ms
times_array = np.arange(start= t_start,stop= t_stop, step=dt)

state_initial = 0.1*np.ones((16))
# state_initial[0:4] = 0.0 # voltages
# state_initial[4:12] = 0.0 # gating variables
# state_initial[12:16] = 0.0 # synapse variables

# Solve system
sol = scipy.integrate.solve_ivp(fun=dStatedt, t_span=[t_start,t_stop], y0=state_initial, t_eval=times_array, args=(params,))


# Plotting
fig = plt.figure(figsize=(20, 10))
axs =  fig.add_subplot(111)
axs.plot(sol.t,(sol.y[0]).transpose(), label="V1")
axs.plot(sol.t,(sol.y[1]).transpose(), label="V2")
axs.plot(sol.t,(sol.y[2]).transpose(), label="V3")
# axs.plot(sol.t,(sol.y[12]).transpose(), label="S12")

axs.set_xlabel("Time (ms)")
axs.set_ylabel("State variable")
axs.legend()
plt.title("Simulated 3 NaKL Network")
fig.show()

saved_data = np.vstack((sol.t,sol.y)).transpose()
print(f"saved_data.shape:{saved_data.shape}")
save_utilities.save_text(saved_data,"save","simulated_data/3NN/solution.txt")
save_utilities.save_fig_with_makedir(figure=fig,
                                     save_location="simulated_data/3NN/solution.png")
plt.close("all")