# class NaKL_neuron():
#
#     def __init__(self, C = 1.0, gN = 120.0, vNa = 50.0,
#                  gK = 20.0, vK = -77.0, gL = 0.3, vL = -54.4, vN = -55.0, dvn = 30.0,
#                  tn0 = 1.0, tn1 = 5.0, vm = -40.0, dvm = 15.0, tm0 = 0.1, tm1 = 0.4, vh = -60.0,
#                  dvh = -15.0, th0 = 1.0, th1 = 7.0,
#                  g12 = 0.35, g23 = 0.27, g32 = 0.215, g31 = 0.203,
#                  Ereve = 0.0, Erevi = -80.0, tau1 = 1.0, tau2 = 3.0, S1 = 3.0/2.0, S2 = 5.0/3.0):
#
#
#         # current in nA
#         # units of g: muS
#
#         # intraneuronal variables
#         self.V = -30.0 # chosen arbitrarily
#         self.m = 0  # chosen arbitrarily
#         self.h = 0  # chosen arbitrarily
#         self.n = 0 # chosen arbitrarily
#
#         self.C = C # nF
#         self.gN = gN
#         self.vNa = vNa
#         self.gK = gK
#         self.vK =  vK
#         self.gL = gL
#         self.vL =  vL
#
#         self.vN =  vN
#         self.dvn = dvn
#         self.tn0 = tn0
#         self.tn1 = tn1
#         self.vm =  vm
#         self.dvm = dvm
#         self.tm0 = tm0
#         self.tm1 = tm1
#         self.vh = vh
#         self.dvh = dvh
#         self.th0 = th0
#         self.th1 = th1
#
#         # synaptic variables
#         self.g12 =   g12
#         self.g23 =   g23
#         self.g32 =   g32
#         self.g31 =   g31
#         self.Ereve = Ereve
#         self.Erevi = Erevi
#         self.tau1 =  tau1
#         self.tau2 =  tau2
#         self.S1 =    S1
#         self.S2 =    S2
#
#
#     def dVdt(self, V, synapse_g_matrix, E_rev_matrix):
#         dVdt = 0
#
#         # internal dynamics: Na
#         dVdt += self.gN*(self.m**3)*self.h*(self.vN - self.V)
#         # internal dynamics: K
#         dVdt += self.gK * self.n * (self.vK - self.V)
#         # internal dynamics: L
#         dVdt += self.gL * (self.vL - self.V)
#         # synapses
#         dVdt += synapse_g_matrix * SYNAPSE STUFF * (E_rev_matrix - self.V)
#
#         return dVdt


# class network():
#     def __init__(self):
#         # synaptic variables
#         self.g12 = 0.35
#         self.g23 = 0.27
#         self.g32 = 0.215
#         self.g31 = 0.203
#         self.Ereve = 0.0
#         self.Erevi = -80.0
#         self.tau1 = 1.0
#         self.tau2 = 3.0
#         self.S1 = 3.0 / 2.0
#         self.S2 = 5.0 / 3.0
#
#         self.neuron_list = None
#         self.synapse_g_matrix = None
#         self.exc_inh_matrix = None
#         self.dVNeurondt = None
#
#
#     def prepare(self):
#         self.dANeurondt = []
#         for neuron in neuron_list:
#             self.dANeurondt.append(neuron.dVdt)
#
#     def dNetworkdt(self):
#
#     def integrate(self, times_array):
#
#         sol = odeint(self.dNetworkdt, times_array)
#
#         return sol
#
#
#
#
# # What do I want the end result to look like?
# t_start = 0.0 # ms
# t_stop = 10.0 # ms
# dt = 0.02     # ms
# times_array = np.arange(start= t_start,stop= t_stop, step=dt)
#
# NaKL1 = NaKL_neuron()
# NaKL2 = NaKL_neuron()
# NaKL3 = NaKL_neuron()
# neuron_list = [NaKL1, NaKL2, NaKL3]
#
# network1 = network()
# network1.neuron_list = neuron_list
# network1.synapse_g_matrix = synapse_g_matrix
# network1.exc_inh_matrix   = exc_inh_matrix
# network1.prepare()
# network1.integrate(times_array)
