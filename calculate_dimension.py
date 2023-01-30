import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time

# Good resource: https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/chaospaper/node30.html

def correlation_dimension(arr):
    """
    Calculate correlation dimension of time series "arr". See Wikipedia page on "correlation dimension"

    :param arr: (# timesteps, # spatial coords)
    :return: dimension (scalar)
    """
    typical_distance_between_closest_points = np.average(np.linalg.norm(arr-np.roll(arr,shift=1,axis=0),axis=1))
    epsilon_list = [typical_distance_between_closest_points*i for i in range(1,5)]
    # epsilon_1 = typical_distance_between_closest_points * 5
    # epsilon_2 = typical_distance_between_closest_points * 10
    # print(f"epsilon_1 = {epsilon_1}")
    # print(f"epsilon_2 = {epsilon_2}")

    print(f"Using array of shape {arr.shape}")

    N = arr.shape[0]
    C_list = []

    def calc_C(arr, epsilon):
        # calculate g by counting total number out of all N^2 time points less than distance epsilon
        g = 0

        # difference_matrix = np.subtract.outer(arr, arr)
        # g_matrix = np.zeros((difference_matrix.shape))
        # g_matrix[difference_matrix < epsilon] == 1
        # g = np.sum(g_matrix)
        # print(f"g = {g}")
        num_random_t = 1000
        timestep_uncorrelated_window = 1500
        # Pick max = N or N-1, doesn't matter much
        random_t_list = np.random.random_integers(low=0,high=N-1,size=num_random_t)
        added_pair = 0
        for t_index, t in enumerate(random_t_list):
            print(f"Checking for t_index={t_index}")
            for t_other in range(N):
                distance = float(np.linalg.norm(arr[t] - arr[t_other]))
                if np.abs(t - t_other)>timestep_uncorrelated_window:
                    if distance < epsilon:
                        g += 1
                    added_pair += 1

        # Correlation integral C in the limit of large N:
        C = float(g) / (added_pair)
        return C

    for eps_index, epsilon in enumerate(epsilon_list):
        print(f"Calculating for epsilon = {epsilon} (index {eps_index})")
        C_list.append(calc_C(arr, epsilon))

        # C = g_over_N2 = (some coefficient)* epsilon^v for small epsilon
        # log(C) = log(coeff) + v*log(epsilon)
        # use log(C) vs log(epsilon)
        # dimension v is the slope
        # Calculate C_1 using epsilon_1, and C_2 using epsilon_2. Assuming coeff is same both times, then
        # log(C_2)-log(C_1) = v*(log(epsilon_2) - log(epsilon_1))
        # So then v= (log(C_2/C_1))/(log(epsilon_2/epsilon_1))
        # v = (np.log(C_2/C_1))/(np.log(epsilon_2/epsilon_1))

    # print(f"Estimated correlation dimension is {v}")
    return [C_list, epsilon_list]

program_start_time = time.time()


# imported_data = np.loadtxt("simulated_data/3NN/solution.txt")
# corr_dim = correlation_dimension(imported_data[:1000,1:])
num_datapoints = 100000

imported_data = np.loadtxt("simulated_data/Other/lorenz63_data.txt")
C_list, epsilon_list = correlation_dimension(imported_data[:num_datapoints][:])
C_arr = np.array(C_list)
epsilon_arr = np.array(epsilon_list)

plt.figure()
plt.plot(np.log(epsilon_arr), np.log(C_arr))
plt.xlabel("log(epsilon)")
plt.ylabel("log(C)")
plt.title(f"L63: C(epsilon) for N={num_datapoints} Points")
plt.show()

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(epsilon_arr,C_arr)
print(f"Slope is around {slope}")

program_end_time = time.time()
print(f"Program ran in {program_end_time-program_start_time} seconds")
