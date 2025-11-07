import numpy as np
from scipy.stats import expon, norm, pareto
import queue
import matplotlib.pyplot as plt


# Parameters
arrival_rate = 0.9684  # Arrival rate (customers/hour)
service_rate1 = 1/3  # Service rate 1 (customers/hour/server)
service_rate2 = 1/30  # Service rate 2 (customers/hour/server)
service_rate3 = 1/100  # Service rate 3 (customers/hour/server)
p1 = 0.9  # Percentage of small jobs
p2 = 0.984  # (p2 - p1) = Percentage of medium jobs     AND     (1 - p2) = Percentage of large jobs
servers = 10         # Number of servers
simulation_time = 107100.0  # Simulation time (hours)

Q = queue.Queue()
servers_busy = np.full(servers, np.inf)  # Servers are initially idle
arrival_counts = np.zeros((servers, int(simulation_time)*60))  # Track arrivals at each server at each minute
queue_times = []  # Track queue times for each customer
service_times = []  # Track service times for each customer
inter_arrival_times = []  # Track inter-arrival times for each customer

# Initialize simulation
arrival_time = expon.rvs(scale=1/arrival_rate)  # First arrival time
count = 0
previous_check_time_minute = 0

# Run the simulation
while min(arrival_time, np.min(servers_busy)) < simulation_time:

    current_time_minutes = int(np.floor(min(arrival_time, np.min(servers_busy))*60))
    for i in range(servers):
        if servers_busy[i] == np.inf:
            arrival_counts[i, previous_check_time_minute:current_time_minutes] = 0
        else:
            arrival_counts[i, previous_check_time_minute:current_time_minutes] = 1

    previous_check_time_minute = current_time_minutes  # Update the previous check time

    # Checking if the next event is an arrival or a service
    if arrival_time < np.min(servers_busy):
        # New customer arrives
        idle_server = np.where(servers_busy == np.inf)[0]
        if idle_server.size > 0:
            # There is an idle server, start serving immediately
            temp = np.random.rand()
            if temp < p1:
                next_service_time = expon.rvs(scale=1/service_rate1)  # Service time with rate 1
            elif p1 <= temp < p2:
                next_service_time = expon.rvs(scale=1/service_rate2)  # Service time with rate 2
            else:
                next_service_time = expon.rvs(scale=1/service_rate3)  # Service time with rate 3
            service_times.append(next_service_time)
            servers_busy[idle_server[0]] = arrival_time + next_service_time
        else:
            # All servers are busy, customer waits in queue
            Q.put(arrival_time)
        # Scheduling next arrival
        next_arrival_time = expon.rvs(scale=1/arrival_rate)
        inter_arrival_times.append(next_arrival_time)
        arrival_time += next_arrival_time
    else:
        # Server finishes serving customer
        server_index = np.argmin(servers_busy)
        if not Q.empty():
            # Start serving next customer in queue
            arrival_of_current = Q.get()
            queue_times.append(servers_busy[server_index] - arrival_of_current)  # Track queue time for this customer
            temp = np.random.rand()
            if temp < p1:
                next_service_time = expon.rvs(scale=1/service_rate1)  # Service time with rate 1
            elif p1 <= temp < p2:
                next_service_time = expon.rvs(scale=1/service_rate2)  # Service time with rate 2
            else:
                next_service_time = expon.rvs(scale=1/service_rate3)  # Service time with rate 3
            service_times.append(next_service_time)
            servers_busy[server_index] = servers_busy[server_index] + next_service_time
        else:
            # No customers in queue, server becomes idle
            servers_busy[server_index] = np.inf


print(f"\nTotal ticket number: {len(service_times)}")
print(f"\nTotal simulation time: {simulation_time} hours")

mean_service_time = np.mean(service_times)
print(f"\nMean service time: {mean_service_time} hours")

mean_queue_time = np.mean(queue_times)
print(f"\nMean queue time: {mean_queue_time} hours")

variance_service_times = np.var(service_times)
#print(f"\nVariance of service times: {variance_service_times}")

CV_service_times = variance_service_times / (mean_service_time * mean_service_time)
print(f"\nCoefficient of variation of service times: {CV_service_times}")

load = arrival_rate * mean_service_time  # ρ = λ * E[S]
print(f"\nLoad ρ of the system: {load}")
print(f"\nLoad ρ of each server in the system: {load/servers}")  # ρ = λ_server * E[S]


# Plot arrival counts
fig, axs = plt.subplots(5, 2, figsize=(10, 15))
axs = axs.ravel()
for i in range(servers):
    axs[i].plot(range(int(simulation_time)*60), arrival_counts[i], color='blue')
    axs[i].set_title('Server {}'.format(i+1))
    axs[i].set_xlabel('Time (minutes)')
    axs[i].set_ylabel('Arrivals')
plt.tight_layout(pad=3.0)


# Plotting inter-arrival times
x = np.linspace(0, max(inter_arrival_times), 1000)  # Generate a new range of x values
y = arrival_rate * np.exp(-arrival_rate * x)  # Calculate the corresponding y values
plt.figure()
plt.hist(inter_arrival_times, bins=200, color='green', alpha=0.7, density=True)
plt.plot(x, y, 'r-', lw=1)  # Plot the exponential distribution
plt.title('Inter-Arrival Time Distribution')
plt.xlabel('Inter-Arrival Time (minutes)')
plt.ylabel('Probability')


# Plotting service times for the whole system
x = np.linspace(0, max(service_times), 1000)  # Generate a range of x values
y1 = expon.pdf(x, scale=1/service_rate1)
y2 = expon.pdf(x, scale=1/service_rate2)
y3 = expon.pdf(x, scale=1/service_rate3)
y = p1*y1 + (p2-p1)*y2 + (1-p2)*y3
plt.figure()
plt.hist(service_times, bins=200, color='blue', alpha=0.7, density=True)
plt.plot(x, y, 'r-', lw=1)  # Plot the normal distribution
plt.title('Service Time Distribution')
plt.xlabel('Service Time (minutes)')
plt.ylabel('Probability')

plt.show()

