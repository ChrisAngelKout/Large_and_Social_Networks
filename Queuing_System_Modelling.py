import csv
import datetime
import ast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections as coll
from functions_for_HTML import business_hours, process_csv


# Used to trim the csv file
# process_csv("data_9servers.csv", "data_trimmed_2.csv")

filename = 'data_trimmed_2.csv'
servers = 10

with open(filename, 'r') as f:
    reader = csv.reader(f)

    # Get the header row (if any)
    headers = next(reader, None)
    count = 0
    # Loop through the rest of the rows
    for row in reader:
        ticket_num = row[0]
        server = row[1]
        day_created = row[2]
        created_date = datetime.datetime.strptime(row[3], '%m/%d/%Y %H:%M')
        day_resolved = row[4]
        resolved_date = datetime.datetime.strptime(row[5], '%m/%d/%Y %H:%M')
        tuple_str = row[6]
        tuple_int = ast.literal_eval(tuple_str)  # This will convert string tuple to actual tuple
        business_hours_diff = tuple_int[0]
        business_minutes_diff = tuple_int[1]
        if business_hours_diff < 180:
            count += 1
        else:
            print(ticket_num)
        # Code to print all csv data row by row
        '''
        print(f'Ticket Number: {ticket_num}')
        print(f'Server: {server}')
        print(f'Day Created: {day_created}')
        print(f'Created Date: {created_date}')
        print(f'Day Resolved: {day_resolved}')
        print(f'Resolved Date: {resolved_date}')
        print(f'Business Hours and Minutes Difference: {business_hours_diff} hours, {business_minutes_diff} minutes')
        print()'''

    print(f'\n{count}')


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --- General Mean Service time PER SERVER

print("---------------------------------------------------------------------------------------------------------------")
print("General data, Means without any queueing taken into consideration\n\n")

with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row if it exists

    # Initializing counters
    total_server_service_time = np.zeros(servers)
    ticket_count = np.zeros(servers)
    service_times = []
    inter_arrival_times = []

    # Initialize variables to calculate mean arrival rate
    first_request_time = None
    last_request_time = None
    total_requests = 0

    # Loop through each row in the csv
    for row in reader:
        server_num = int(row[1])  # Getting the server number
        tuple_str = row[6]  # Get the business hours between arrival and completion
        tuple_int = ast.literal_eval(tuple_str)
        service_time = tuple_int[0]*60 + tuple_int[1]
        arrival_time = datetime.datetime.strptime(row[3], '%m/%d/%Y %H:%M')

        total_server_service_time[server_num - 1] += service_time
        service_times.append(service_time)

        ticket_count[server_num - 1] += 1

        # Calculate inter-arrival time and store it
        if last_request_time is not None:
            temp = business_hours(last_request_time, arrival_time)
            inter_arrival_time = temp[0] * 60 + temp[1]
            inter_arrival_times.append(inter_arrival_time)

        # Update first and last request times and total requests for arrival rate calculation
        if first_request_time is None or arrival_time < first_request_time:
            first_request_time = arrival_time
        if last_request_time is None or arrival_time > last_request_time:
            last_request_time = arrival_time
        total_requests += 1

    # Calculate mean arrival rate
    final_time_with_vacation = last_request_time - datetime.timedelta(days=110)  # Subtract 20 days for vacation for every year (2018-2022) + 10 days for half of 2023
    total_business_hours = business_hours(first_request_time, final_time_with_vacation)[0]
    mean_arrival_rate = total_requests / total_business_hours


for i in range(servers):
    # Calculate mean service time
    mean_server_service_time = divmod(total_server_service_time[i] / ticket_count[i], 60)
    print(f"Mean Service Time of server {i+1}: {int(mean_server_service_time[0])} hours, {int(mean_server_service_time[1])} minutes")

total_tickets = 0
for i in range(len(ticket_count)):
    total_tickets += ticket_count[i]

print(f"\nTotal ticket number: {int(total_requests)}")
print(f"\nTotal time of the servers running: {total_business_hours} hours")

service_times_array = np.array(service_times)
service_times_var = np.var(service_times_array/60)
service_times_mean = np.mean((service_times_array/60))

print(f"\nMean service time: {service_times_mean} hours")

CV_service_times = service_times_var / (service_times_mean * service_times_mean)
print(f'\nThe Coefficient of Variation for Service Time Distribution: {CV_service_times}')

print(f"\nMean Arrival Rate: {mean_arrival_rate} requests per hour\n")


lamda = 1/np.mean(inter_arrival_times)  # λ for exp(λ) arrival rates

x = np.linspace(0, max(inter_arrival_times), 1000)  # Generate a new range of x values
y = lamda * np.exp(-lamda * x)  # Calculate the corresponding y values
# Plotting inter-arrival times
plt.figure()
plt.hist(inter_arrival_times, bins=200, color='green', alpha=0.7, density=True)
plt.plot(x, y, 'r-', lw=0.5)  # Plot the exponential distribution
plt.title('Inter-Arrival Time Distribution')
plt.xlabel('Inter-Arrival Time (minutes)')
plt.ylabel('Probability')


mew = 1/(service_times_mean*10)  # μ for exp(μ) service rates

x = np.linspace(0, max(service_times), 1000)  # Generate a range of x values
y = mew * np.exp(-mew/2 * x)  # Calculate the corresponding y values
# Plotting service times for the whole system
plt.figure()
plt.hist(service_times, bins=250, color='blue', alpha=0.7, density=True)
plt.plot(x, y, 'r-', lw=0.5)  # Plot the exponential distribution
plt.title('Service Time Distribution')
plt.xlabel('Service Time (minutes)')
plt.ylabel('Probability')
plt.show()


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --- Mean Service time AND Mean Queue time AS FCFS

print("---------------------------------------------------------------------------------------------------------------")
print("Mean Service time AND Mean Queue time AS FIFO\n")

queue_times = []  # list to store queueing times for each request
actual_service_times = []  # list to store actual service times for each request


with open(filename, 'r') as file:
    reader = csv.reader(file)
    next(reader)

    # Initializing counters
    total_service_time = np.zeros(servers)
    total_queueing_time = np.zeros(servers)
    ticket_count = np.zeros(servers)
    server_queues = [coll.deque() for _ in range(servers)]
    server_service_end_times = [datetime.datetime.min for _ in range(servers)]  # When the last request was finished by the server

    # Loop through each row in the csv
    for row in reader:
        server = int(row[1])  # Getting the server number
        tuple_str = row[6]  # Get the business hours between arrival and completion
        tuple_int = ast.literal_eval(tuple_str)
        service_time = tuple_int[0]*60 + tuple_int[1]
        arrival_time = datetime.datetime.strptime(row[3], '%m/%d/%Y %H:%M')

        # If there are requests in the queue or the request arrives before the last request was finished, then this request has to wait
        if server_service_end_times[server - 1] > arrival_time:
            queueing_time = server_service_end_times[server - 1] - arrival_time
            total_queueing_time[server - 1] += queueing_time.total_seconds() / 60  # Convert the queuing time to minutes
            queue_times.append(queueing_time.total_seconds() / 60)  # Store the queueing time
        else:
            queueing_time = datetime.timedelta(seconds=0)
            queue_times.append(0)  # Store zero queueing time if no queue

        # Calculate the time when the service for this request will end
        server_service_end_times[server - 1] = max(server_service_end_times[server - 1], arrival_time) + datetime.timedelta(minutes=service_time)

        # Add the request to the queue of the server
        server_queues[server - 1].append(service_time)

        # Total service, excluding queueing time
        actual_service_time = service_time - queueing_time.total_seconds() / 60
        total_service_time[server - 1] += max(0, actual_service_time)  # Ensures service time is not negative
        actual_service_times.append(max(0, actual_service_time))  # Store the actual service time

        ticket_count[server - 1] += 1

    # General mean service and queueing time counters
    mean_service_time_total = 0.0
    mean_queueing_time_total = 0.0

    for i in range(servers):
        # Calculate mean service time and mean queueing time
        mean_service_time = divmod(total_service_time[i] / ticket_count[i], 60)
        mean_queueing_time = divmod(total_queueing_time[i] / ticket_count[i], 60)
        print()
        print(f"Server {i+1} - Mean Service Time: {int(mean_service_time[0])} hours, {int(mean_service_time[1])} minutes")
        print(f"Server {i+1} - Mean Queueing Time: {int(mean_queueing_time[0])} hours, {int(mean_queueing_time[1])} minutes")

        mean_service_time_total += total_service_time[i]
        mean_queueing_time_total += total_queueing_time[i]

    # General mean service and queueing time
    mean_service_time_total = divmod(mean_service_time_total / total_requests, 60)
    mean_queueing_time_total = divmod(mean_queueing_time_total / total_requests, 60)

    print("\n")
    print(f"\nTotal ticket number: {int(total_tickets)}")
    print(f"\nTotal time of the servers running: {total_business_hours} hours")
    print(f"\nTotal Mean Service Time: {int(mean_service_time_total[0])} hours, {int(mean_service_time_total[1])} minutes")
    print(f"\nTotal Mean Queueing Time: {int(mean_queueing_time_total[0])} hours, {int(mean_queueing_time_total[1])} minutes")


# Calculate Coefficient of Variation for service times
CV_service_times = np.var(actual_service_times) / (np.mean(actual_service_times) * np.mean(actual_service_times))
print('\nCoefficient of Variation for Service Time Distribution: ', CV_service_times)

print(f"\nMean Arrival Rate: {mean_arrival_rate} requests per hour\n")


lamda = 1/np.mean(inter_arrival_times)  # λ for exp(λ) arrival rates

x = np.linspace(0, max(inter_arrival_times), 1000)  # Generate a new range of x values
y = lamda * np.exp(-lamda * x)  # Calculate the corresponding y values
# Plotting inter-arrival times
plt.figure()
plt.hist(inter_arrival_times, bins=200, color='green', alpha=0.7, density=True)
plt.plot(x, y, 'r-', lw=0.5)  # Plot the exponential distribution
plt.title('Inter-Arrival Time Distribution')
plt.xlabel('Inter-Arrival Time (minutes)')
plt.ylabel('Probability')


mew = 1 / np.mean(actual_service_times)  # μ for exp(μ) service rates

x = np.linspace(min(actual_service_times), max(actual_service_times), 1000)  # Generate a range of x values
y = mew * np.exp(-mew * x)  # Calculate the corresponding y values

# Plot the actual service times as a power density function
plt.figure()
weights = np.ones_like(actual_service_times) / len(actual_service_times)
plt.hist(actual_service_times, weights=weights, bins=100, color='blue', alpha=0.7, density=True)
plt.plot(x, y, 'r-', lw=0.5)  # Plot the exponential distribution
plt.title('Service Time Distribution, with Queueing')
plt.xlabel('Actual Service Time (minutes)')
plt.ylabel('Probability')


load = mean_arrival_rate * (np.mean(actual_service_times))/60
print(f"Load ρ of the system: {load}")  # ρ = λ * E[S]
print(f"Load ρ of each server in the system: {load/servers}")  # ρ = λ_server * E[S]

plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------------------------

# Reading the CSV file
df = pd.read_csv(filename, header=None)

# Convert the date time column
df[3] = pd.to_datetime(df[3], format='%m/%d/%Y %H:%M')

# Timestamps to hours and minutes of the day
df['Hours_of_day'] = df[3].dt.hour + df[3].dt.minute / 60 + df[3].dt.second / 3600
df['Minutes_of_day'] = df[3].dt.hour * 60 + df[3].dt.minute + df[3].dt.second / 60

# Add a new column for the month
df['Month'] = df[3].dt.month

# Time interval (in hours and minutes)
hour_interval = 1  # One hour
minute_interval = 1  # One minute

# Bins for the histograms
hour_bins = np.arange(0, 24 + hour_interval, hour_interval)
minute_bins = np.arange(0, 24 * 60 + minute_interval, minute_interval)

# Arrivals per hour and per minute for the whole dataset
hour_arrival_counts, hour_bin_edges = np.histogram(df['Hours_of_day'], bins=hour_bins)
minute_arrival_counts, minute_bin_edges = np.histogram(df['Minutes_of_day'], bins=minute_bins)

# Arrival rates
hour_arrival_rates = hour_arrival_counts / hour_interval
minute_arrival_rates = minute_arrival_counts / minute_interval

# Plot arrival rates for the whole dataset
plt.figure()
plt.plot(hour_bin_edges[:-1], hour_arrival_rates)
plt.xlabel('Hour of the day')
plt.ylabel('Arrival rate (arrivals per hour)')
plt.title('Arrival Rate per Hour of Day')
plt.grid(True)

plt.figure()
plt.plot(minute_bin_edges[:-1] / 60, minute_arrival_rates)
plt.xlabel('Hour of the day')
plt.ylabel('Arrival rate (arrivals per minute)')
plt.title('Arrival Rate per Minute of Day')
plt.grid(True)

# Subplots
fig1, axs1 = plt.subplots(4, 3, figsize=(15, 20))  # For Arrivals per hour for each server
axs1 = axs1.flatten()
fig2, axs2 = plt.subplots(4, 3, figsize=(15, 20))  # For Arrivals per minute for each server
axs2 = axs2.flatten()
fig3, axs3 = plt.subplots(4, 3, figsize=(15, 15))  # For Arrivals per hour for each month of the year
axs3 = axs3.ravel()

# Plotting Arrival rates per hour and per minute for each server
for i in range(servers):
    server_df = df[df[1] == i + 1]

    # Number of Arrivals in each interval
    server_hour_arrival_counts, server_hour_bin_edges = np.histogram(server_df['Hours_of_day'], bins=hour_bins)
    server_minute_arrival_counts, server_minute_bin_edges = np.histogram(server_df['Minutes_of_day'], bins=minute_bins)

    # Arrival Rates (arrivals per hour or per minute)
    server_hour_arrival_rates = server_hour_arrival_counts / hour_interval
    server_minute_arrival_rates = server_minute_arrival_counts / minute_interval

    # Plotting Arrival Rates
    axs1[i].plot(server_hour_bin_edges[:-1], server_hour_arrival_rates)
    axs1[i].set_xlabel('Hour of the day')
    axs1[i].set_ylabel('Arrival rate (arrivals per hour)')
    axs1[i].set_title(f'Arrival Rate per Hour of Day for Server {i + 1}')
    axs1[i].grid(True)

    axs2[i].plot(server_minute_bin_edges[:-1] / 60, server_minute_arrival_rates)  # Divide bin_edges by 60 to convert back to hours for plotting
    axs2[i].set_xlabel('Hour of the day')
    axs2[i].set_ylabel('Arrival rate (arrivals per minute)')
    axs2[i].set_title(f'Arrival Rate per Minute of Day for Server {i + 1}')
    axs2[i].grid(True)


# Calculating and plotting arrival rates for each month of the year
for i, month in enumerate(range(1, 13)):
    # Get data for this month
    month_df = df[df['Month'] == month]

    # Number of arrivals in each interval
    month_arrival_counts, month_bin_edges = np.histogram(month_df['Minutes_of_day'], bins=minute_bins)

    # Arrival rate (arrivals per minute)
    month_arrival_rates = month_arrival_counts / minute_interval

    # Plot this month's arrival rates
    axs3[i].plot(month_bin_edges[:-1] / 60,
                 month_arrival_rates)  # Divide bin_edges by 60 to convert back to hours for plotting
    axs3[i].set_xlabel('Hour of the day')
    axs3[i].set_ylabel('Arrival rate (arrivals per minute)')
    axs3[i].set_title(f'Arrival Rate per Minute of Day for Month {month}')
    axs3[i].grid(True)


# Requests per day
requests_per_day = df[2].value_counts()

# Plotting the arrival distribution
plt.figure()
requests_per_day.plot(kind='bar')
plt.xlabel('Day of the week')
plt.ylabel('Number of requests')
plt.title('Distribution of requests per day')
plt.grid(True)

fig1.tight_layout(pad=6.0)
fig2.tight_layout(pad=6.0)
fig3.tight_layout(pad=3.0)

# Show all plots
#plt.show()

