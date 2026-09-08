import numpy as np
import matplotlib.pyplot as plt

data = [0.5, 1.2, 0.7, 1.6]


def exponential_pdf(x, lambda_param):
    return lambda_param * np.exp(-1 * lambda_param * x)


def log_likelihood(data, lambda_param):
    llh = 0
    for x in data:
        llh += np.log(exponential_pdf(x, lambda_param))
    return llh


def gradient_descent(
    data, initial_lambda=0.2723, lr=0.1, max_iterations=1000, tolerance=0.01
):
    lambda_param = initial_lambda
    lambda_history = []
    llh_history = []
    lambda_history.append(lambda_param)
    llh = log_likelihood(data, lambda_param)
    llh_history.append(llh)
    data_size = len(data)
    data_sum = np.sum(data)

    for episode in range(max_iterations):
        update = ((data_size / lambda_param) - data_sum) * lr
        lambda_param += update
        lambda_history.append(lambda_param)
        llh = log_likelihood(data, lambda_param)
        llh_history.append(llh)
        if abs(update) < tolerance:
            break

    return lambda_param, lambda_history, llh_history


estimate, lambda_history, llh_history = gradient_descent(data)

print(f"Lambda estimate after {len(lambda_history)} iterations:", estimate)
print("Lambda history:", end="")
for l in lambda_history[:-1]:
    print(l, end=", ")
print(lambda_history[-1])
print("Log-Likelihood history: ", end="")
for llh in llh_history[:-1]:
    print(llh, end=", ")
print(llh_history[-1])

plot, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(lambda_history)
axes[0].set_title("Lambda History")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Lambda Value")
axes[1].plot(llh_history)
axes[1].set_title("Log-Likelihood History")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Log-Likelihood Value")

plt.show()
