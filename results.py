import matplotlib.pyplot as plt
import ABM as abm
import csv
import numpy as np


def write_file(data, path, name, rounds):
    with open(path+name, "w", newline="") as file:
        writer = csv.writer(file)
        r = [i for i in range(rounds+1)]
        for metadata, dat in data.items():
            writer.writerow(metadata)
            writer.writerow(r)
            for elem in dat:
                writer.writerow(elem)
            writer.writerow("\n")


def plot_from_data(data, rounds, runs, path, name):
    x = [i for i in range(rounds+1)]
    for t, result in data.items():
        y = list()
        for i in range(0, rounds + 1):
            y.append(np.sum((result[:, i]) / runs))
        plt.plot(x, y, label=t)
        plt.ylim(0, 1)
    plt.xlabel("rounds")
    plt.ylabel("results")
    plt.title(name[:-4])
    plt.legend()
    plt.savefig(path + name)
    plt.show()


# Parameters should be given in the format: type, data sharing, nr_agents, pulls, t1, t2, distance, N

parameters = [["Doubt_halving", "Steadfast", "Doubt_N"], [True], [10], [10], [0.501], [0.5], [None], [5]]
rounds = 1000
runs = 10000

d = abm.space(runs, rounds, parameters)

write_file(d, "results/", "doubt_halve_steadfast_N_10pulls2.csv", rounds)
plot_from_data(d, rounds, runs, "results/", "Doubt halving, N, steadfast 2.png")
