import matplotlib.pyplot as plt
import ABM as abm
import csv
import numpy as np
import cProfile


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
            epsilon = np.sum((result[:, i]) / runs)
            y.append(epsilon)
        plt.plot(x, y, label=t)
        # plt.ylim(0, 1)
    plt.xlabel("rounds")
    plt.ylabel("results")
    plt.title(name[:-4])
    plt.legend()
    plt.savefig(path + name)
    plt.clf()


def alt_plot(data, rounds, runs, path, name):
    x = [i for i in range(rounds + 1)]
    for t, result in data.items():
        y1, y2, y3= list(), list(), list()
        for i in range(0, rounds + 1):
            l = result[:, i].tolist()
            y1.append((l.count(1)) / runs)
            # y2.append(l.count(0))
            # y3.append(sum(1 for elem in l if elem != 1 and elem != 0))
        plt.plot(x, y1, label=t)
        # plt.ylim(0.3, 0.7)
    plt.xlabel("rounds")
    plt.ylabel("convergence")
    plt.title(name[:-4])
    plt.legend()
    plt.savefig(path + name)
    plt.clf()


def alt_plot_polarization(data, rounds, runs, path, name):
    x = [i for i in range(rounds + 1)]
    for t, result in data.items():
        y1, y2, y3= list(), list(), list()
        for i in range(0, rounds + 1):
            l = result[:, i].tolist()
            # y1.append(l.count(1) + l.count(0))
            # y2.append(l.count(0))
            y3.append(sum(1 for elem in l if elem != 1 and elem != 0) / runs)
        plt.plot(x, y3, label=t)
        # plt.ylim(0, runs)
    plt.xlabel("rounds")
    plt.ylabel("results")
    plt.title(name[:-4])
    plt.legend()
    plt.savefig(path + name)
    plt.clf()

# Parameters should be given in the format: type, data sharing, nr_agents, pulls, t1, t2, distance, N

for agents in range(10, 11):
    for pulls in [1000]:
        for theory_1 in [0.501]:
            N = pulls/2
            parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Doubt_halving"], [True], [agents], [pulls], [theory_1], [0.5], [None], [N], [None], ["Greedy"]]
            rounds = 1000
            runs = 100
            path = "test_results/"
            d = abm.space(runs, rounds, parameters)
            write_file(d, path, f"greddy_some_variations.csv", rounds)
            alt_plot(d, rounds, runs, path, f"greddy_some_variations.png")



