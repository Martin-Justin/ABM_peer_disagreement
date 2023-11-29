import matplotlib.pyplot as plt
import ABM as abm
import csv
import numpy as np


def write_file(data, path, name, rounds):
    with open(path+name, "w", newline="") as file:
        writer = csv.writer(file, dialect="excel-tab")
        r = [i for i in range(rounds+1)]
        for metadata, dat in data.items():
            writer.writerow(metadata)
            writer.writerow(r)
            writer.writerows(dat)
            writer.writerow("\n")


def write_file_gist(data, path, name, rounds, runs):
    with open(path+name, "w", newline="") as file:
        writer = csv.writer(file, dialect="excel-tab")
        for metadata, dat in data.items():
            writer.writerow(metadata)
            data_point = [np.count_nonzero(dat[:, rounds] == 1) / runs]
            # data_point = [np.sum((dat[:, rounds]) / runs)]
            writer.writerow(data_point)
            writer.writerow("\n")


def write_file_average_run(data, path, name, rounds, runs):
    with open(path+name, "w", newline="") as file:
        writer = csv.writer(file, dialect="excel-tab")
        r = [i for i in range(rounds + 1)]

        for metadata, dat in data.items():
            writer.writerow(metadata)
            writer.writerow(r)
            l = list()
            for i in range(0, rounds + 1):
                l.append(np.count_nonzero(dat[:, i] == 1) / runs)
                # data_point = [np.sum((dat[:, rounds]) / runs)]
            writer.writerow(l)
            writer.writerow("\n")


def plot_from_data(data, rounds, runs, path, name):
    x = [i for i in range(rounds+1)]
    for t, result in data.items():
        y = list()
        for i in range(0, rounds + 1):
            epsilon = np.sum((result[:, i]) / runs)
            y.append(epsilon)
        plt.plot(x, y, label=t[0])
        plt.ylim(0, 0.01)
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
        plt.plot(x, y1, label=t[0])
        plt.ylim(0, 1)
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

pulls = 1000
n = pulls / 2
rounds = 10000
runs = 1000
path = "results/conference_priors/"

for t in ["Normal", "Greedy", "Cautious"]:
    parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [10], [pulls], [0.501], [0.5], [None], [n], [None], [t], [250], [(False, 0.2)], [10]]
    d = abm.space(runs, rounds, parameters)
    write_file(d, path, f"{t}_10_{pulls}_{n}_all_prior_250.csv", rounds)
    write_file_average_run(d, path, f"{t}_10_{pulls}_{n}_all_prior_250_average.csv", rounds, runs)
    write_file_gist(d, path, f"{t}_10_{pulls}_{n}_all_prior_250_gist.csv", rounds, runs)
    alt_plot(d, rounds, runs, path, f"{t}_10_{pulls}_{n}_all_prior_250.png")

# n = 2000
# for agents in range(10, 11):
#     parameters = [["Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [None], ["Normal"], [4], [(True, 10)], [5]]
#     d = abm.space(runs, rounds, parameters)
#     write_file_average_run(d, path, f"{agents}_{pulls}_{n}_basic_extraboost_average.csv", rounds, runs)
#     write_file_gist(d, path, f"{agents}_{pulls}_{n}_basic_extraboost_gist.csv", rounds, runs)
#     alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_basic_extraboost.png")

# n = pulls / 2
# for agents in range(10, 11):
#     for nr_good in range(5, 6):
#         parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [nr_good], ["Normal"], [4], [(True, 10)], [5]]
#         d = abm.space(runs, rounds, parameters)
#         # write_file_average_run(d, path, f"{agents}_{pulls}_{n}_basic_with_{nr_good}_good_average.csv", rounds, runs)
#         # write_file_gist(d, path, f"{agents}_{pulls}_{n}_basic_setup_with_{nr_good}_good_gist.csv", rounds, runs)
#         write_file(d, path, f"{agents}_{pulls}_{n}_basic_setup_with_{nr_good}_good3.csv", rounds)
#         alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_basic_setup_with_{nr_good}_good3.png")

# for agents in range(10, 11):
#     for priors in [3000]:
#         parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [None], ["Normal"], [priors], [(True, 10)], [5]]
#         d = abm.space(runs, rounds, parameters)
#         write_file_average_run(d, path, f"{agents}_{pulls}_{n}_basic_prior{priors}_average.csv", rounds, runs)
#         write_file_gist(d, path, f"{agents}_{pulls}_{n}_basic_prior{priors}_gist.csv", rounds, runs)
#         alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_basic_prior{priors}.png")

# for agents in range(10, 11):
#     parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [None], ["Greedy"], [4], [(False, 0.2)], [5]]
#     d = abm.space(runs, rounds, parameters)
#     write_file_average_run(d, path, f"{agents}_{pulls}_{n}_greedy_stable_all_average.csv", rounds, runs)
#     write_file_gist(d, path, f"{agents}_{pulls}_{n}_greedy_stable_all_gist.csv", rounds, runs)
#     alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_greedy_stable_all.png")

# for agents in range(10, 11):
#     parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [None], ["Greedy"], [4], [(True, 10)], [5]]
#     d = abm.space(runs, rounds, parameters)
#     write_file_average_run(d, path, f"{agents}_{pulls}_{n}_greedy_changing_all_average.csv", rounds, runs)
#     write_file_gist(d, path, f"{agents}_{pulls}_{n}_greedy_changing_all_gist.csv", rounds, runs)
#     alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_greedy_changing_all.png")

# for agents in range(10, 11):
#     parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [None], ["Greedy"], [4], [(True, 1)], [5]]
#     d = abm.space(runs, rounds, parameters)
#     write_file_average_run(d, path, f"{agents}_{pulls}_{n}_greedy_changing_smallE_average.csv", rounds, runs)
#     write_file_gist(d, path, f"{agents}_{pulls}_{n}_greedy_changing_smallE_gist.csv", rounds, runs)
#     alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_greedy_changing_smallE.png")

# for agents in range(10, 11):
#     parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [None], ["Cautious"], [4], [(True, 1)], [5]]
#     d = abm.space(runs, rounds, parameters)
#     write_file_average_run(d, path, f"{agents}_{pulls}_{n}_cautious_all_average.csv", rounds, runs)
#     write_file_gist(d, path, f"{agents}_{pulls}_{n}_cautious_all_gist.csv", rounds, runs)
#     alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_cautious_all.png")

# for agents in range(10, 11):
#     parameters = [["Conciliate", "Steadfast", "Doubt", "Boost", "Mixed"], [True], [agents], [pulls], [0.501], [0.5], [None], [n], [None], ["Combined"], [4], [(True, 10)], [5]]
#     d = abm.space(runs, rounds, parameters)
#     write_file_average_run(d, path, f"{agents}_{pulls}_{n}_combined_all_average.csv", rounds, runs)
#     write_file_gist(d, path, f"{agents}_{pulls}_{n}_combined_all_gist.csv", rounds, runs)
#     alt_plot(d, rounds, runs, path, f"{agents}_{pulls}_{n}_combined_all.png")


# ["Conciliate", "Steadfast", "Doubt", "Boost", "Doubt_halving"]
