import matplotlib.pyplot as plt
import ABM as abm
import csv
import numpy as np
import scienceplots

plt.style.use('science')
plt.rcParams.update({'font.size': 8})


def write_file(data, path, name, rounds):
    with open(path+name, "w", newline="") as file:
        writer = csv.writer(file, dialect="excel-tab")
        r = [i for i in range(rounds+1)]
        for metadata, dat in data.items():
            writer.writerow(metadata)
            writer.writerow(r)
            writer.writerows(dat)
            writer.writerow("\n")


def write_file_data(data, path, name):
    with (open(path+name, "w", newline="") as file):
        writer = csv.writer(file, dialect="excel-tab")
        for metadata, dat in data.items():
            results, errors = dat
            writer.writerow(metadata)
            writer.writerow(results)
            writer.writerow(errors)
            writer.writerow("\n")


def write_file_gist(data, path, name, rounds, runs):
    with open(path+name, "w", newline="") as file:
        writer = csv.writer(file, dialect="excel-tab")
        for metadata, dat in data.items():
            writer.writerow(metadata)
            data_point = [np.count_nonzero(dat[:, rounds] == 1) / runs, 1.96 * np.std(dat[:, rounds]) / np.sqrt(runs)]
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


def plot_confidence(data, rounds, runs, path, name):
    x = [i for i in range(rounds + 1)]
    for t, result in data.items():
        y, ymin, ymax = list(), list(), list()
        for i in range(0, rounds + 1):
            l = result[:, i].tolist()
            mean = l.count(1) / runs
            y.append(mean)
            # ci = 1.96 * np.std(l)/np.sqrt(runs)
            ymin.append(mean - 1.96 * np.sqrt(mean * (1 - mean) / runs))
            ymax.append(mean + 1.96 * np.sqrt(mean * (1 - mean) / runs))
        plt.style.use('science')
        plt.plot(x, y, label=t[0])
        plt.fill_between(x, ymin, ymax, alpha=.3)
    plt.xlabel("rounds")
    plt.ylabel("convergence")
    plt.title(name[:-4])
    plt.legend()
    plt.savefig(path + name, dpi=250)
    plt.clf()


def plot_errorbars(x_axis, data, path, name, xl, ticks):
    for t, d in data.items():
        results, errors = d
        plt.errorbar(x_axis, results, yerr=errors, fmt=".-", label=t, capsize=3)
        plt.ylim(0, 1.05)
    if ticks:
        plt.xticks(x_axis)
    plt.xlabel(xl)
    plt.ylabel("Convergence")
    plt.title(name[:-4])
    plt.legend(loc="lower right", fontsize="small")
    plt.savefig(path + name, dpi=250)
    plt.clf()


def end_convergence(data, rounds, runs):
    for metadata, dat in data.items():
        return np.count_nonzero(dat[:, rounds] == 1) / runs


# def error(data, rounds, runs):
#     for metadata, dat in data.items():
#         return 1.96 * np.std(dat[:, rounds]) / np.sqrt(runs)

def error(data, rounds, runs):
    for metadata, dat in data.items():
        mean = np.count_nonzero(dat[:, rounds] == 1) / runs
        return 1.96 * np.sqrt(mean * (1 - mean) / runs)


# Parameters should be given in the format: type, data sharing, nr_agents, pulls, t1, t2, distance, N, nr_good, kind, prior, epsilon, threshold
# Example run below

pulls = 1000
t1 = 0.501
t2 = 0.5
n = pulls / 2
rounds = 10000
runs = 1000

path = "final_results/rational_intertia/new-algorithm/"
data = dict()
for tip in ["Doubt"]:
    data[tip] = list()
    results = list()
    errors = list()
    agents = 10
    for threshold in [20, 30, 40, 50, 60, 70, 80]:
        parameters = [[tip], [True], [agents], [pulls], [t1], [t2], [None], [500], [None], ["Cautious"], [250], [(False, 0.005)], [threshold]]
        d = abm.space(runs, rounds, parameters)
        write_file(d, path, f"{tip}, {agents}, {threshold}, new algorithm.csv", rounds)
        plot_confidence(d, rounds, runs, path, f"{tip}, {agents}, {threshold}, new algorithm.png")
        results.append(end_convergence(d, rounds, runs))
        errors.append(error(d, rounds, runs))
    data[tip].append(results)
    data[tip].append(errors)
write_file_data(data, path, f"Inertia, doubt, new algorithm.csv")
plot_errorbars([20, 30, 40, 50, 60, 70, 80], data, path, f"Inertia, doubt, new algorithm.png", "Threshold", True)

