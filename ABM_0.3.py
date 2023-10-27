import numpy as np
from scipy.stats import beta
from itertools import product


class Round:
    def __init__(self):
        self.agents = list()
        self.round_results = list()

    def create_agents(self, nr_agents):   # Creates a list of agents, the number of agents is given as an argument
        for i in range(nr_agents):        # returns the ratio of agents supporting theory1 vs theory2
            agent = Agent(i)
            self.agents.append(agent)
            self.round_results.append(agent.my_theory)
        return self.round_results.count(True) / nr_agents

    def collect_data(self, theory1, theory2, pulls, sharing):     # Does two things based on the value of the parameter "sharing"
        if sharing:                                               # if sharing == True:
            alphas1, betas1, alphas2, betas2 = 0, 0, 0, 0         # -- collect new data from all agents
            for agent in self.agents:                             # -- tell agents to update based on this data
                alpha1, beta1, alpha2, beta2 = agent.experiment(pulls, theory1, theory2)
                alphas1 += alpha1
                betas1 += beta1
                alphas2 += alpha2
                betas2 += beta2
            for agent in self.agents:
                agent.update_distribution(alphas1, betas1, alphas2, betas2)
        else:                                                    # if sharing == "False":
            for agent in self.agents:                            # -- every agent does its own experiments ...
                agent.update(pulls, theory1, theory2)            # -- ... and updates only based on them

    def action(self, type, interval, n):    # Defines different ways in which agents can act every round
        if type == "Conciliate":     # Here, agents split the difference
            agg1 = sum([agent.mean1 for agent in self.agents]) / len(self.agents)   # The function calculate the average ...
            agg2 = sum([agent.mean2 for agent in self.agents]) / len(self.agents)   # ... of their credence for both theories
            for agent in self.agents:
                agent.conciliate(agg1, agg2)      # agents then take these averages as their new credences
                agent.update_theory()             # they also update which theory they work on

        if type == "Doubt_halving":    # Here, agents change their variance not credence
            for agent in self.agents:
                agent.update_theory()
                for agent2 in self.agents:
                    if agent.my_theory != agent2.my_theory:    # If agents find another agent who support a different theory
                        agent.doubt()                          # they change their variance in a way described below
                        break

        if type == "Doubt_N":         # Here, agents change their variance in a different way
            for agent in self.agents:
                agent.update_theory()
                n0 = 0
                for agent2 in self.agents:
                    if agent.name != agent2.name:          # Each agent count the number of agents who (dis)agree with them
                        n0 += (1 if agent.my_theory == agent2.my_theory else -1)    # This is represented by one number
                if n0:                                     # if this number is different from 0 ...
                    agent.doubt_alternative(n0, n)         # they recalculate the agents variance

        if type == "Conciliate_degree":     # Here, agents split the difference only with agents within certain interval
            credences1 = [agent.mean1 for agent in self.agents]    # The interval is given as a parameter of the model
            credences2 = [agent.mean2 for agent in self.agents]
            for agent in self.agents:
                interesting_credences1 = list()
                interesting_credences2 = list()
                for credence1 in credences1:
                    if abs(agent.mean1 - credence1) < interval:      # The agents check if the credence of an agent ...
                        interesting_credences1.append(credence1)     # ... is within the interval and collect it ...
                for credence2 in credences2:                         # ... if it is
                    if abs(agent.mean2 - credence2) < interval:
                        interesting_credences2.append(credence2)
                agg1 = sum(interesting_credences1) / len(interesting_credences1)
                agg2 = sum(interesting_credences2) / len(interesting_credences2)
                agent.conciliate(agg1, agg2)                        # then they take the average of the collected ...
                agent.update_theory()                               # ... credences as their new credence

        if type == "Steadfast":        # Here, agents work alone, so they just update the theory they support
            for agent in self.agents:
                agent.update_theory()

    def results(self):                # This function report results of the round in the form of a ration between ...
        self.round_results = list()   # ... the number of agents who support theory1 vs theory2
        for agent in self.agents:
            self.round_results.append(agent.my_theory)
        return self.round_results.count(True) / len(self.agents)


class Agent:
    def __init__(self, name):       # Each agent has alpha, beta, mean and variance for both theories
        self.name = name            # value my_theory tells which theory the agents thinks is better
        self.alpha1 = np.random.uniform(1, 4)
        self.beta1 = np.random.uniform(1, 4)
        self.alpha2 = np.random.uniform(1, 4)
        self.beta2 = np.random.uniform(1, 4)
        self.mean1, self.var1 = beta.stats(self.alpha1, self.beta1, moments="mv")
        self.mean2, self.var2 = beta.stats(self.alpha2, self.beta2, moments="mv")
        self.my_theory = self.mean1 > self.mean2  # so, True = theory1, False = theory2; theory1 is better by design

    def experiment(self, n, theory1, theory2):    # Agents take N pulls from the theory they support
        if self.my_theory:     # Agent takes n pulls from binomial distribution with mean = theory1
            successes = np.random.binomial(n, theory1)
            return successes, (n - successes), 0, 0
        else:                  # Agent takes n pulls from binomial distribution with mean = theory2
            successes = np.random.binomial(n, theory2)  # The function four values, two of them are zero, since ...
            return 0, 0, successes, (n - successes)     # ... the agents only work on one theory

    def update_distribution(self, alphas1, betas1, alphas2, betas2):    # Updating for agents who share data
        self.alpha1 += alphas1
        self.alpha2 += alphas2
        self.beta1 += betas1
        self.beta2 += betas2
        self.mean1, self.var1 = beta.stats(self.alpha1, self.beta1, moments="mv")
        self.mean2, self.var2 = beta.stats(self.alpha2, self.beta2, moments="mv")

    def update(self, n, theory1, theory2):      # Updating alpha, beta, mean, variance for agents who do not share data
        if self.my_theory:
            successes = np.random.binomial(n, theory1)
            self.alpha1 += successes
            self.beta1 += (n - successes)
        else:
            successes = np.random.binomial(n, theory2)
            self.alpha2 += successes
            self.beta2 += (n - successes)
        self.mean1, self.var1 = beta.stats(self.alpha1, self.beta1, moments="mv")
        self.mean2, self.var2 = beta.stats(self.alpha2, self.beta2, moments="mv")

    def update_theory(self):            # Updating value my_theory
        self.my_theory = self.mean1 > self.mean2

    def conciliate(self, agg_mean1, agg_mean2):    # Updating alpha and beta for conciliatory agents
        self.mean1 = agg_mean1                     # this two values represent averaged means of all agents ...
        self.mean2 = agg_mean2                     # ... for both theories

        self.alpha1 = ((1 - self.mean1) / self.var1 - 1 / self.mean1) * self.mean1 ** 2
        self.beta1 = self.alpha1 * (1 / self.mean1 - 1)

        self.alpha2 = ((1 - self.mean2) / self.var2 - 1 / self.mean2) * self.mean2 ** 2
        self.beta2 = self.alpha2 * (1 / self.mean2 - 1)

    def doubt(self):               # Changing variance for "doubtful" agents
        if self.my_theory:
            new_sum1 = (self.alpha1 + self.beta1) / 2       # Agents half the sum of alpha and beta
            self.alpha1 = self.mean1 * new_sum1             # mean stays the same
            self.beta1 = (1 - self.mean1) * new_sum1
        else:
            new_sum2 = (self.alpha2 + self.beta2) / 2
            self.alpha2 = self.mean2 * new_sum2
            self.beta2 = (1 - self.mean2) * new_sum2

    def doubt_alternative(self, nr_agreeing, n):       # Changing variance in a different way
        if self.alpha1 > n and self.beta1 > n:
            new_sum1 = (self.alpha1 + self.beta1) + nr_agreeing * n  # Agents add to the sum some number
            self.alpha1 = self.mean1 * new_sum1                      # this number depends on two things
            self.beta1 = (1 - self.mean1) * new_sum1                 # it is positive or negative depending on if ...

        if self.alpha2 > n and self.beta2 > n:                       # ... more agents agree of disagree with them
            new_sum2 = (self.alpha2 + self.beta2) + nr_agreeing * n  # its value also depends on N with is ...
            self.alpha2 = self.mean2 * new_sum2                      # ... a parameter of the model
            self.beta2 = (1 - self.mean2) * new_sum2


class Simulation:
    def __init__(self, runs, rounds):   # One simulation represents multiple runs of the model with the same parameters
        self.results = np.empty(rounds+1, dtype=float)
        self.rounds = rounds
        self.runs = runs

    def run(self, rounds, type, data_sharing, nr_agents, pulls, theory1, theory2, interval, n):  # Initiates one run of the model
        result = list()
        round = Round()
        result.append(round.create_agents(nr_agents))   # Collects results for the initial setup (before any action)

        for i in range(rounds):           # Initiates one round of the model
            round.collect_data(theory1, theory2, pulls, data_sharing)   # For each round, agents collect data
            round.action(type, interval, n)                             # agents do their actions
            result.append(round.results())                              # agents report results
        return result       # returs results for one run of the model, as a list

    def add_results(self, result):
        self.results = np.vstack((self.results, result))         # adds results in a 2d array (a table)

    def calculate_output(self):
        results_calculated = list()
        for i in range(0, self.rounds + 1):
            results_calculated.append(np.sum((self.results[1:, i]) / self.runs))     # Calculates the averages multiple runs

        return results_calculated


def space(runs, rounds, parameters, file_path, file_name):   # Parameters must be a list of lists with the right order of elements
    output = open(file_path+file_name, "w")
    combinations = list(product(*parameters))   # This calculates all the different combinations of the given parameters
    for combination in combinations:            # Runs simulation for every combination of parameters
        simulation = Simulation(runs, rounds)
        t, data_sharing, nr_agents, pulls, theory1, theory2, interval, n = combination
        for i in range(runs):
            simulation.add_results(simulation.run(rounds, t, data_sharing, nr_agents, pulls, theory1, theory2, interval, n))
        output.write(f"type: {t}, data sharing: {data_sharing}, agents: {nr_agents}, pulls: {pulls}, bandits: {theory1, theory2},  special: {interval, n}\n {simulation.calculate_output()}\n\n")
    output.flush()  # this function returns results as a .txt file with data for all simulations


parameters = [["Conciliate", "Steadfast"], [True], [10], [1000], [0.501], [0.5], [None], [None]]
space(100, 500, parameters, "refactored_code_test/", "test_random.txt")






# def run(type, data_sharing, nr_agents, rounds, pulls, theory1, theory2, interval, n):
#     result = list()
#     round = Round()
#     result.append(round.create_agents(nr_agents))
#
#     for i in range(rounds):
#         if data_sharing:
#             round.collect_data(theory1, theory2, pulls)
#         round.action(type, interval, n)
#         result.append(round.results())
#
#     return result
#
#
# def simulation(runs, file_path, file_name, type, data_sharing, nr_agents, rounds, pulls, theory1, theory2, interval, n):
#     results = np.empty(rounds+1, dtype=float)
#     output = open(file_path+file_name, "w")
#     for i in range(runs):
#         result = np.array(run(type, data_sharing, nr_agents, rounds, pulls, theory1, theory2, interval, n), dtype=float)
#         results = np.vstack((results, result))
#     results_calculated = list()
#     for i in range(0, rounds + 1):
#         results_calculated.append(np.sum(results[1:, i] / runs))
#     output.write(f"{type}, data sharing {data_sharing}, {nr_agents} agents, {pulls} pulls, {interval}, {n}: {results_calculated}\n\n")
#     output.flush()
#
#
# simulation(100, "refactored_code_test/", "test3_doubting.txt", "Doubt_halving", True, 10, 500, 1000, 0.501, 0.5, None, None)
#
