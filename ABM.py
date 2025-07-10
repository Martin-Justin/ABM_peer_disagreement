import numpy as np
from itertools import product


class Round:
    def __init__(self):
        self.agents = list()
        self.round_results_convergence = list()

    def create_agents(self, nr_agents, nr_good, kind, prior, epsilon, threshold):   # Creates a list of agents, the number of agents is given as an argument
        agent_type = {"Greedy": Greedy, "Normal": Agent, "Cautious": Cautious, "Combined": Combined}
        if nr_good or nr_good == 0:
            j = 0
            for i in range(nr_good):        # returns the ratio of agents supporting theory1 vs theory2
                agent = agent_type[kind](j, 1, prior, epsilon, threshold)
                self.agents.append(agent)
                self.round_results_convergence.append(agent.my_theory)
                j += 1
            for i in range(nr_agents - nr_good):
                agent = agent_type[kind](j, 2, prior, epsilon, threshold)
                self.agents.append(agent)
                self.round_results_convergence.append(agent.my_theory)
                j += 1
        else:
            for i in range(nr_agents):
                agent = agent_type[kind](i, 0, prior, epsilon, threshold)
                self.agents.append(agent)
                self.round_results_convergence.append(agent.my_theory)

        return self.round_results_convergence.count(True) / nr_agents

    def collect_data(self, theory1, theory2, pulls, sharing, nr_agents):     # Does two things based on the value of the parameter "sharing"
        if sharing:                                               # if sharing == True:
            alphas1, betas1, alphas2, betas2 = 0, 0, 0, 0         # -- collect new data from all agents
            for agent in self.agents:                             # -- tell agents to update based on this data
                alpha1, beta1, alpha2, beta2 = agent.experiment(pulls, theory1, theory2, nr_agents)
                alphas1 += alpha1
                betas1 += beta1
                alphas2 += alpha2
                betas2 += beta2
            for agent in self.agents:
                agent.update_distribution(alphas1, betas1, alphas2, betas2)
        else:                                                    # if sharing == "False":
            for agent in self.agents:                            # -- every agent does its own experiments ...
                agent.update(pulls, theory1, theory2, nr_agents)            # -- ... and updates only based on them

    def action(self, type, interval, n, pulls, nr_agents, constraint):    # Defines different ways in which agents can act every round
        if type == "Conciliate":     # Here, agents split the difference
            agg1 = sum([agent.mean1 for agent in self.agents]) / len(self.agents)   # The function calculate the average ...
            agg2 = sum([agent.mean2 for agent in self.agents]) / len(self.agents)   # ... of their credence for both theories
            for agent in self.agents:
                agent.conciliate(agg1, agg2)      # agents then take these averages as their new credences
                agent.update_theory(pulls, nr_agents)             # they also update which theory they work on

        if type == "Boost":    # Here, agents change their variance not credence
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)
                n0 = 0
                for agent2 in self.agents:
                    if agent.name != agent2.name:     # Each agent count the number of agents who (dis)agree with them
                        n0 += 1 if agent.my_theory == agent2.my_theory else 0
                agent.doubt_alternative(n0, n)         # they recalculate the agents variance

        if type == "Doubt":         # Here, agents change their variance in a different way
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)
                n0 = 0
                for agent2 in self.agents:
                    if agent.name != agent2.name:     # Each agent count the number of agents who (dis)agree with them
                        n0 -= 1 if agent.my_theory != agent2.my_theory else 0
                agent.doubt_alternative(n0, n)         # they recalculate the agents variance

        if type == "Doubt_distance_linear":         # Here, agents change their variance in a different way
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)
                distances1, distances2 = 0, 0
                for agent2 in self.agents:
                    if agent.name != agent2.name:
                        distances1 += abs(agent.mean1 - agent2.mean1)
                        distances2 += abs(agent.mean2 - agent2.mean2)
                agent.doubt_distance_linear(distances1, distances2, n)

        if type == "Doubt_distance_sigmoid":
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)
                for agent2 in self.agents:
                    if agent.name != agent2.name:
                        distance1 = abs(agent.mean1 - agent2.mean1)
                        distance2 = abs(agent.mean2 - agent2.mean2)
                        agent.doubt_distance_sigmoid(distance1, distance2, n)

        if type == "Doubt_fancy":
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)
                most_distant1 = 0
                most_distant2 = 0
                for agent2 in self.agents:
                    if agent.name != agent2.name:
                        if abs(agent.mean1 - agent2.mean1) > abs(agent.mean1 - most_distant1) or most_distant1 == 0:
                            most_distant1 = agent2.mean1
                        if abs(agent.mean2 - agent2.mean2) > abs(agent.mean2 - most_distant2) or most_distant2 == 0:
                            most_distant2 = agent2.mean2
                agent.doubt_fancy(most_distant1, most_distant2, constraint)


        if type == "Mixed":
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)
                n0 = 0
                for agent2 in self.agents:
                    if agent.name != agent2.name:
                        n0 += 1 if agent.my_theory == agent2.my_theory else -1
                agent.doubt_alternative(n0, n)

        if type == "Doubt_halving":
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)
                for agent2 in self.agents:
                    if agent.my_theory != agent2.my_theory:
                        agent.doubt(n)
                        break


        if type == "Conciliate_degree":     # Here, agents split the difference only with agents within certain interval
            credences1 = [agent.mean1 for agent in self.agents]    # The interval is given as a parameter of the model
            credences2 = [agent.mean2 for agent in self.agents]
            for agent in self.agents:
                interesting_credences1 = list()
                interesting_credences2 = list()
                for credence1 in credences1:
                    if abs(agent.mean1 - credence1) <= interval:      # The agents check if the credence of an agent ...
                        interesting_credences1.append(credence1)     # ... is within the interval and collect it ...
                for credence2 in credences2:                         # ... if it is
                    if abs(agent.mean2 - credence2) <= interval:
                        interesting_credences2.append(credence2)
                agg1 = sum(interesting_credences1) / len(interesting_credences1)
                agg2 = sum(interesting_credences2) / len(interesting_credences2)
                agent.conciliate(agg1, agg2)                        # then they take the average of the collected ...
                agent.update_theory(pulls, nr_agents)                               # ... credences as their new credence

        if type == "Steadfast":        # Here, agents work alone, so they just update the theory they support
            for agent in self.agents:
                agent.update_theory(pulls, nr_agents)

    def results(self):                # This function report results of the round in the form of a ration between ...
        self.round_results_convergence = list()   # ... the number of agents who support theory1 vs theory2
        # variance = list()
        # sums = list()
        distance = list()
        # epsilons = list()
        for agent in self.agents:
            self.round_results_convergence.append(agent.my_theory)
            # distance.append(abs(agent.mean1 - agent.mean2))
            # epsilons.append(agent.epsilon)
            # sums.append(agent.alpha1 + agent.beta1 + agent.alpha2 + agent.alpha2)
            # variance.append(agent.var1)
        # return self.round_results
        return self.round_results_convergence.count(True) / len(self.agents)
        # return sum(distance) / len(self.agents)
        # return sum(variance) / len(self.agents)
        # return sum(epsilons) / len(self.agents)


class Agent:
    def __init__(self, name, theory, prior, epsilon, threshold):       # Each agent has alpha, beta, mean and variance for both theories
        self.name = name            # value my_theory tells which theory the agents thinks is better
        if theory == 1:
            self.alpha1 = np.random.uniform(1, prior)
            self.beta1 = 1
            self.alpha2 = 1
            self.beta2 = np.random.uniform(1, prior)
        elif theory == 2:
            self.alpha1 = 1
            self.beta1 = np.random.uniform(1, prior)
            self.alpha2 = np.random.uniform(1, prior)
            self.beta2 = 1
        else:
            self.alpha1 = np.random.uniform(1, prior)
            self.beta1 = np.random.uniform(1, prior)
            self.alpha2 = np.random.uniform(1, prior)
            self.beta2 = np.random.uniform(1, prior)
        self.mean1 = self.alpha1 / (self.alpha1 + self.beta1)
        self.mean2 = self.alpha2 / (self.alpha2 + self.beta2)
        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))
        self.my_theory = self.mean1 > self.mean2  # so, True = theory1, False = theory2; theory1 is better by design

    def experiment(self, pulls, theory1, theory2, nr_agents):    # Agents take N pulls from the theory they support
        if self.my_theory:     # Agent takes n pulls from binomial distribution with mean = theory1
            successes = np.random.binomial(pulls, theory1)
            return successes, (pulls - successes), 0, 0
        else:                  # Agent takes n pulls from binomial distribution with mean = theory2
            successes = np.random.binomial(pulls, theory2)  # The function four values, two of them are zero, since ...
            return 0, 0, successes, (pulls - successes)     # ... the agents only work on one theory

    def update_distribution(self, alphas1, betas1, alphas2, betas2):    # Updating for agents who share data
        self.alpha1 += alphas1
        self.alpha2 += alphas2
        self.beta1 += betas1
        self.beta2 += betas2
        self.mean1 = self.alpha1 / (self.alpha1 + self.beta1)
        self.mean2 = self.alpha2 / (self.alpha2 + self.beta2)
        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))


    def update(self, pulls, theory1, theory2, nr_agents):      # Updating alpha, beta, mean, variance for agents who do not share data
        if self.my_theory:
            successes = np.random.binomial(pulls, theory1)
            self.alpha1 += successes
            self.beta1 += (pulls - successes)
        else:
            successes = np.random.binomial(pulls, theory2)
            self.alpha2 += successes
            self.beta2 += (pulls - successes)
        self.mean1 = self.alpha1 / (self.alpha1 + self.beta1)
        self.mean2 = self.alpha2 / (self.alpha2 + self.beta2)
        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))

    def update_theory(self, pulls, nr_agents):            # Updating value my_theory
        self.my_theory = self.mean1 > self.mean2

    def conciliate(self, agg_mean1, agg_mean2):    # Updating alpha and beta for conciliatory agents
        self.mean1 = agg_mean1                     # this two values represent averaged means of all agents ...
        self.mean2 = agg_mean2                     # ... for both theories

        self.alpha1 = ((1 - self.mean1) / self.var1 - 1 / self.mean1) * self.mean1 ** 2
        self.beta1 = self.alpha1 * (1 / self.mean1 - 1)

        self.alpha2 = ((1 - self.mean2) / self.var2 - 1 / self.mean2) * self.mean2 ** 2
        self.beta2 = self.alpha2 * (1 / self.mean2 - 1)

    def doubt(self, n):   # Changing variance for "doubtful" agents
        factor = n / 250

        new_sum1 = (self.alpha1 + self.beta1) / factor   # Agents divides the sum of alpha and beta
        self.alpha1 = self.mean1 * new_sum1             # mean stays the same
        self.beta1 = (1 - self.mean1) * new_sum1

        new_sum2 = (self.alpha2 + self.beta2) / factor
        self.alpha2 = self.mean2 * new_sum2
        self.beta2 = (1 - self.mean2) * new_sum2

        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))

    def doubt_alternative(self, nr_agreeing, n):       # Changing variance in a different way
        update = nr_agreeing * n
        if (self.alpha1 + self.beta1) + update > 0:
            new_sum1 = (self.alpha1 + self.beta1) + nr_agreeing * n  # Agents add to the sum some number
            self.alpha1 = self.mean1 * new_sum1                      # this number depends on two things
            self.beta1 = (1 - self.mean1) * new_sum1                 # it is positive or negative depending on if ...
        else:
            new_sum1 = 1
            self.alpha1 = self.mean1 * new_sum1
            self.beta1 = (1 - self.mean1) * new_sum1

        if (self.alpha2 + self.beta2) + update > 0:                       # ... more agents agree of disagree with them
            new_sum2 = (self.alpha2 + self.beta2) + nr_agreeing * n     # its value also depends on N with is ...
            self.alpha2 = self.mean2 * new_sum2                         # ... a parameter of the model
            self.beta2 = (1 - self.mean2) * new_sum2
        else:
            new_sum2 = 1
            self.alpha2 = self.mean2 * new_sum2
            self.beta2 = (1 - self.mean2) * new_sum2

        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))

    def doubt_distance_linear(self, distance1, distance2, n):
        update1 = distance1 * n        # Changing variance in a different way
        if (self.alpha1 + self.beta1) - update1 > 0:
            new_sum1 = (self.alpha1 + self.beta1) - update1
            self.alpha1 = self.mean1 * new_sum1
            self.beta1 = (1 - self.mean1) * new_sum1
        else:
            new_sum1 = 250
            self.alpha1 = self.mean1 * new_sum1
            self.beta1 = (1 - self.mean1) * new_sum1

        update2 = distance2 * n
        if (self.alpha2 + self.beta2) - update2 > 0:
            new_sum2 = (self.alpha2 + self.beta2) - update2
            self.alpha2 = self.mean2 * new_sum2
            self.beta2 = (1 - self.mean2) * new_sum2
        else:
            new_sum2 = 250
            self.alpha2 = self.mean2 * new_sum2
            self.beta2 = (1 - self.mean2) * new_sum2

        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))

    def doubt_distance_sigmoid(self, distance1, distance2, n):
        update1 = n * (distance1 / (1 - distance1))       # Changing variance in a different way
        if (self.alpha1 + self.beta1) - update1 > 0:
            new_sum1 = (self.alpha1 + self.beta1) - update1
            self.alpha1 = self.mean1 * new_sum1
            self.beta1 = (1 - self.mean1) * new_sum1
        else:
            new_sum1 = 250
            self.alpha1 = self.mean1 * new_sum1
            self.beta1 = (1 - self.mean1) * new_sum1

        update2 = n * (distance2 / (1 - distance2))
        if (self.alpha2 + self.beta2) - update2 > 0:
            new_sum2 = (self.alpha2 + self.beta2) - update2
            self.alpha2 = self.mean2 * new_sum2
            self.beta2 = (1 - self.mean2) * new_sum2
        else:
            new_sum2 = 250
            self.alpha1 = self.mean1 * new_sum2
            self.beta1 = (1 - self.mean1) * new_sum2

        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))


    def doubt_fancy(self, most_distant1, most_distant2, n):
        constraint1 = n
        a1 = abs(self.mean1 - most_distant1)
        b2 = abs((1 - self.mean1) - (1 - most_distant1))
        new_sum1 = constraint1 / (a1 + b2)
        if new_sum1 < (self.alpha1 + self.beta1):
            self.alpha1 = self.mean1 * new_sum1
            self.beta1 = (1 - self.mean1) * new_sum1

        constraint2 = n
        a1 = abs(self.mean2 - most_distant2)
        b2 = abs((1 - self.mean2) - (1 - most_distant2))
        new_sum2 = constraint2 / (a1 + b2)
        if new_sum2 < (self.alpha2 + self.beta2):
            self.alpha2 = self.mean2 * new_sum2
            self.beta2 = (1 - self.mean2) * new_sum2

        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))


    def doubt_alternative_alternative(self, nr_agreeing, n):
        update = nr_agreeing * n
        if self.my_theory:
            if (self.alpha1 + self.beta1) + update > 0:
                new_sum1 = (self.alpha1 + self.beta1) + nr_agreeing * n  # Agents add to the sum some number
                self.alpha1 = self.mean1 * new_sum1  # this number depends on two things
                self.beta1 = (1 - self.mean1) * new_sum1  # it is positive or negative depending on if ...
            else:
                new_sum1 = 1
                self.alpha1 = self.mean1 * new_sum1
                self.beta1 = (1 - self.mean1) * new_sum1

        if not self.my_theory:
            if (self.alpha2 + self.beta2) + update > 0:  # ... more agents agree of disagree with them
                new_sum2 = (self.alpha2 + self.beta2) + nr_agreeing * n  # its value also depends on N with is ...
                self.alpha2 = self.mean2 * new_sum2  # ... a parameter of the model
                self.beta2 = (1 - self.mean2) * new_sum2
            else:
                new_sum2 = 1
                self.alpha2 = self.mean2 * new_sum2
                self.beta2 = (1 - self.mean2) * new_sum2

        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))



class Greedy(Agent):
    def __init__(self, name, theory, prior, epsilon, threshold):
        super().__init__(name, theory, prior, epsilon, threshold)
        self.epsilon_type, self.epsilon_value = epsilon
        self.epsilon = 0

    def experiment(self, pulls, theory1, theory2, nr_agents):
        sum = self.alpha1 + self.beta1 + self.alpha2 + self.alpha1
        factor = nr_agents * pulls * self.epsilon_value

        if self.epsilon_type:
            self.epsilon = factor / sum if sum > (factor / 0.3) else float(1/3)
        else:
            self.epsilon = self.epsilon_value

        random_numbers = np.random.random(pulls)                 # For each pull, there is a 1-epsilon chance
        pulls2 = np.sum(random_numbers > (1 - self.epsilon))  # that the agent will pull the other bandit
        pulls1 = pulls - pulls2                               # pulls1 represent pulls from the preferred bandit

        # pulls = np.random.randint(0, self.epsilon, n)        # This is an alternative version of calculating 1 - epsilon
        # pulls2 = np.count_nonzero((pulls == self.epsilon - 1))
        # pulls1 = n - pulls2

        if self.my_theory:
            successes1, successes2 = np.random.binomial(pulls1, theory1), np.random.binomial(pulls2, theory2)
            return successes1, (pulls1 - successes1), successes2, (pulls2 - successes2)

        else:
            successes1, successes2 = np.random.binomial(pulls2, theory1), np.random.binomial(pulls1, theory2)
            return successes1, (pulls2 - successes1), successes2, (pulls1 - successes2)

    def update(self, pulls, theory1, theory2, nr_agents):      # Updating alpha, beta, mean, variance for agents who do not share data
        sum = self.alpha1 + self.beta1 + self.alpha2 + self.alpha1
        factor = nr_agents * pulls * self.epsilon
        if self.epsilon_type:
            factor = nr_agents * pulls * self.epsilon_value
            self.epsilon = factor / sum if sum > (factor / 0.3) else float(1/3)
        else:
            self.epsilon = self.epsilon_value

        random_numbers = np.random.random(pulls)                 # For each pull, there is a 1-epsilon chance
        pulls2 = np.sum(random_numbers > (1 - self.epsilon))  # that the agent will pull the other bandit
        pulls1 = pulls - pulls2                               # pulls1 represent pulls from the preferred bandit

        if self.my_theory:
            successes1 = np.random.binomial(pulls1, theory1)
            successes2 = np.random.binomial(pulls2, theory2)
            self.alpha1 += successes1
            self.beta1 += (pulls1 - successes1)
            self.alpha2 += successes2
            self.beta2 += (pulls2 - successes2)
        else:
            successes1 = np.random.binomial(pulls2, theory1)
            successes2 = np.random.binomial(pulls1, theory2)
            self.alpha1 += successes1
            self.beta1 += (pulls2 - successes1)
            self.alpha2 += successes2
            self.beta2 += (pulls1 - successes2)

        self.mean1 = self.alpha1 / (self.alpha1 + self.beta1)
        self.mean2 = self.alpha2 / (self.alpha2 + self.beta2)
        self.var1 = (self.alpha1 * self.beta1) / ((self.alpha1 + self.beta1) ** 2 * (self.alpha1 + self.beta1 + 1))
        self.var2 = (self.alpha2 * self.beta2) / ((self.alpha2 + self.beta2) ** 2 * (self.alpha2 + self.beta2 + 1))


class Cautious(Agent):
    def __init__(self, name, theory, prior, epsilon, threshold):
        super().__init__(name, theory, prior, epsilon, threshold)
        self.switch_point = 0
        self.threshold = threshold
        self.counter = 0


    def update_theory(self, pulls, nr_agents):      # Updating value my_theory
        factor = nr_agents * pulls * self.threshold

        # if self.my_theory == (self.mean1 > self.mean2):   # Switching based on confidence, but only looking at the other theory
        #     self.switch_point = 0
        #     self.counter = 0
        #
        # if self.my_theory != (self.mean1 > self.mean2) and not self.switch_point:
        #     if self.my_theory:
        #         self.switch_point = self.alpha2 + self.beta2
        #     else:
        #         self.switch_point = self.alpha1 + self.beta1
        #
        # if self.my_theory != (self.mean1 > self.mean2) and self.switch_point:
        #     if self.my_theory:
        #         if (self.alpha2 + self.beta2) > (self.switch_point + factor) or self.counter >= 50:
        #             self.my_theory = False
        #             self.switch_point = 0
        #             self.counter = 0
        #         else:
        #             self.counter += 1
        #     else:
        #         if (self.alpha1 + self.beta1) > (self.switch_point + factor) or self.counter >= 50:
        #             self.my_theory = True
        #             self.switch_point = 0
        #             self.counter = 0
        #         else:
        #             self.counter += 1

        if self.my_theory == (self.mean1 > self.mean2):        # Switching based on rounds
            self.counter = 0

        if self.my_theory != (self.mean1 > self.mean2) and self.counter < self.threshold:
            self.counter += 1
        elif self.my_theory != (self.mean1 > self.mean2) and self.counter >= self.threshold:
            self.my_theory = self.mean1 > self.mean2
            self.counter = 0

        # if self.my_theory == (self.mean1 > self.mean2):   # Switching based on confidence, but looking at both theories
        #     self.switch_point = tuple()
        #
        # if self.my_theory != (self.mean1 > self.mean2) and not self.switch_point:
        #     self.switch_point = (self.alpha1 + self.beta1, self.alpha2 + self.beta2)
        #
        # factor = nr_agents * n * 5
        #
        # if self.my_theory != (self.mean1 > self.mean2) and self.switch_point:
        #     if (self.alpha1 + self.beta1) > (self.switch_point[0] + factor) and (self.alpha2 + self.beta2) > (self.switch_point[1] + factor):
        #         self.my_theory = self.mean1 > self.mean2
        #         self.switch_point = tuple()


class Combined(Agent):
    def __init__(self, name, theory, prior, epsilon, threshold):
        super().__init__(name, theory, prior, epsilon, threshold)
        self.epsilon_type, self.epsilon_value = epsilon
        self.epsilon = 0
        self.switch_point = 0
        self.threshold = threshold
        self.counter = 0

    def experiment(self, pulls, theory1, theory2, nr_agents):
        sum = self.alpha1 + self.beta1 + self.alpha2 + self.alpha1
        factor = nr_agents * pulls * self.epsilon_value
        if self.epsilon_type:
            self.epsilon = factor / sum if sum > (factor / 0.3) else float(1/3)
        else:
            self.epsilon = self.epsilon_value

        random_numbers = np.random.random(pulls)                 # For each pull, there is a 1-epsilon chance
        pulls2 = np.sum(random_numbers > (1 - self.epsilon))  # that the agent will pull the other bandit
        pulls1 = pulls - pulls2                               # pulls1 represent pulls from the preferred bandit

        # pulls = np.random.randint(0, self.epsilon, n)        # This is an alternative version of calculating 1 - epsilon
        # pulls2 = np.count_nonzero((pulls == self.epsilon - 1))
        # pulls1 = n - pulls2

        if self.my_theory:
            successes1, successes2 = np.random.binomial(pulls1, theory1), np.random.binomial(pulls2, theory2)
            return successes1, (pulls1 - successes1), successes2, (pulls2 - successes2)

        else:
            successes1, successes2 = np.random.binomial(pulls2, theory1), np.random.binomial(pulls1, theory2)
            return successes1, (pulls2 - successes1), successes2, (pulls1 - successes2)

    def update_theory(self, pulls, nr_agents):
        if self.my_theory == (self.mean1 > self.mean2):        # Switching based on rounds
            self.counter = 0

        if self.my_theory != (self.mean1 > self.mean2) and self.counter < self.threshold:
            self.counter += 1
        elif self.my_theory != (self.mean1 > self.mean2) and self.counter >= self.threshold:
            self.my_theory = self.mean1 > self.mean2
            self.counter = 0

        # factor = nr_agents * pulls * self.threshold
        #
        # if self.my_theory == (self.mean1 > self.mean2):   # Switching based on confidence, but only looking at the other theory
        #     self.switch_point = 0
        #     self.counter = 0
        #
        # if self.my_theory != (self.mean1 > self.mean2) and not self.switch_point:
        #     if self.my_theory:
        #         self.switch_point = self.alpha2 + self.beta2
        #     else:
        #         self.switch_point = self.alpha1 + self.beta1
        #
        # if self.my_theory != (self.mean1 > self.mean2) and self.switch_point:
        #     if self.my_theory:
        #         if (self.alpha2 + self.beta2) > (self.switch_point + factor) or self.counter >= 50:
        #             self.my_theory = False
        #             self.switch_point = 0
        #             self.counter = 0
        #         else:
        #             self.counter += 1
        #     else:
        #         if (self.alpha1 + self.beta1) > (self.switch_point + factor) or self.counter >= 50:
        #             self.my_theory = True
        #             self.switch_point = 0
        #             self.counter = 0
        #         else:
        #             self.counter += 1


class Simulation:
    def __init__(self, runs, rounds):   # One simulation represents multiple runs of the model with the same parameters
        self.results = None
        self.rounds = rounds
        self.runs = runs

    def run(self, rounds, type, data_sharing, nr_agents, pulls, theory1, theory2, interval, n, nr_good, kind, prior, epsilon, threshold, constraint):  # Initiates one run of the model
        result = list()
        round = Round()
        result.append(round.create_agents(nr_agents, nr_good, kind, prior, epsilon, threshold))   # Collects results for the initial setup (before any action)

        for i in range(rounds):           # Initiates one round of the model
            round.collect_data(theory1, theory2, pulls, data_sharing, nr_agents)   # For each round, agents collect data
            round.action(type, interval, n, pulls, nr_agents, constraint)                             # agents do their actions
            result.append(round.results())                       # agents report results
        return result       # returns results for one run of the model, as a list

    def add_results(self, result):
        if self.results is None:
            self.results = np.array(result)       # adds results in a 2d array (a table)
        else:
            self.results = np.vstack((self.results, result))

    def calculate_output(self):
        results_calculated = list()
        for i in range(0, self.rounds + 1):
            results_calculated.append(np.sum((self.results[1:, i]) / self.runs))     # Calculates the averages multiple runs

        return results_calculated


def space(runs, rounds, parameters):   # Parameters must be a list of lists with the right order of elements
    results = dict()
    combinations = list(product(*parameters))   # This calculates all the different combinations of the given parameters

    for combination in combinations:            # Runs simulation for every combination of parameters
        simulation = Simulation(runs, rounds)
        t, data_sharing, nr_agents, pulls, theory1, theory2, interval, n, nr_good, kind, prior, epsilon, threshold, constraint = combination
        for i in range(runs):
            simulation.add_results(simulation.run(rounds, t, data_sharing, nr_agents, pulls, theory1, theory2, interval, n, nr_good, kind, prior, epsilon, threshold, constraint))
            if i % 100 == 0:
                print(f"{combination}, run {i} of {runs}")
        results[combination] = simulation.results

    return results
