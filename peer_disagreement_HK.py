import random
import mesa
import numpy as np
import pandas as pd


beliefs = [random.uniform(0,1) for i in range(25)]


class Agent(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.belief = beliefs[unique_id]
        self.posterior = float()
        self.evidence = list()

    def step(self):
        prior_list = list()
        for agent in self.model.schedule.agents:
            if abs(self.belief - agent.belief) <= self.model.epsilon:
                prior_list.append(agent.belief)
        prior = sum(prior_list) / len(prior_list)

        type, value = self.model.noise
        truth = self.model.truth
        if type == "uniform":
            signal = random.uniform(truth-value, truth+value)
        elif type == "normal":
            signal = float(np.random.normal(truth, value / 3, 1))
        else:
            signal = self.model.truth

        if len(self.evidence) < (self.model.memory + 2):
            self.evidence.append(signal)
        else:
            self.evidence.pop(0)
            self.evidence.append(signal)


        self.posterior = (self.model.alpha * prior +
                          (1 - self.model.alpha) * (sum(self.evidence)/len(self.evidence)))

    def advance(self):
        self.belief = self.posterior


class Model(mesa.Model):
    def __init__(self, N, epsilon, alpha, noise_type, noise_value, memory, truth):
        super().__init__()
        self.num_agents = N
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.alpha = alpha
        self.memory = memory
        self.epsilon = epsilon
        self.noise = noise_type, noise_value
        self.truth = truth

        for i in range(self.num_agents):
            a = Agent(i, self)
            self.schedule.add(a)

        self.datacollector = mesa.DataCollector(
            agent_reporters={"Belief": "belief"}
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()




params = {
    "N":25,
    "epsilon": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "alpha":[0.5],
    "noise_type":"uniform",
    "noise_value":0.2,
    "memory":10,
    "truth":0.75,
}

results = mesa.batch_run(
    Model,
    parameters=params,
    iterations=1,
    max_steps=50,
    number_processes=1,
    data_collection_period=1,
    display_progress=True
)

results_df = pd.DataFrame(results)
results_df.to_csv("seminarska-test.csv")
