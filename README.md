# README

Documentation for a modification of an agent based model first introduced to philosophy of science by Kevin Zollman ([2007](https://www.cambridge.org/core/journals/philosophy-of-science/article/communication-structure-of-epistemic-communities/B1A3770084C04C26A3533626E7AABBFD), [2010](https://link.springer.com/article/10.1007/s10670-009-9194-6)).

This is very much still work in progress. The development of this model is part of my masters’ thesis project at Faculty of Arts, University of Ljubljana. 

## Basics

In this model, agents can choose to pull from one of the two slot machines or “bandits”. Each bandit has a different success rate; the goal of the agents as a group is to converge to the better one.

Agents’ beliefs about the success rate of the two bandits are represented by two beta distributions, where alpha represents the number of successful pulls from a bandit and beta the number of unsuccessful pulls. Agents’ assessments of the bandits are thus equal to the means of the two distributions. 

$C_1 = \mu_1= \frac{\alpha_1}{\alpha_1+\beta_1}$$

$C_2 = \mu_2= \frac{\alpha_2}{\alpha_2+\beta_2}$

Each agents starts with some prior beliefs about the success rate of each bandit. Alphas and betas for the priors are usually pulled from an uniform distribution between some interval.

Agents can also share data about their pulls. The simulation runs in rounds: each round agents pull from one of the two bandits, share data and update their assessments.

## Usage in philosophy of science

This model has be used to study grup learning under different conditions. Some examples include:

- Zollman ([2007](https://www.cambridge.org/core/journals/philosophy-of-science/article/communication-structure-of-epistemic-communities/B1A3770084C04C26A3533626E7AABBFD)) studied how different ways of sharing the data between agents (i.e., the social structure of the model) affect the speed and successfulness of the group inquiry
- Kummerfeld & Zollman ([2016](https://www.journals.uchicago.edu/doi/full/10.1093/bjps/axv013)) studied different exploratory strategies where agents pull from the preferred bandit with 1 - epsilon probability
- Weatherall, O’Connor and Bruner ([2020](https://www.journals.uchicago.edu/doi/10.1093/bjps/axy062)) looked at what happened if some agents selectively report their data or if some agents can make more pulls each round than others
- Watherall and O’Connor ([2020](https://link.springer.com/article/10.1007/s11229-019-02520-2)) studied the effects of agents confirming to the majority when choosing which bandit to pull
- Fry and Šešelja ([2020](https://www.journals.uchicago.edu/doi/10.1093/bjps/axy039)) tested the robustness of Zollman’s 2007 and 2010 results under more relaxed assumptions

## This model

In this version of the model, I tried to implement the possibility of agents to disagree and react differently to disagreement. This is an attempt to bridge the gap between simulation work in philosophy of science and epistemological debates about peer [disagreement](https://plato.stanford.edu/entries/disagreement/). 

In addition to pulling from bandits, sharing data and updating their assessments, agents can thus perform some additional actions.

### **********************Conciliation**********************

Some epistemologists [suggest](https://www.jstor.org/stable/4494542) that when two epistemic peers disagree, they should “split the difference” between their believes. 

In the model, agents who act in accordance with this norm average their assessment every round after updating on evidence. The averaged assessment is calculated as 

 

$$
C_{avg} = \frac{\sum_{i=1}^{n}C_i+C_{i+1}+...+C_n}{n},
$$

where ***n*** is the number of agents. 

Agents then update so that 

$$
C_i' = C_{avg},
$$


where C’ is a posterior assessment of an agent. 

Agents then also update their alphas and betas so that the variance of their distributions remains the same.

$\alpha'_i = (\frac{1-C_i'}{\sigma^2_i} - \frac{1}{C_i'})\times C'^2_i$

$\beta'_i = \alpha_i\times(\frac{1}{C'_i}-1)$

(They do this for the distributions for both bandits.)

### Degree of conciliation

In an alternative version of this norm, agents only split the difference with agents who’s assessments of the success rate of the bandits is not to distant from theirs. 

In other words, agent A updates based on the assessments for which 

 

$$
|C_a - C_i| < distance
$$

where *********distance********* is a parameter of the model.

(They do this for the distributions for both bandits.)

### Doubt

Alternatively, some epistemologists [suggest](https://academic.oup.com/book/35093/chapter/299156015) that when faced with disagreement, agents should lower their confidence in their assessment not the actual assessment. 

The available literature doesn’t provide a technical notion of confidence. In this model, I understand  confidence change as change in variance of the distribution (in contrast to change in the mean).

The model implements two different ways of variance change.

****************Doubt #1****************

Agents who act in accordance with this norm first check whether any of the other agents disagree with them about which of the bandits has a higher success rate. 

If they discover such an agent, their update their alpha and beta in the following way:

$$
\alpha_i' = C_i \times (\frac{\alpha_i+\beta_i}{2})
$$

$$
\beta'_i = (1-C_i)\times(\frac{\alpha_i+\beta_i}{2})
$$

In other words, they halve the sum of alpha and beta and calculate new ones so that the mean of the distribution remains the same. That way, the variance increases. 

(They do this only for the distribution of the bandit they think is the better one).

********************Doubt #2********************

Here, agents count the number of other agents how agree and disagree with them. This is represented by one number that can be either negative (if more agents disagree), positive (if more agents agree) or zero (if an equal number of agents agree and disagree).

Then, they update their alphas and betas in the following way:

$$
\alpha_i' = C_i \times ((\alpha_i+\beta_i) + a \times N)
$$

$$
\alpha_i' = (1 - C_i) \times ((\alpha_i+\beta_i) + a \times N)
$$

Where **a** is the count of (dis)agreeing agents and **N** is a parameter of the model.

(They do this for the distributions for both bandits.)

### Steadfastness

Some epistemologists also suggest that in face of peer disagreement, agents should stick to their guns. 

Agents who act in accordance with this norm act in the same way as the agents in Zollman’s version of the model. They collect data, exchange it and update based on it.

## Plan of work

In further developing the model, I plan to implement some other, perhaps more principled was of doubtful behavior. In addition, I want to try some other exploratory strategies of agents and explore the interaction between them and the above described behaviors.