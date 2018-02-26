# This file has been modified from:
# https://github.com/DEAP/deap/blob/master/examples/ga/xkcd.py

#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
"""This example shows a possible answer to a problem that can be found in this
xkcd comics: http://xkcd.com/287/. In the comic, the characters want to get
exactly 15.05$ worth of appetizers, as fast as possible."""

import random
from operator import attrgetter
from collections import Counter

# We delete the reduction function of the Counter because it doesn't copy added
# attributes. Because we create a class that inherits from the Counter, the
# fitness attribute was not copied by the deepcopy. This is just a DEAP technicality
# due to the fact we're using a standard Python class for our Individuals
try:
    del Counter.__reduce__
except: pass

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Create the item dictionary: item id is an integer, and value is 
# a (name, weight, value) 3-uple. Since the comic didn't specified a time for
# each menu item, we'll create our own times.
ITEMS = {'French Fries': (2.75, 5),
         'Hot Wings': (3.55, 7),
         'Mixed Fruit': (2.15, 2),
         'Mozzarella Sticks': (4.2, 4),
         'Sampler Plate': (5.8, 10),
         'Side Salad': (3.35, 3)}
ITEMS_NAME = list(ITEMS.keys())

# Our fitness function will have three values:
# cost difference from target price (we want to spend as much as possible but not more than our budget),
# time to eat (max of time to each individual appetizers),
# amount of food (sum of counts of appetizers)
#
# -1.0 means we are minimzing that value, 1.0 means we are maximizing
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))

# Our "individual" will be a dictionary where the values are counts,
# so, a Python Counter object; we'll look at example individuals below.
creator.create("Individual", Counter, fitness=creator.FitnessMulti)

# initially, create individuals that have 2 items
IND_INIT_SIZE = 2

toolbox = base.Toolbox()
# individuals will be made up of counts associated with food names (from ITEM_NAMES)
toolbox.register("attr_item", random.choice, ITEMS_NAME)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalXKCD(individual, target_price):
    """Evaluates the fitness and return the error on the price and the time
    taken by the order if the chef can cook everything in parallel."""
    price = 0.0
    times = [0]
    food = 0
    for item, number in individual.items():
        if number > 0:
            price += ITEMS[item][0] * number
            times.append(ITEMS[item][1])
            food += number
    return (price - target_price), max(times), food

def cxCounter(ind1, ind2, indpb):
    """Swaps the number of randomly-chosen items between two individuals"""
    for key in ITEMS.keys():
        if random.random() < indpb:
            ind1[key], ind2[key] = ind2[key], ind1[key]
    return ind1, ind2

def mutCounter(individual):
    """Adds or remove an item from an individual"""
    if random.random() > 0.5:
        individual.update([random.choice(ITEMS_NAME)])
    else:
        val = random.choice(ITEMS_NAME)
        individual.subtract([val])
        if individual[val] < 0:
            del individual[val]
    return individual,

import sys

# register the functions, some with extra fixed arguments (like target_price)
toolbox.register("evaluate", evalXKCD, target_price=15.05)
# severely penalize individuals that are over budget ($15.05))
# return a fixed terrible fitness
toolbox.decorate("evaluate", tools.DeltaPenalty(lambda ind: evalXKCD(ind, 15.05)[0] <= 0,
                                               (-sys.float_info.max,
                                                sys.float_info.max,
                                                -sys.float_info.max)))
toolbox.register("mate", cxCounter, indpb=0.5)
toolbox.register("mutate", mutCounter)
toolbox.register("select", tools.selNSGA2)


# Simulation parameters:
# Number of generations = 100
NGEN = 100
# The number of individuals to select for the next generation (eliminate bad ones).
MU = 50
# The number of children to produce at each generation.
LAMBDA = 20
# The probability that an offspring is produced by crossover.
CXPB = 0.3
# The probability that an offspring is produced by mutation.
MUTPB = 0.3

# Start population at 50
pop = toolbox.population(n=MU)

# Also keep track of the "hall of fame" individual

# "The hall of fame contains the best individual that ever lived
# in the population during the evolution." DEAP docs

# Pareto Front means the best individual is the one that is equal
# or better on all dimensions of the fitness function; ours has
# two dimensions (cost difference from target and max time),
# so the HOF individual is the one that is lowest on both
hof = tools.ParetoFront()

# compute some statistics as the simulation proceeds
price_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
time_stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
food_stats = tools.Statistics(key=lambda ind: ind.fitness.values[2])

stats = tools.MultiStatistics(price=price_stats, time=time_stats, food=food_stats)
stats.register("avg", numpy.mean, axis=0)

# run the simulation
algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                          stats, halloffame=hof)

print("Best:", hof[0], hof[0].fitness.values)



