import numpy as np
import matplotlib.pyplot as plt
from Source.World import World


def plot_scores(scores):
    fig = plt.figure()
    ax2 = plt.axes()

    x = np.linspace(0, scores.__len__(), scores.__len__())
    ax2.plot(x, np.array(scores));
def test(world):
    steps = 1000
    world.mutate_rate = 0.0
    world.supremacy_num = 100
    world.breed_rate = 0.0
    world.load_creatures(path = "Creatures.pkl")
    world.mutate_rate = 0.0
    world.sort_creatures()

    world.renew(100,100)
    for i in range(10):
        # world.set_random
        # world.set_destination(world.get_rand_loc())

        for i in range(steps):
            world.make_step()

        world.update_scores()
        world.sort_creatures()
        world.print_creatures()
        world.plot_world()
        world.update_plot()

        plt.show()
        world.destination = np.array(world.get_rand_loc())
        world.starting_point = np.array(world.get_rand_loc())
        world.renew(100,100)

def brain_disection(world):

    world.load_creatures(path="Creatures_bkp.pkl")
    world.sort_creatures()
    world.renew(10, 10)
    for c in world.Creatures:
        print(c)
        c.brain.print()

def train(num_creatures, world, epochs, steps, survive, falloff = 0.8, period=10, min_cr = 30, max_cr = 1000):
    for i in range(num_creatures):
        world.create_creature()
    world.summary()

    scores = []
    for e in range(epochs):
        print("Epoch: " + str(e))
        for i in range(steps):
            world.make_step()
        world.update_scores()
        world.sort_creatures()
        world.print_creatures()
        # world.update_creature_dict()
        scores.append(world.get_best_score())
        if e+1 == epochs:
            break
        world.renew(int(survive*num_creatures), num_creatures)
        if (e+1) % period == 0:
            if num_creatures > min_cr and num_creatures < max_cr:
                num_creatures = int(num_creatures *falloff)
    world.update_plot()
    plot_scores(scores)
    plt.show()
    world.save_creatures()


# start = time()


######### Main #############
#0 is testing mode
#1 is training mode
mode = 1

world = World(size = 1000,
              scale = 600,
              step_size = 75,
              destination = [500,500],
              starting_point=[-1000,-1000],
              food_list=[[0,-1000],[0,0],[-100,-700],[-700,-500],[43,21]],
              rand_loc= True, #Overwrites the other parameters entered above
              enable_food = False)
survive = 0.5
mutation_rate = 0.7
breed_rate = 0.4
num_creatures =200
steps = 1000
epochs = 200
world.plot_world()
world.breed_rate = breed_rate
world.mutate_rate = mutation_rate

#brain_disection(world)
if mode:
    train(num_creatures, world, epochs, steps,survive, min_cr = 30, period= 10, max_cr = 1000, falloff = 0.5)
else:
    test(world)