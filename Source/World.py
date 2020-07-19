import numpy as np
import noise
import matplotlib.pyplot as plt
import matplotlib
from random import random
import random
import pickle
from Source.Creature import Creature

plt.ioff()
matplotlib.use('Qt5Agg')
def create_perlin_noise(shape = (4000, 4000),scale = 600.0,octaves = 2,persistence = 0.3,lacunarity = 2.0):
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=shape[0],
                                        repeaty=shape[1],
                                        base=0)
    return world


class World:
    def __init__(self, scale = 600,
                 size = 1024,
                 step_size = 50,
                 destination =[1000,1000] ,
                 starting_point = [0,-1000],
                 food_list = [[0,-1000],[-500,-500],[-100,700],[800,-500],[43,21]],
                 rand_loc = False,
                 enable_food = False):
        print("Building World...")
        self.step_size = step_size
        self.world_size = size + self.step_size
        self.map = create_perlin_noise(shape=(2*self.world_size, 2*self.world_size), scale=scale)
        self.Creatures = []

        self.border_up = size -1
        self.border_down = -size
        if rand_loc:
            self.destination = np.array(self.get_rand_loc())
            self.starting_point = np.array(self.get_rand_loc())
        else:
            self.destination = np.array(destination)
            self.starting_point = np.array(starting_point)

        self.mutate_rate = 0.2
        self.breed_rate = 0.2
        self.supremacy_num = 3

        self.id = 0
        self.f_n = 100.0
        self.f = np.linalg.norm(self.destination - self.starting_point, ord=2)/self.f_n

        self.enable_food = enable_food
        if self.enable_food:
            if rand_loc:
                self.food_list = [self.get_rand_loc(),
                                  self.get_rand_loc(),
                                  self.get_rand_loc(),
                                  self.get_rand_loc(),
                                  self.get_rand_loc()
                                  ]
            else:
                self.food_list = food_list


        print("World built!")
    def new_world(self, scale = 400, octave = 1):
        self.map = create_perlin_noise(scale=scale, octaves = octave)
    def get_rand_loc(self):
        return [random.randint(self.border_down,self.border_up), random.randint(self.border_down,self.border_up)]
    def summary(self):
        print("Map size: " + str(self.map.shape))
        print("Creatures:" + str(self.Creatures.__len__()))
        print("Destination: " + str(self.destination))
        print("Mutation Rate: " + str(self.mutate_rate) + " Breeding Rate: " + str(self.breed_rate) + " Supremacy: " + str(self.supremacy_num) )
        print("f: " +str(self.f))

    def get_height(self, x, y):
        return self.map[x+self.world_size][y+self.world_size]

    def create_creature(self, id = None, brain = None, starting_energy = 0.0):
        if id == None:
            id = self.get_unique_id()
        self.Creatures.append(Creature(id,starting_energy,self.starting_point,0,  self.destination, step_size=self.step_size,brain = brain, enable_food=self.enable_food))

    def update_scores(self):
        for c in self.Creatures:
            c.update_score(self.f)
    def set_start(self, start):
        self.starting_point = np.array(start)
        self.f = np.linalg.norm(self.destination - self.starting_point, ord=2) / self.f_n
        for i in range(self.Creatures.__len__()):
            self.Creatures[i].loc = self.starting_point
            self.Creatures[i].loc = [self.starting_point]
    def set_destination(self, destination):
        self.destination = np.array(destination)
        self.f = np.linalg.norm(self.destination - self.starting_point, ord=2) / self.f_n
        for i in range(self.Creatures.__len__()):
            self.Creatures[i].destination = self.destination

    def get_score(self,creature):
        return creature.score
    def get_unique_id(self):
        self.id += 1
        return self.id
    def sort_creatures(self):
        self.Creatures.sort(key=self.get_score, reverse=True)
    def make_step(self):
        i = 0
        for c in self.Creatures:

            if c.reached_destination_flag or c.dead:
                continue
            # if c.id in self.creature_dict:
            #     self.Creatures[i] = self.creature_dict[c.id]
            #     self.Creatures[i].skip = True
            #     continue
            s = c.make_step(self.border_up, self.border_down)
            loc = c.get_loc()

            height = self.get_height(loc[0], loc[1])
            heights = [self.get_height(loc[0]+self.step_size, loc[1]+self.step_size),
                       self.get_height(loc[0]+self.step_size, loc[1]-self.step_size),
                       self.get_height(loc[0]-self.step_size, loc[1]+self.step_size),
                       self.get_height(loc[0]-self.step_size, loc[1]-self.step_size)]
            prev_height = c.get_height()
            c.update_energy(prev_height - height, np.linalg.norm(s, ord = 2)/float(self.step_size*self.step_size))
            #experimental:
            if self.enable_food:
                c.update_food(self.food_list)

            c.set_height(height, heights)
            c.reached_destination_check()
            c.death_check(self.border_up,self.border_down)
    def print_creatures(self, max = 10):
        i = 0
        print("Printing out Creatures")
        for c in self.Creatures:
            if i > max:
                break
            print(c)
            i += 1
    def print_best(self):
        print("Best Creature:")
        print(self.Creatures[0])
    def get_best_score(self):
        return self.Creatures[0].score
    def plot_world(self):
        x = np.linspace(-self.world_size, self.world_size, 2*self.world_size)
        y = np.linspace(-self.world_size, self.world_size, 2*self.world_size)
        X, Y = np.meshgrid(x, y)
        Z = self.map

        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.plot_surface(X, Y, Z,  cmap='binary', alpha = 0.5)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z');
        self.ax.view_init(60, 35)
        d = list(self.destination)
        self.ax.plot([d[0]],[d[1]],[self.get_height(d[0],d[1])+0.1],markerfacecolor=(0,1,0), markeredgecolor='k', marker='o',markersize= 10, alpha = 1)
        d = list(self.starting_point)
        self.ax.plot([d[0]],[d[1]],[self.get_height(d[0],d[1])+0.1],markerfacecolor=(1,0,1), markeredgecolor='k', marker='o',markersize= 10, alpha = 1)
        if self.enable_food:
            for d in self.food_list:
                self.ax.plot([d[0]], [d[1]], [self.get_height(d[0], d[1]) + 1], markerfacecolor=(1, 1, 1),
                             markeredgecolor='k', marker='o', markersize=10, alpha=1)

    def update_plot(self, pl_max = 5):
        i = 0
        ii = 0
        iii = 0
        for c in self.Creatures:
            tr = c.trajectory
            x = []
            y = []
            z = []
            for t in tr:
                x.append(t[0])
                y.append(t[1])
                z.append(self.get_height(t[0], t[1]))
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            if c.reached_destination_flag:
                if c.score > 3.0:
                    if iii < pl_max:
                        iii+=1
                        self.ax.plot3D(x, y, z, 'green')
                else:
                    if ii < pl_max:
                        self.ax.plot3D(x, y, z, 'red')
                        ii+=1
            else:
                if i < pl_max:
                    self.ax.plot3D(x, y, z, 'blue')
                    i+=1

        plt.draw()

    def renew(self, sv, num):
        #self.remove_dead()
        goodbois = self.Creatures[:sv]
        self.Creatures = []
        for i in range(num):
            if i < goodbois.__len__():
                id = goodbois[i].id
                if random.random() < self.mutate_rate and i > self.supremacy_num:
                    goodbois[i].brain.mutate()
                    id = None
                self.create_creature(id = id,brain=goodbois[i].brain.get_params(), starting_energy=goodbois[i].starting_energy)
            else:
                if random.random () < self.breed_rate:
                    c_1 = random.choice(goodbois)
                    c_2 = random.choice(goodbois)
                    p_3 = self.breed(c_1, c_2)
                    self.create_creature(brain = p_3, starting_energy=c_1.starting_energy)
                else:
                    #self.create_creature(starting_energy= float(random.randint(1,10)))
                    self.create_creature(starting_energy= 0.0)
    def remove_dead(self):
        temp = []
        for c in self.Creatures:
            if not c.dead:
                temp.append(c)
        self.Cretures = temp
    def save_creatures(self, path = 'Creatures.pkl'):
        with open(path, 'wb') as output:
            pickle.dump(self.Creatures, output, pickle.HIGHEST_PROTOCOL)
    def load_creatures(self, path = 'Creatures.pkl'):
        with open(path, 'rb') as input:
            self.Creatures = pickle.load(input)
    def breed(self, c_1, c_2):
        p_1 = c_1.brain.get_params()
        p_2 = c_2.brain.get_params()
        p_3 = []
        for i in range(p_1.__len__()):
            m = np.random.randint(0,2, p_1[i].shape)
            p_3.append(p_1[i]*m + p_2[i]*(1-m))
        return p_3
