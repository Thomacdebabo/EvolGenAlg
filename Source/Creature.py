import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Source.Brain import Brain
plt.ioff()
matplotlib.use('Qt5Agg')
class Creature:
    def __init__(self, id, starting_energy, starting_loc, starting_height, destination,step_size = 10, brain = None, enable_food = False):
        self.id = id
        self.brain = Brain(10,2, brain)
        self.trajectory = [np.array(starting_loc)]
        self.score = 0

        self.starting_energy = starting_energy
        self.energy = starting_energy
        self.loc = np.array(starting_loc)

        self.height = starting_height
        self.heights = [starting_height, starting_height, starting_height, starting_height] #additional heigths at [+1,-1], [+1,+1]

        self.delta_energy = 0
        self.delta_height = 0
        self.last_step = [1,1]

        self.destination = destination
        self.step_size = step_size

        self.reached_destination_flag = False
        self.dead = False
        # self.skip = False
        if enable_food:
            self.food = np.zeros((5), dtype= "float")

    def __str__(self):
        return "ID: " + str(self.id) + " Location: " + str(self.loc) + " Height: " + str(self.height) + " Energy: " + str(self.energy) + " Score: " + str(self.score)\
               + " Reached Destination:" + str(self.reached_destination_flag) + " Dead: " + str(self.dead)


    def make_step(self, border_up, border_down):
        step = self.decide()
        # if step[0]+step[1] == 0:
        #     self.dead = True
        self.last_step = step
        # self.loc[0] += step[0]
        # self.loc[1] += step[1]
        self.loc += step
        # self.loc[0] = max(min(self.loc[0],border_up), border_down)
        # self.loc[1] = max(min(self.loc[1],border_up), border_down)
        self.loc = np.maximum(np.minimum(self.loc,border_up), border_down)
        self.trajectory.append(self.loc)
        return step
    def update_score(self, f=10.0):
        # norm = 3.0 / (np.linalg.norm(
        #     np.array(self.destination) - np.array(self.loc), ord=2) + 1)
        if np.all(self.loc == self.trajectory[0]):
            self.dead = True
        norm = np.linalg.norm(self.destination - self.loc, ord=2)
        norm_2 = max(f / (norm + 1.0), 6.0)


        self.score = 10.0*(self.energy-self.starting_energy)  + norm_2 - 0.005*norm
        if hasattr(self, 'food'):self.score += 5.0*np.sum(self.food)
        if self.dead: self.score -= 2.0
        if self.reached_destination_flag: self.score += 2.0

    def get_loc(self):
        return self.loc
    def get_height(self):
        return self.height
    def set_height(self, height, heigths):
        self.delta_height = height - self.delta_height
        self.height = height
        self.heights = heigths
    def reached_destination_check(self):
        # if self.loc == self.destination:
        #     self.reached_destination_flag = True
        # if  self.destination[0] - self.step_size <= self.loc[0] <= self.destination[0]+ self.step_size\
        #         and self.destination[1] - self.step_size <= self.loc[1] <= self.destination[1]+ self.step_size :
        l = self.destination - self.step_size
        u = self.destination + self.step_size

        if np.all( l <= self.loc) and np.all( self.loc <= u):
            self.reached_destination_flag = True
    def death_check(self, up, down):
        #if up in self.loc or down in self.loc or self.energy < 0.0:
        if up in list(self.loc) or down in list(self.loc):
            self.dead = True
    def update_energy(self, energy,step_len):
        #energy_per_step = -0.0
        if energy > 0:
            energy = energy*0.4
        else:
             energy = energy*1.3
        self.energy += energy
        #self.energy += energy_per_step
        self.delta_energy = energy
    def update_food(self, f_list):
        i = 0
        for f in f_list:
            norm = np.linalg.norm(np.array(f) - self.loc, ord=2)
            norm_2 = 1.0 / (norm + 1.0)
            self.food[i] = max(self.food[i],norm_2)
            i+=1
    def initiate_food(self, len):
        self.food = np.array(len, dtype = "float")


    def decide(self):
        input = np.array([
                        self.energy,
                        self.height,
                        self.delta_height,
                        self.destination[0] - self.loc[0],
                        self.destination[1] - self.loc[1],
                        # self.trajectory[-1][0],
                        # self.trajectory[-1][1],
                        self.heights[0],
                        self.heights[1],
                        self.heights[2],
                        self.heights[3],
                        # self.last_step[0],
                        # self.last_step[1],
                        self.trajectory.__len__()/100.0,
                        # np.sum(self.food)
                         ]
                         )
        out = self.brain.predict(input)
        out = np.maximum(np.minimum(out/(np.linalg.norm(out, ord = 2) + 0.000001)*self.step_size,self.step_size), -self.step_size)
        #out = np.minimum(out*self.step_size,self.step_size)
        # return (out/norm*step_size).astype("int")

        return (out).astype("int")





