# in order to be able to solve based on problem stateement only allow it to break apart simpler terms
"""
create report for me
goes online searches based on how i would do it
give template language

#reward function  close as possible to what we would have gotten
#file search
#answer questions


1}internet-search
2}internet-open_url
3}internet-collect data
4}word complete data learnt
"""

from sklearn.metrics.pairwise import cosine_similarity

action = ["leave", "take"]
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from sentence_transformers import SentenceTransformer

emodel = SentenceTransformer('bert-base-nli-mean-tokens')


class Online_Data_Env(Env):
    def __init__(self, max):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(2)
        self.max = max
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([700]))
        # Set start temp
        self.state = 1
        # Set shower length
        self.links_length = max

    def step(self, cosine_factor, action):

        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        self.state += action
        # Reduce shower length by 1 second
        self.links_length -= 1

        # Calculate reward
        val_confirm = (self.max - self.links_length + 1) / self.state * cosine_factor * (
                self.max - self.links_length) / self.max
        # progress bound to
        # run twice but invert list
        print(val_confirm)
        if val_confirm >= 0.7:
            reward = 1
        else:
            reward = -1

            # Check if shower is done
        if self.links_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        # self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return val_confirm, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset shower temperature
        return self.state


def _main(env, i):
    state = env.reset()
    test = "lovely"
    done = False
    score = 0
    cosine_factor = cosine_similarity([emodel.encode(test)], [emodel.encode(i)])[0][0]
    # print(cosine_factor)
    if cosine_factor >= 0.69:
        action = 1
    else:
        action = 0
    n_state, reward, done, info = env.step(cosine_factor, action)

    return n_state


list_test = ["love", "lovely", "hate"]


def main(list_test):
    env = Online_Data_Env(len(list_test))
    # this script is essential for data but
    values = [_main(env, i) for i in list_test]
    list_test.reverse()
    values.reverse()
    values = np.average([values, [_main(env, i) for i in list_test]], axis=0)
    print(values)
    values = zip(values, list_test)
    values = sorted(values, key=lambda x: x[1], reverse=True)
    print(values)
    # return values


main(list_test)

"""
values = sorted(values, key=values.count, reverse=True)
print(values)
print(set(values))
"""
# data gathering internet
# instruction giving
#shortcut utilization
# output giving store in terminal
