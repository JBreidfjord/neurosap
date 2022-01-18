import time

import numpy as np

from neurosap import actions, neurosap
from neurosap.encoding import decode, encode
from neurosap.helpers import display_data, get_game_imgs, get_legal_moves, is_game_over
from neurosap.index import decoding_str_index as decoding_index


class SAP:
    def __init__(self, population_count: int = 10):
        self.population_count = population_count
        self.ns = neurosap.NeuroSAP(population_count)
        self._finished_agents = 0
        self.data = [-1 for _ in range(51)]

        self.timer = 10

    def step(self):
        rgb_img, gray_img = get_game_imgs()
        self._reset_data()
        self.data = encode(rgb_img, gray_img, self.data)

        legal_mask = np.array(get_legal_moves(self.data))

        outputs = np.array(self.ns.step(self.data))
        outputs[outputs < 0] = 0
        outputs[legal_mask] = 0
        outputs /= np.sum(outputs)  # Get probabilities

        if outputs.argmax() == 61 or self.timer == 0:
            print("Turn Ended\nTimer reset")
            decode(61)()
            self.timer = 10
            self.handle_combat()
        else:
            idx = outputs.argmax()
            print(decoding_index[idx])
            decode(idx)()
            self.timer -= 1
            time.sleep(1)
            self.step()

    def handle_combat(self):
        for _ in range(20):
            actions.safe()
            time.sleep(0.5)

        if is_game_over():
            print("Game Over")
            self.handle_game_over()
        else:
            self.step()
            
    def handle_game_over(self):
        if self._finished_agents < self.population_count:
            self.finish_agent(self.data[2]) # Trophies
            # self.step() # Continue

    def finish_agent(self, fitness: float) -> None:
        self.ns.finish_agent(fitness)
        self._finished_agents += 1
        if self._finished_agents == self.population_count:
            self.ns.evolve()
            self._finished_agents = 0

    def evolve(self) -> bool:
        if self._finished_agents == self.population_count:
            self.ns.evolve()
            return True
        else:
            return False

    def display_data(self):
        display_data(self.data)

    def _reset_data(self) -> None:
        # We don't want to reset the data for held items
        stored_indices = [7, 13, 19, 25, 31]
        self.data = [-1 if i not in stored_indices else self.data[i] for i in range(52)]
