import time

import numpy as np

from neurosap import actions, neurosap
from neurosap.encoding import decode, encode
from neurosap.helpers import display_data, get_game_imgs, get_legal_moves, is_game_over
from neurosap.index import decoding_str_index as decoding_index
import logging


class SAP:
    def __init__(self, population_count: int = 10):
        self.population_count = population_count
        self.ns = neurosap.NeuroSAP(population_count)
        self.finished_agents = 0
        self.data = [-1 for _ in range(51)]

        self.timer = 10

    def start(self):
        logging.info("Starting Generation")
        for i in range(self.population_count - self.finished_agents):
            logging.info(f"Agent {i} | Starting")
            self.turn_time = time.perf_counter()
            actions.start()
            time.sleep(2)
            while not is_game_over():
                self.start_time = time.perf_counter()
                self.step()

            logging.info(f"Agent {i} | Game over | Fitness: {self.data[2]}")
            self.handle_game_over()

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
            # self.data[3] is turn
            logging.debug(
                f"Turn {self.data[3]} ended | Timer reset | "
                + f"{time.perf_counter() - self.turn_time:.2f}s"
            )
            decode(61)()
            self.timer = 10
            self.handle_combat()
        else:
            idx = outputs.argmax()
            logging.debug(decoding_index[idx] + f" | {time.perf_counter() - self.start_time:.2f}s")
            decode(idx)()
            self.timer -= 1
            time.sleep(0.5)

    def handle_combat(self):
        for _ in range(20):
            actions.safe()
            time.sleep(0.5)
        self.turn_time = time.perf_counter()

    def handle_game_over(self):
        if self.finished_agents < self.population_count:
            # 2 * Trophies + Turns (past 6)
            self.finish_agent(self.data[2] * 2 + self.data[3] - 6)

    def finish_agent(self, fitness: float) -> None:
        self.ns.finish_agent(fitness)
        self.finished_agents += 1
        if self.finished_agents == self.population_count:
            self.ns.evolve()
            self.finished_agents = 0

    def evolve(self) -> bool:
        if self.finished_agents == self.population_count:
            logging.info("Evolving population")
            self.ns.evolve()

    def display_data(self):
        display_data(self.data)

    def _reset_data(self) -> None:
        # We don't want to reset the data for held items
        stored_indices = [7, 13, 19, 25, 31]
        self.data = [-1 if i not in stored_indices else self.data[i] for i in range(52)]
