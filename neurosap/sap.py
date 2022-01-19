import logging
import time

import numpy as np

from neurosap import actions, neurosap
from neurosap.encoding import decode, encode
from neurosap.helpers import display_data, get_game_imgs, get_legal_moves, is_game_over
from neurosap.index import decoding_str_index as decoding_index


class SAP:
    def __init__(self, population_count: int = 10, ns: neurosap.NeuroSAP = None):
        self.ns = neurosap.NeuroSAP(population_count) if ns is None else ns
        self.population_count = population_count
        self.finished_agents = 0
        self.data = [-1 for _ in range(51)]

        self.timer = 5
        self._excess_gold = 0
        self.turn_time = time.perf_counter()
        
    @classmethod
    def load(cls, filepath: str):
        ns = neurosap.load(filepath)
        population_count = ns.population_count()
        return cls(population_count, ns)

    def start(self):
        logging.info("Starting Generation")
        for i in range(self.population_count - self.finished_agents):
            logging.info(f"Agent {i} | Starting")
            self._reset_data(full=True)
            self.turn_time = time.perf_counter()
            if is_game_over():
                actions.start()
            time.sleep(2)
            while not is_game_over():
                self.step()

            logging.info(
                f"Agent {i} | Game over | Fitness: "
                + f"{10 + (self.data[2] * 2) + (self.data[3] - 6) - (self._excess_gold / 10):.2f}"
            )
            self.handle_game_over()
            time.sleep(2)

    def step(self):
        self.start_time = time.perf_counter()

        rgb_img, gray_img = get_game_imgs()
        self._reset_data()
        self.data = encode(rgb_img, gray_img, self.data)
        while self.data[0] == -1 or self.data[1] == -1 or self.data[2] == -1 or self.data[3] == -1:
            logging.debug(
                "Error parsing data | "
                + f"{self.data[0]} {self.data[1]} {self.data[2]} {self.data[3]}"
                + " | Retrying..."
            )
            rgb_img, gray_img = get_game_imgs()
            self.data = encode(rgb_img, gray_img, self.data)

        logging.debug(display_data(self.data))

        legal_mask = np.array(get_legal_moves(self.data))

        outputs = np.array(self.ns.step(self.data))
        outputs -= np.min(outputs)  # Lowest value is 0
        outputs[legal_mask] = 0
        outputs /= np.sum(outputs)  # Get probabilities

        idx = outputs.argmax()
        if idx == 61 or (self.timer <= 0 and idx in range(20)):
            # self.data[3] is turn
            logging.debug(
                f"Turn {self.data[3]} ended | Timer reset | "
                + f"{self.data[0]} excess gold | "
                + f"{time.perf_counter() - self.turn_time:.2f}s"
            )
            self._excess_gold += self.data[0]

            # Skip Agent if battle starts with no team
            if all([pet == -1 for pet in [self.data[i] for i in range(4, 29, 6)]]):
                logging.warn("Battle started with no team | Skipping Agent")
                actions.abandon()
                self.finish_agent(0)
                self._excess_gold = 0
                self.timer = 5
                return

            decode(61)()
            self.timer = 5
            self.handle_combat()
        else:
            if idx in range(20):
                self.handle_move(idx)
            elif idx in range(45, 55):
                self.handle_food(idx)
            logging.debug(
                decoding_index[idx]
                + f" | {time.perf_counter() - self.start_time:.2f}s"
                + f" | {self.timer}"
            )
            decode(idx)()
            time.sleep(0.5)

    def handle_combat(self):
        for _ in range(20):
            actions.safe()
            time.sleep(0.5)
        self.turn_time = time.perf_counter()

    def handle_game_over(self):
        if self.finished_agents < self.population_count:
            # 10 + 2 * Trophies + Turns (past 6) - (Excess Gold / 10)
            self.finish_agent(
                10 + (self.data[2] * 2) + (self.data[3] - 6) - (self._excess_gold / 10)
            )
        self._excess_gold = 0

    def handle_move(self, idx: int):
        if idx >= 20:
            return

        self.timer -= 1
        team_slot_indices = [4, 10, 16, 22, 28]
        team_slots = [self.data[i] for i in team_slot_indices]
        item_id_indices = [7, 13, 19, 25, 31]
        item_ids = [self.data[i] for i in item_id_indices]
        team_indices = [int(t.replace("Team ", "")) for t in decoding_index[idx].split(" to ")]

        from_idx = team_indices[0]
        to_idx = team_indices[1]
        from_item = item_ids[from_idx]
        to_item = item_ids[to_idx]
        from_pet = team_slots[from_idx]
        to_pet = team_slots[to_idx]

        if from_item == -1 and to_item == -1:  # No change
            return

        if to_pet == -1:  # Empty destination
            if from_item != -1:
                self.data[item_id_indices[to_idx]] = from_item
                self.data[item_id_indices[from_idx]] = -1
        elif from_pet == to_pet:  # Merged pets
            if from_item == -1 or to_item == -1:  # Only one has item
                self.data[item_id_indices[to_idx]] = from_item if from_item != -1 else to_item
                self.data[item_id_indices[from_idx]] = -1
            else:  # Both have items
                # to_item remains the same
                self.data[item_id_indices[from_idx]] = -1
        else:  # Different pets
            if abs(from_idx - to_idx) == 1:  # Pets are adjacent
                # Adjacent pets are swapped
                self.data[item_id_indices[to_idx]] = from_item
                self.data[item_id_indices[from_idx]] = to_item
            else:  # Pets are not adjacent
                # Empty slots are -1 in team_slots
                if from_idx < to_idx:  # Pet is moved right
                    # Other pets are pushed left
                    try:
                        empty_idx = 4 - team_slots[::-1].index(-1, from_idx - 5, to_idx - 5)
                    except ValueError:
                        empty_idx = from_idx
                    self.data[item_id_indices[to_idx]] = from_item
                    self.data[item_id_indices[from_idx]] = -1
                    for i in range(to_idx, empty_idx, -1):
                        self.data[item_id_indices[i - 1]] = item_ids[i]
                else:  # Pet is moved left
                    # Other pets are pushed right
                    try:
                        empty_idx = team_slots.index(-1, to_idx, from_idx)
                    except ValueError:
                        empty_idx = from_idx
                    self.data[item_id_indices[to_idx]] = from_item
                    self.data[item_id_indices[from_idx]] = -1
                    for i in range(to_idx, empty_idx):
                        self.data[item_id_indices[i + 1]] = item_ids[i]

    def handle_food(self, idx: int):
        if idx not in range(45, 55):
            return

        buff_indices = [2, 5, 6, 7, 8, 9, 14]
        if self.data[49 if idx < 50 else 50] not in buff_indices:
            # Only handle food that is held
            return

        item_id_indices = [7, 13, 19, 25, 31]
        to_idx = int(decoding_index[idx].split(" to ")[1].replace("Team ", ""))
        self.data[item_id_indices[to_idx]] = self.data[49 if idx < 50 else 50]

    def handle_sell(self, idx: int):
        if idx not in range(55, 60):
            return

        item_id_indices = [7, 13, 19, 25, 31]
        slot = list(range(55, 60)).index(idx)
        self.data[item_id_indices[slot]] = -1

    def handle_buy(self, idx: int):
        if idx not in range(20, 45):
            return

        team_indices = [
            int(t.replace("Team ", "").replace("Shop ", ""))
            for t in decoding_index[idx].split(" to ")
        ]
        team_idx = team_indices[1]
        shop_idx = team_indices[0]
        
        team_slot_indices = [4, 10, 16, 22, 28]
        team_slots = [self.data[i] for i in team_slot_indices]
        item_id_indices = [7, 13, 19, 25, 31]
        item_ids = [self.data[i] for i in item_id_indices]

        team_pet = team_slots[team_idx]
        shop_pet = self.data[34 + 3 * shop_idx]
        
        if team_pet == -1: # Empty destination
            return
        
        if team_pet == shop_pet: # Merged pets
            return
        
        try:
            right_empty = team_slots.index(-1, team_idx)
        except ValueError:
            right_empty = -5
        try:
            left_empty = 4 - team_slots[::-1].index(-1, team_idx - 5)
        except ValueError:
            left_empty = -5
            
        if abs(right_empty - team_idx) <= abs(left_empty - team_idx):
            # Pets are pushed right
            for i in range(team_idx, right_empty):
                self.data[item_id_indices[i + 1]] = item_ids[i]
        else:
            # Pets are pushed left
            for i in range(team_idx, left_empty, -1):
                self.data[item_id_indices[i - 1]] = item_ids[i]
        self.data[item_id_indices[team_idx]] = -1
        

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

    def _reset_data(self, full: bool = False) -> None:
        if full:
            self.data = [-1 for _ in range(51)]
        else:
            # We don't want to reset the data for held items
            stored_indices = [7, 13, 19, 25, 31]
            self.data = [-1 if i not in stored_indices else self.data[i] for i in range(52)]

    def save(self, filepath: str):
        self.ns.save(filepath)
