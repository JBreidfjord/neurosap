from neurosap import neurosap
from neurosap.encoding import decode, encode
from neurosap.helpers import display_data, get_game_imgs


class SAP:
    def __init__(self, population_count: int = 10):
        self.population_count = population_count
        self.ns = neurosap.NeuroSAP(population_count)
        self._finished_agents = 0
        self.data = [-1 for _ in range(51)]

    def step(self):
        rgb_img, gray_img = get_game_imgs()
        self._reset_data()
        self.data = encode(rgb_img, gray_img, self.data)
        outputs = self.ns.step(self.data)
        print(decode(outputs))

    def finish_agent(self, fitness: float) -> None:
        self.ns.finish_agent(fitness)
        self._finished_agents += 1

    def evolve(self) -> bool:
        if self._finished_agents == self.population_count:
            self.ns.evolve()
            return True
        else:
            return False

    def display_data(self):
        display_data(self.data)

    def _reset_data(self) -> None:
        # We don't want to reset the data for held items, levels, and experience
        stored_indices = [7, 8, 9, 13, 14, 15, 19, 20, 21, 25, 26, 27, 31, 32, 33]
        self.data = [-1 if i not in stored_indices else self.data[i] for i in range(52)]
