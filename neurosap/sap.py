from neurosap import neurosap
from neurosap.index import encoding_index, food_index, pet_index


class SAP:
    def __init__(self, population_count: int = 10):
        self.population_count = population_count
        self.ns = neurosap.NeuroSAP(population_count)
        self._finished_agents = 0
        self.data = [-1 for _ in range(51)]

    def step(self, inputs: list[float]) -> list[float]:
        return self.ns.step(inputs)

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
        team_id_indices = [4, 10, 16, 22, 28]
        team_att_indices = [5, 11, 17, 23, 29]
        team_hp_indices = [6, 12, 18, 24, 30]
        shop_id_indices = [34, 37, 40, 43, 46]
        shop_att_indices = [35, 38, 41, 44, 47]
        shop_hp_indices = [36, 39, 42, 45, 48]
        food_id_indices = [49, 50]
        team = ["", "", "", "", ""]
        shop = ["", "", "", "", ""]
        food = ["", ""]
        for i, x in enumerate(self.data):
            if x == -1:
                continue

            if i in team_id_indices:
                idx = team_id_indices.index(i)
                team[idx] = pet_index[x]
            elif i in team_att_indices:
                idx = team_att_indices.index(i)
                team[idx] += f" ({x},"
            elif i in team_hp_indices:
                idx = team_hp_indices.index(i)
                team[idx] += f"{x})"
            elif i in shop_id_indices:
                idx = shop_id_indices.index(i)
                shop[idx] = pet_index[x]
            elif i in shop_att_indices:
                idx = shop_att_indices.index(i)
                shop[idx] += f" ({x},"
            elif i in shop_hp_indices:
                idx = shop_hp_indices.index(i)
                shop[idx] += f"{x})"
            elif i in food_id_indices:
                idx = food_id_indices.index(i)
                food[idx] = food_index[x]
            else:
                print(f"{encoding_index[i]}: {x}")
        print(team, shop, food, sep="\n")

    def _reset_data(self) -> None:
        # We don't want to reset the data for held items, levels, and experiments
        stored_indices = [7, 8, 9, 13, 14, 15, 19, 20, 21, 25, 26, 27, 31, 32, 33]
        self.data = [-1 if i not in stored_indices else self.data[i] for i in range(52)]
