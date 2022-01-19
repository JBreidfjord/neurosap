# %%
import logging
import os

from neurosap.sap import SAP

# %%
logging.basicConfig(
    filename="logs/program.log",
    level=logging.DEBUG,
    datefmt="%d/%m/%Y %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
if os.path.exists("population.json"):
    sap = SAP.load("population.json")
else:
    sap = SAP(10)

# %%
# Runs a single generation
sap.start()
# %%
sap.save("population.json")
# %%
