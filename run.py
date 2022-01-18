# %%
from neurosap.sap import SAP
import logging

# %%
logging.basicConfig(
    filename="logs/program.log",
    level=logging.DEBUG,
    datefmt="%d/%m/%Y %H:%M:%S",
    format="%(asctime)s - %(message)s",
)
sap = SAP(10)
# %%
# Runs a single generation
sap.start()
# %%
