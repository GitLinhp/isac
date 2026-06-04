import sionna
from sionna.phy.config import config

if __name__ == "__main__":
    sionna.phy.config.device = "cpu"
    print(config.device)
    print(config.seed)
    print(config.precision)
