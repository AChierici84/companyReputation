import configparser
from pathlib import Path

class Config:
    def __init__(self, config_file: str = "config.ini"):
        self.config = configparser.ConfigParser()

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {config_file}")

        self.config.read(config_path)

    def get(self, section: str, option: str, fallback=None):
        return self.config.get(section, option, fallback=fallback)

    def get_int(self, section: str, option: str, fallback=None):
        return self.config.getint(section, option, fallback=fallback)

    def get_bool(self, section: str, option: str, fallback=None):
        return self.config.getboolean(section, option, fallback=fallback)

    def get_section(self, section: str) -> dict:
        if section in self.config:
            return dict(self.config[section])
        raise KeyError(f"Sezione '{section}' non trovata nel file di configurazione.")