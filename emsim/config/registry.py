from hydra.core.config_store import ConfigStore
from .app import AppConfig
from .resolvers import register_resolvers


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="base_config", node=AppConfig)

    register_resolvers()
