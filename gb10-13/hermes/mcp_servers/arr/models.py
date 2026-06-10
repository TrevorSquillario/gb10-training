
from dataclasses import dataclass


@dataclass
class SonarrConfig:
	"""Configuration for Sonarr service.

	Attributes:
		base_url: Full base URL to Sonarr API (including scheme, host, port, and base path).
		api_key: API key for Sonarr.
	"""
	base_url: str
	api_key: str


@dataclass
class RadarrConfig:
	"""Configuration for Radarr service.

	Attributes:
		base_url: Full base URL to Radarr API (including scheme, host, port, and base path).
		api_key: API key for Radarr.
	"""
	base_url: str
	api_key: str

