"""Region-specific grid carbon intensity presets for CO2e calculations.

Data sources:
- Ember Climate Electricity Data Explorer (2023/2024)
- IEA Electricity Market Report
- electricitymaps.com grid intensity data

Values are in kg CO2e per kWh (lifecycle emissions including upstream methane).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class GridCarbonPreset:
    """Grid carbon intensity preset for a region.

    Attributes:
        region_code: ISO 3166-1 alpha-2 country code.
        region_name: Human-readable region name.
        carbon_intensity_kg_per_kwh: Grid carbon intensity in kg CO2e per kWh.
        data_year: Year of the carbon intensity data.
        source: Data source reference.
        notes: Additional context about the region's grid.
    """

    region_code: str
    region_name: str
    carbon_intensity_kg_per_kwh: float
    data_year: int = 2023
    source: str = "Ember/IEA"
    notes: str = ""


# Global average (used as fallback)
GLOBAL_AVERAGE = 0.35

# Regional presets based on latest available data
CARBON_PRESETS: dict[str, GridCarbonPreset] = {
    # North America
    "US": GridCarbonPreset(
        region_code="US",
        region_name="United States",
        carbon_intensity_kg_per_kwh=0.38,
        data_year=2023,
        notes="Varies significantly by state (e.g., Vermont ~0.02, Wyoming ~0.75)",
    ),
    "CA": GridCarbonPreset(
        region_code="CA",
        region_name="Canada",
        carbon_intensity_kg_per_kwh=0.13,
        data_year=2023,
        notes="High hydroelectric share, varies by province",
    ),
    "MX": GridCarbonPreset(
        region_code="MX",
        region_name="Mexico",
        carbon_intensity_kg_per_kwh=0.41,
        data_year=2023,
    ),
    # Europe
    "DE": GridCarbonPreset(
        region_code="DE",
        region_name="Germany",
        carbon_intensity_kg_per_kwh=0.38,
        data_year=2023,
        notes="Coal phase-out in progress, high renewables variability",
    ),
    "FR": GridCarbonPreset(
        region_code="FR",
        region_name="France",
        carbon_intensity_kg_per_kwh=0.05,
        data_year=2023,
        notes="High nuclear share (~70%)",
    ),
    "GB": GridCarbonPreset(
        region_code="GB",
        region_name="United Kingdom",
        carbon_intensity_kg_per_kwh=0.21,
        data_year=2023,
        notes="Rapid coal phase-out, high offshore wind",
    ),
    "NO": GridCarbonPreset(
        region_code="NO",
        region_name="Norway",
        carbon_intensity_kg_per_kwh=0.02,
        data_year=2023,
        notes="Almost entirely hydroelectric (~95%)",
    ),
    "SE": GridCarbonPreset(
        region_code="SE",
        region_name="Sweden",
        carbon_intensity_kg_per_kwh=0.01,
        data_year=2023,
        notes="Hydro + nuclear (~95% low-carbon)",
    ),
    "FI": GridCarbonPreset(
        region_code="FI",
        region_name="Finland",
        carbon_intensity_kg_per_kwh=0.08,
        data_year=2023,
        notes="Nuclear + hydro + bioenergy",
    ),
    "DK": GridCarbonPreset(
        region_code="DK",
        region_name="Denmark",
        carbon_intensity_kg_per_kwh=0.12,
        data_year=2023,
        notes="High wind penetration (~55%)",
    ),
    "NL": GridCarbonPreset(
        region_code="NL",
        region_name="Netherlands",
        carbon_intensity_kg_per_kwh=0.24,
        data_year=2023,
    ),
    "BE": GridCarbonPreset(
        region_code="BE",
        region_name="Belgium",
        carbon_intensity_kg_per_kwh=0.12,
        data_year=2023,
        notes="Nuclear phase-out transitioning to gas + renewables",
    ),
    "ES": GridCarbonPreset(
        region_code="ES",
        region_name="Spain",
        carbon_intensity_kg_per_kwh=0.17,
        data_year=2023,
        notes="High wind and solar penetration",
    ),
    "IT": GridCarbonPreset(
        region_code="IT",
        region_name="Italy",
        carbon_intensity_kg_per_kwh=0.33,
        data_year=2023,
    ),
    "PL": GridCarbonPreset(
        region_code="PL",
        region_name="Poland",
        carbon_intensity_kg_per_kwh=0.66,
        data_year=2023,
        notes="High coal dependency",
    ),
    "EU": GridCarbonPreset(
        region_code="EU",
        region_name="European Union (average)",
        carbon_intensity_kg_per_kwh=0.23,
        data_year=2023,
        notes="Weighted average across EU-27",
    ),
    "IE": GridCarbonPreset(
        region_code="IE",
        region_name="Ireland",
        carbon_intensity_kg_per_kwh=0.26,
        data_year=2023,
        notes="High wind penetration but gas backup needed",
    ),
    "CH": GridCarbonPreset(
        region_code="CH",
        region_name="Switzerland",
        carbon_intensity_kg_per_kwh=0.02,
        data_year=2023,
        notes="Hydro and nuclear dominant",
    ),
    # Asia-Pacific
    "CN": GridCarbonPreset(
        region_code="CN",
        region_name="China",
        carbon_intensity_kg_per_kwh=0.55,
        data_year=2023,
        notes="Rapid renewables deployment but still coal-heavy (~60%)",
    ),
    "IN": GridCarbonPreset(
        region_code="IN",
        region_name="India",
        carbon_intensity_kg_per_kwh=0.71,
        data_year=2023,
        notes="High coal dependency (~75%)",
    ),
    "JP": GridCarbonPreset(
        region_code="JP",
        region_name="Japan",
        carbon_intensity_kg_per_kwh=0.49,
        data_year=2023,
        notes="Post-Fukushima LNG and coal increase",
    ),
    "KR": GridCarbonPreset(
        region_code="KR",
        region_name="South Korea",
        carbon_intensity_kg_per_kwh=0.44,
        data_year=2023,
    ),
    "AU": GridCarbonPreset(
        region_code="AU",
        region_name="Australia",
        carbon_intensity_kg_per_kwh=0.56,
        data_year=2023,
        notes="Coal-heavy (~60%), high solar penetration in some states",
    ),
    "NZ": GridCarbonPreset(
        region_code="NZ",
        region_name="New Zealand",
        carbon_intensity_kg_per_kwh=0.08,
        data_year=2023,
        notes="High hydro + geothermal (~85% renewable)",
    ),
    "SG": GridCarbonPreset(
        region_code="SG",
        region_name="Singapore",
        carbon_intensity_kg_per_kwh=0.40,
        data_year=2023,
        notes="Natural gas dominant, solar deployment limited by land",
    ),
    # South America
    "BR": GridCarbonPreset(
        region_code="BR",
        region_name="Brazil",
        carbon_intensity_kg_per_kwh=0.12,
        data_year=2023,
        notes="High hydro share (~65%), varies with rainfall",
    ),
    "CL": GridCarbonPreset(
        region_code="CL",
        region_name="Chile",
        carbon_intensity_kg_per_kwh=0.29,
        data_year=2023,
        notes="Northern grid coal-heavy, southern hydro-heavy",
    ),
    "AR": GridCarbonPreset(
        region_code="AR",
        region_name="Argentina",
        carbon_intensity_kg_per_kwh=0.35,
        data_year=2023,
    ),
    # Middle East & Africa
    "ZA": GridCarbonPreset(
        region_code="ZA",
        region_name="South Africa",
        carbon_intensity_kg_per_kwh=0.90,
        data_year=2023,
        notes="Extremely coal-heavy (~85%), severe load shedding",
    ),
    "NG": GridCarbonPreset(
        region_code="NG",
        region_name="Nigeria",
        carbon_intensity_kg_per_kwh=0.50,
        data_year=2023,
        notes="Gas + diesel generators, unreliable grid",
    ),
    "EG": GridCarbonPreset(
        region_code="EG",
        region_name="Egypt",
        carbon_intensity_kg_per_kwh=0.43,
        data_year=2023,
        notes="Natural gas dominant",
    ),
    "SA": GridCarbonPreset(
        region_code="SA",
        region_name="Saudi Arabia",
        carbon_intensity_kg_per_kwh=0.60,
        data_year=2023,
        notes="Oil and gas dominant",
    ),
    "AE": GridCarbonPreset(
        region_code="AE",
        region_name="United Arab Emirates",
        carbon_intensity_kg_per_kwh=0.45,
        data_year=2023,
        notes="Gas + growing nuclear/solar",
    ),
}

# Cloud provider region mappings (approximate based on primary location)
CLOUD_REGION_MAP: dict[str, str] = {
    # AWS US regions
    "us-east-1": "US",  # N. Virginia
    "us-east-2": "US",  # Ohio
    "us-west-1": "US",  # N. California
    "us-west-2": "US",  # Oregon
    "us-gov-east-1": "US",
    "us-gov-west-1": "US",
    # AWS Canada
    "ca-central-1": "CA",
    # AWS Europe
    "eu-west-1": "IE",  # Ireland
    "eu-west-2": "GB",  # London
    "eu-west-3": "FR",  # Paris
    "eu-central-1": "DE",  # Frankfurt
    "eu-central-2": "CH",  # Zurich (use EU avg)
    "eu-north-1": "SE",  # Stockholm
    "eu-south-1": "IT",  # Milan
    "eu-south-2": "ES",  # Spain
    # AWS Asia-Pacific
    "ap-northeast-1": "JP",  # Tokyo
    "ap-northeast-2": "KR",  # Seoul
    "ap-northeast-3": "JP",  # Osaka
    "ap-southeast-1": "SG",  # Singapore
    "ap-southeast-2": "AU",  # Sydney
    "ap-southeast-3": "ID",  # Jakarta (use SE Asia avg ~0.55)
    "ap-southeast-4": "AU",  # Melbourne
    "ap-south-1": "IN",  # Mumbai
    "ap-south-2": "IN",  # Hyderabad
    "ap-east-1": "HK",  # Hong Kong (use CN ~0.55)
    # AWS South America
    "sa-east-1": "BR",  # São Paulo
    # AWS Middle East
    "me-south-1": "BH",  # Bahrain (use ~0.55)
    "me-central-1": "AE",  # UAE
    # AWS Africa
    "af-south-1": "ZA",  # South Africa
    # GCP regions (examples)
    "us-central1": "US",
    "europe-west1": "BE",  # Belgium
    "europe-west2": "GB",
    "europe-west3": "DE",
    "europe-west4": "NL",
    "asia-east1": "TW",  # Taiwan (use ~0.50)
    "asia-northeast1": "JP",
    "asia-southeast1": "SG",
    "australia-southeast1": "AU",
    # Azure regions (examples)
    "eastus": "US",
    "westeurope": "NL",
    "northeurope": "IE",
    "uksouth": "GB",
    "francecentral": "FR",
    "germanywestcentral": "DE",
    "southeastasia": "SG",
    "japaneast": "JP",
    "australiaeast": "AU",
    "brazilsouth": "BR",
    "centralindia": "IN",
}

# Fallback for regions not in main presets
FALLBACK_PRESETS: dict[str, float] = {
    "IE": 0.26,  # Ireland
    "CH": 0.02,  # Switzerland (mostly hydro/nuclear)
    "ID": 0.62,  # Indonesia (coal-heavy)
    "HK": 0.55,  # Hong Kong
    "BH": 0.55,  # Bahrain
    "TW": 0.50,  # Taiwan
}


def get_carbon_intensity(
    region: str,
    fallback_to_global: bool = True,
) -> float:
    """Get carbon intensity for a region.

    Args:
        region: Region identifier (ISO country code, cloud region name,
               or region name like 'US', 'EU', 'DE', 'us-east-1').
        fallback_to_global: If True, return global average for unknown regions.
                           If False, raise ValueError.

    Returns:
        Carbon intensity in kg CO2e per kWh.

    Raises:
        ValueError: If region not found and fallback_to_global is False.

    Examples:
        >>> get_carbon_intensity("US")
        0.38
        >>> get_carbon_intensity("us-east-1")  # AWS region
        0.38
        >>> get_carbon_intensity("DE")
        0.38
    """
    # Normalize region code (but keep original for cloud lookup)
    region_upper = region.upper()

    # Direct lookup in presets
    if region_upper in CARBON_PRESETS:
        return CARBON_PRESETS[region_upper].carbon_intensity_kg_per_kwh

    # Check cloud region mapping (case-insensitive)
    cloud_region = region.lower()
    if cloud_region in CLOUD_REGION_MAP:
        country_code = CLOUD_REGION_MAP[cloud_region]
        if country_code in CARBON_PRESETS:
            return CARBON_PRESETS[country_code].carbon_intensity_kg_per_kwh
        if country_code in FALLBACK_PRESETS:
            return FALLBACK_PRESETS[country_code]

    # Check fallback presets
    if region_upper in FALLBACK_PRESETS:
        return FALLBACK_PRESETS[region_upper]

    if fallback_to_global:
        return GLOBAL_AVERAGE

    raise ValueError(
        f"Unknown region '{region}'. "
        f"Use ISO country code (e.g., 'US', 'DE') or cloud region (e.g., 'us-east-1'). "
        f"Available countries: {list(CARBON_PRESETS.keys())}"
    )


def get_preset(region: str) -> GridCarbonPreset | None:
    """Get the full preset for a region.

    Args:
        region: Region identifier (ISO country code or cloud region).

    Returns:
        GridCarbonPreset if found, None otherwise.
    """
    region_upper = region.upper()

    if region_upper in CARBON_PRESETS:
        return CARBON_PRESETS[region_upper]

    # Check cloud region mapping (case-insensitive)
    cloud_region = region.lower()
    if cloud_region in CLOUD_REGION_MAP:
        country_code = CLOUD_REGION_MAP[cloud_region]
        if country_code in CARBON_PRESETS:
            return CARBON_PRESETS[country_code]

    return None


def list_regions() -> list[str]:
    """List all available region codes.

    Returns:
        Sorted list of available ISO country codes.
    """
    return sorted(CARBON_PRESETS.keys())


def list_cloud_regions() -> dict[str, str]:
    """List all mapped cloud regions.

    Returns:
        Dictionary mapping cloud region to country code.
    """
    return dict(sorted(CLOUD_REGION_MAP.items()))


def get_regional_co2e_savings(
    annual_energy_kwh: float,
    region: str,
) -> float:
    """Calculate annual CO2e savings for a region.

    Args:
        annual_energy_kwh: Annual energy consumption in kWh.
        region: Region identifier.

    Returns:
        Annual CO2e in kg.
    """
    intensity = get_carbon_intensity(region)
    return annual_energy_kwh * intensity


def compare_regional_impact(
    annual_energy_kwh: float,
    regions: list[str] | None = None,
) -> dict[str, float]:
    """Compare CO2e impact across multiple regions.

    Args:
        annual_energy_kwh: Annual energy consumption in kWh.
        regions: List of region codes to compare. If None, uses all presets.

    Returns:
        Dictionary mapping region code to annual CO2e in kg.
    """
    if regions is None:
        regions = list(CARBON_PRESETS.keys())

    return {region: get_regional_co2e_savings(annual_energy_kwh, region) for region in regions}
