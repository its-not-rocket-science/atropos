"""Tests for region-specific carbon presets."""

import pytest

from atropos.carbon_presets import (
    CARBON_PRESETS,
    CLOUD_REGION_MAP,
    GLOBAL_AVERAGE,
    compare_regional_impact,
    get_carbon_intensity,
    get_preset,
    get_regional_co2e_savings,
    list_cloud_regions,
    list_regions,
)


class TestGetCarbonIntensity:
    """Tests for get_carbon_intensity function."""

    def test_get_us_carbon_intensity(self):
        """Test getting US carbon intensity."""
        intensity = get_carbon_intensity("US")
        assert intensity == 0.38

    def test_get_germany_carbon_intensity(self):
        """Test getting Germany carbon intensity."""
        intensity = get_carbon_intensity("DE")
        assert intensity == 0.38

    def test_get_france_carbon_intensity(self):
        """Test getting France carbon intensity (low nuclear)."""
        intensity = get_carbon_intensity("FR")
        assert intensity == 0.05

    def test_get_cloud_region(self):
        """Test getting carbon intensity for cloud region."""
        intensity = get_carbon_intensity("us-east-1")
        assert intensity == 0.38  # Maps to US

    def test_get_eu_west_region(self):
        """Test getting carbon intensity for EU region."""
        intensity = get_carbon_intensity("eu-west-1")
        assert intensity == 0.26  # Ireland

    def test_unknown_region_fallback(self):
        """Test fallback to global average for unknown region."""
        intensity = get_carbon_intensity("XX")
        assert intensity == GLOBAL_AVERAGE

    def test_unknown_region_no_fallback(self):
        """Test error for unknown region when fallback disabled."""
        with pytest.raises(ValueError, match="Unknown region"):
            get_carbon_intensity("XX", fallback_to_global=False)

    def test_case_insensitive(self):
        """Test region lookup is case insensitive."""
        assert get_carbon_intensity("us") == get_carbon_intensity("US")
        assert get_carbon_intensity("de") == get_carbon_intensity("DE")


class TestGetPreset:
    """Tests for get_preset function."""

    def test_get_existing_preset(self):
        """Test getting an existing preset."""
        preset = get_preset("US")
        assert preset is not None
        assert preset.region_code == "US"
        assert preset.region_name == "United States"

    def test_get_cloud_region_preset(self):
        """Test getting preset for cloud region."""
        preset = get_preset("us-east-1")
        assert preset is not None
        assert preset.region_code == "US"

    def test_get_unknown_preset(self):
        """Test getting unknown preset returns None."""
        preset = get_preset("XX")
        assert preset is None


class TestListRegions:
    """Tests for list_regions function."""

    def test_list_returns_sorted(self):
        """Test that regions are returned sorted."""
        regions = list_regions()
        assert regions == sorted(regions)

    def test_contains_major_regions(self):
        """Test that major regions are included."""
        regions = list_regions()
        assert "US" in regions
        assert "DE" in regions
        assert "FR" in regions
        assert "GB" in regions


class TestListCloudRegions:
    """Tests for list_cloud_regions function."""

    def test_contains_aws_regions(self):
        """Test that AWS regions are included."""
        cloud_regions = list_cloud_regions()
        assert "us-east-1" in cloud_regions
        assert "eu-west-1" in cloud_regions
        assert "ap-southeast-1" in cloud_regions

    def test_maps_to_country_codes(self):
        """Test that cloud regions map to country codes."""
        cloud_regions = list_cloud_regions()
        assert cloud_regions["us-east-1"] == "US"
        assert cloud_regions["eu-west-2"] == "GB"


class TestGetRegionalCo2eSavings:
    """Tests for get_regional_co2e_savings function."""

    def test_calculate_co2e(self):
        """Test CO2e calculation."""
        # 1000 kWh in US (0.38 kg/kWh) = 380 kg CO2e
        co2e = get_regional_co2e_savings(1000, "US")
        assert co2e == 380.0

    def test_calculate_france_co2e(self):
        """Test CO2e calculation for France (low carbon)."""
        # 1000 kWh in France (0.05 kg/kWh) = 50 kg CO2e
        co2e = get_regional_co2e_savings(1000, "FR")
        assert co2e == 50.0


class TestCompareRegionalImpact:
    """Tests for compare_regional_impact function."""

    def test_compare_multiple_regions(self):
        """Test comparing impact across regions."""
        regions = ["US", "FR", "DE"]
        results = compare_regional_impact(1000, regions)

        assert len(results) == 3
        assert results["US"] == 380.0
        assert results["FR"] == 50.0
        assert results["DE"] == 380.0

    def test_compare_all_regions(self):
        """Test comparing impact across all regions."""
        results = compare_regional_impact(1000)

        assert len(results) == len(CARBON_PRESETS)
        assert "US" in results
        assert "FR" in results


class TestCarbonPresetsData:
    """Tests for carbon preset data."""

    def test_all_intensities_positive(self):
        """Test that all carbon intensities are positive."""
        for code, preset in CARBON_PRESETS.items():
            assert preset.carbon_intensity_kg_per_kwh > 0, f"{code} has non-positive intensity"

    def test_all_intensities_reasonable(self):
        """Test that all carbon intensities are within reasonable range."""
        for code, preset in CARBON_PRESETS.items():
            assert 0 < preset.carbon_intensity_kg_per_kwh < 1.0, (
                f"{code} intensity {preset.carbon_intensity_kg_per_kwh} out of range"
            )

    def test_cloud_region_map_valid(self):
        """Test that all cloud region mappings point to valid presets or fallbacks."""
        for cloud_region, _country_code in CLOUD_REGION_MAP.items():
            # Should be able to get intensity without error
            intensity = get_carbon_intensity(cloud_region)
            assert intensity > 0

    def test_nordic_countries_low_carbon(self):
        """Test that Nordic countries have low carbon intensity."""
        assert CARBON_PRESETS["NO"].carbon_intensity_kg_per_kwh < 0.1
        assert CARBON_PRESETS["SE"].carbon_intensity_kg_per_kwh < 0.1
        assert CARBON_PRESETS["FI"].carbon_intensity_kg_per_kwh < 0.1

    def test_coal_heavy_countries_high_carbon(self):
        """Test that coal-heavy countries have high carbon intensity."""
        assert CARBON_PRESETS["PL"].carbon_intensity_kg_per_kwh > 0.5
        assert CARBON_PRESETS["ZA"].carbon_intensity_kg_per_kwh > 0.8
        assert CARBON_PRESETS["IN"].carbon_intensity_kg_per_kwh > 0.5
