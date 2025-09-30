#!/usr/bin/env python3
"""
Multi-Variable ERA5 Climate Data Downloader
==========================================

Downloads comprehensive ERA5 data for detecting ALL climate anomalies:
- Temperature (surface, upper-level)
- Precipitation (total, convective, large-scale)
- Pressure & Circulation (surface, upper-level)
- Wind (10m, 500hPa, 200hPa)
- Humidity & Moisture (dewpoint, relative humidity, water vapor)
- Cloud & Radiation (cloud cover, solar radiation)
- Ocean (SST, wave height)
- Derived Indices (CAPE, vorticity, divergence)
"""

import cdsapi
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class ComprehensiveClimateDownloader:
    """
    Downloads all ERA5 variables needed for comprehensive climate anomaly detection
    """
    
    def __init__(self, api_key=None):
        """Initialize CDS API client"""
        self.client = cdsapi.Client() if api_key is None else cdsapi.Client(key=api_key)
        
        # Complete variable configuration for anomaly detection
        self.variable_config = {
            # SURFACE VARIABLES (Single Levels)
            'surface': {
                'dataset': 'reanalysis-era5-single-levels',
                'variables': {
                    # Temperature
                    '2m_temperature': 't2m',
                    '2m_dewpoint_temperature': 'd2m',
                    'skin_temperature': 'skt',
                    
                    # Precipitation
                    'total_precipitation': 'tp',
                    'convective_precipitation': 'cp', 
                    'large_scale_precipitation': 'lsp',
                    'snowfall': 'sf',
                    'evaporation': 'e',
                    
                    # Pressure
                    'mean_sea_level_pressure': 'msl',
                    'surface_pressure': 'sp',
                    
                    # Wind
                    '10m_u_component_of_wind': 'u10',
                    '10m_v_component_of_wind': 'v10',
                    
                    # Cloud & Radiation
                    'total_cloud_cover': 'tcc',
                    'low_cloud_cover': 'lcc',
                    'medium_cloud_cover': 'mcc',
                    'high_cloud_cover': 'hcc',
                    'surface_solar_radiation_downwards': 'ssrd',
                    'surface_thermal_radiation_downwards': 'strd',
                    'top_net_solar_radiation': 'tsr',
                    'top_net_thermal_radiation': 'ttr',
                    
                    # Moisture & Energy
                    'total_column_water_vapour': 'tcwv',
                    'convective_available_potential_energy': 'cape',
                    'convective_inhibition': 'cin',
                    
                    # Ocean (where available)
                    'sea_surface_temperature': 'sst',
                    'significant_height_of_combined_wind_waves_and_swell': 'swh',
                    'mean_wave_direction': 'mwd',
                    'mean_wave_period': 'mwp'
                }
            },
            
            # PRESSURE LEVEL VARIABLES
            'pressure_levels': {
                'dataset': 'reanalysis-era5-pressure-levels',
                'levels': ['1000', '925', '850', '700', '500', '300', '200', '100'],
                'variables': {
                    # Temperature
                    'temperature': 't',
                    'relative_humidity': 'r',
                    
                    # Wind
                    'u_component_of_wind': 'u',
                    'v_component_of_wind': 'v',
                    'vertical_velocity': 'w',
                    
                    # Geopotential
                    'geopotential': 'z',
                    
                    # Vorticity and Divergence
                    'vorticity': 'vo',
                    'divergence': 'd',
                    
                    # Potential Vorticity (for blocking detection)
                    'potential_vorticity': 'pv'
                }
            },
            
            # LAND VARIABLES (for terrestrial anomalies)
            'land': {
                'dataset': 'reanalysis-era5-land',  
                'variables': {
                    'soil_temperature_level_1': 'stl1',
                    'soil_temperature_level_2': 'stl2',
                    'volumetric_soil_water_layer_1': 'swvl1',
                    'volumetric_soil_water_layer_2': 'swvl2',
                    'leaf_area_index_high_vegetation': 'lai_hv',
                    'leaf_area_index_low_vegetation': 'lai_lv',
                    '2m_temperature': 't2m_land',
                    'total_precipitation': 'tp_land'
                }
            }
        }
        
        print("🌍 Comprehensive Climate Data Downloader Initialized")
        print(f"📊 Surface variables: {len(self.variable_config['surface']['variables'])}")
        print(f"🌪️  Pressure level variables: {len(self.variable_config['pressure_levels']['variables'])}")
        print(f"🌱 Land variables: {len(self.variable_config['land']['variables'])}")
    
    def download_surface_data(self, years, months, region, output_dir):
        """Download all surface variables"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        surface_vars = self.variable_config['surface']['variables']
        
        # Group variables for efficient downloading (CDS API limits)
        var_groups = [
            list(surface_vars.keys())[i:i+10] 
            for i in range(0, len(surface_vars), 10)
        ]
        
        downloaded_files = []
        
        for i, var_group in enumerate(var_groups):
            print(f"📡 Downloading surface variable group {i+1}/{len(var_groups)}")
            print(f"   Variables: {[surface_vars[v] for v in var_group]}")
            
            output_file = output_dir / f"era5_surface_group_{i+1:02d}.nc"
            
            try:
                self.client.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': var_group,
                        'year': [str(y) for y in years],
                        'month': [f"{m:02d}" for m in months],
                        'day': [f"{d:02d}" for d in range(1, 32)],
                        'time': [f"{h:02d}:00" for h in [0, 6, 12, 18]],  # 6-hourly
                        'area': [region['north'], region['west'], region['south'], region['east']],
                        'format': 'netcdf'
                    },
                    str(output_file)
                )
                
                downloaded_files.append(output_file)
                print(f"   ✅ Downloaded: {output_file}")
                
            except Exception as e:
                print(f"   ❌ Error downloading {var_group}: {e}")
        
        return downloaded_files
    
    def download_pressure_level_data(self, years, months, region, output_dir):
        """Download pressure level variables"""
        output_dir = Path(output_dir)  
        output_dir.mkdir(exist_ok=True)
        
        pressure_vars = self.variable_config['pressure_levels']['variables']
        levels = self.variable_config['pressure_levels']['levels']
        
        downloaded_files = []
        
        # Download each variable separately due to size constraints
        for var_name, var_code in pressure_vars.items():
            print(f"🌪️  Downloading pressure level variable: {var_name} ({var_code})")
            
            output_file = output_dir / f"era5_pressure_{var_code}.nc"
            
            try:
                self.client.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': var_name,
                        'pressure_level': levels,
                        'year': [str(y) for y in years],
                        'month': [f"{m:02d}" for m in months],
                        'day': [f"{d:02d}" for d in range(1, 32)],
                        'time': [f"{h:02d}:00" for h in [0, 6, 12, 18]],
                        'area': [region['north'], region['west'], region['south'], region['east']],
                        'format': 'netcdf'
                    },
                    str(output_file)
                )
                
                downloaded_files.append(output_file)
                print(f"   ✅ Downloaded: {output_file}")
                
            except Exception as e:
                print(f"   ❌ Error downloading {var_name}: {e}")
        
        return downloaded_files
    
    def download_land_data(self, years, months, region, output_dir):
        """Download ERA5-Land variables for terrestrial anomalies"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        land_vars = self.variable_config['land']['variables']
        
        downloaded_files = []
        
        # Download land variables in groups
        var_groups = [
            list(land_vars.keys())[i:i+8] 
            for i in range(0, len(land_vars), 8)
        ]
        
        for i, var_group in enumerate(var_groups):
            print(f"🌱 Downloading land variable group {i+1}/{len(var_groups)}")
            
            output_file = output_dir / f"era5_land_group_{i+1:02d}.nc"
            
            try:
                self.client.retrieve(
                    'reanalysis-era5-land',
                    {
                        'variable': var_group,
                        'year': [str(y) for y in years],
                        'month': [f"{m:02d}" for m in months],
                        'day': [f"{d:02d}" for d in range(1, 32)],
                        'time': [f"{h:02d}:00" for h in range(0, 24)],  # Hourly for land
                        'area': [region['north'], region['west'], region['south'], region['east']],
                        'format': 'netcdf'
                    },
                    str(output_file)
                )
                
                downloaded_files.append(output_file)
                print(f"   ✅ Downloaded: {output_file}")
                
            except Exception as e:
                print(f"   ❌ Error downloading land group {i+1}: {e}")
        
        return downloaded_files
    
    def download_comprehensive_dataset(self, start_year=2000, end_year=2024, 
                                     region=None, output_dir="data_comprehensive"):
        """
        Download complete comprehensive climate dataset
        
        Args:
            start_year, end_year: Time range
            region: Geographic bounds
            output_dir: Output directory
        """
        if region is None:
            # Default: Extended European region for comprehensive analysis
            region = {
                'north': 75, 'south': 25,
                'west': -40, 'east': 70
            }
        
        years = list(range(start_year, end_year + 1))
        months = list(range(1, 13))
        
        print("🌍 Starting Comprehensive Climate Data Download")
        print(f"📅 Time range: {start_year}-{end_year} ({len(years)} years)")
        print(f"📍 Region: {region}")
        print(f"💾 Output directory: {output_dir}")
        
        all_downloaded_files = []
        
        # Download surface data
        print("\n" + "="*50)
        print("📡 DOWNLOADING SURFACE DATA")
        print("="*50)
        surface_files = self.download_surface_data(years, months, region, f"{output_dir}/surface")
        all_downloaded_files.extend(surface_files)
        
        # Download pressure level data  
        print("\n" + "="*50)
        print("🌪️  DOWNLOADING PRESSURE LEVEL DATA")
        print("="*50)
        pressure_files = self.download_pressure_level_data(years, months, region, f"{output_dir}/pressure")
        all_downloaded_files.extend(pressure_files)
        
        # Download land data
        print("\n" + "="*50)
        print("🌱 DOWNLOADING LAND DATA")
        print("="*50)
        land_files = self.download_land_data(years, months, region, f"{output_dir}/land")
        all_downloaded_files.extend(land_files)
        
        # Create download summary
        summary = {
            'download_date': datetime.now().isoformat(),
            'time_range': f"{start_year}-{end_year}",
            'region': region,
            'total_files': len(all_downloaded_files),
            'surface_files': len(surface_files),
            'pressure_files': len(pressure_files),
            'land_files': len(land_files),
            'files': [str(f) for f in all_downloaded_files]
        }
        
        summary_file = Path(output_dir) / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Comprehensive Download Complete!")
        print(f"📊 Total files downloaded: {len(all_downloaded_files)}")
        print(f"📋 Summary saved: {summary_file}")
        
        return all_downloaded_files, summary

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Download comprehensive ERA5 climate data for anomaly detection"
    )
    parser.add_argument("--start-year", type=int, default=2000, 
                       help="Start year (default: 2000)")
    parser.add_argument("--end-year", type=int, default=2024,
                       help="End year (default: 2024)")
    parser.add_argument("--region", type=str, default="europe",
                       help="Region: 'europe', 'global', 'north_america', or 'custom'")
    parser.add_argument("--output-dir", type=str, default="data_comprehensive",
                       help="Output directory (default: data_comprehensive)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="CDS API key (optional)")
    
    args = parser.parse_args()
    
    # Define regions
    regions = {
        'global': {'north': 90, 'south': -90, 'west': -180, 'east': 180},
        'europe': {'north': 75, 'south': 25, 'west': -40, 'east': 70},
        'north_america': {'north': 80, 'south': 10, 'west': -170, 'east': -50},
        'asia': {'north': 70, 'south': 0, 'west': 60, 'east': 180}
    }
    
    region = regions.get(args.region, regions['europe'])
    
    # Initialize downloader
    print("🚀 Initializing Comprehensive Climate Data Downloader")
    downloader = ComprehensiveClimateDownloader(api_key=args.api_key)
    
    # Start download
    files, summary = downloader.download_comprehensive_dataset(
        start_year=args.start_year,
        end_year=args.end_year,
        region=region,
        output_dir=args.output_dir
    )
    
    print("\n🎯 Next Steps:")
    print("1. Preprocess downloaded data for anomaly detection")
    print("2. Compute climatologies for each variable")
    print("3. Run comprehensive anomaly detection system")
    print("4. Generate multi-variable anomaly reports")

if __name__ == "__main__":
    main()