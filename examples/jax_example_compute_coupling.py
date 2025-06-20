# -*- coding: utf-8 -*-
"""Jax Example compute Gaussian coupling.

Simple example testing the speed improvement from jaxifying the compute coupling function.
When repeat is 1 we see that the code is slower as it needs to compile the function, 
but in the multiple repeats it is significantly faster as it is already compiled.

"""
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from pyelq.coordinate_system import ENU, LLA
from pyelq.dispersion_model.gaussian_plume import GaussianPlume
from pyelq.jax.dispersion_model.gaussian_plume import GaussianPlume as JaxGaussianPlume
from pyelq.gas_species import CH4
from pyelq.meteorology import Meteorology
from pyelq.sensor.beam import Beam
from pyelq.sensor.sensor import Sensor, SensorGroup
from pyelq.source_map import SourceMap

if __name__ == "__main__":
    overall_results_array = []
    for frequency_seconds in [10, 30, 60, 120]:
        time_axis = pd.array(
        pd.date_range(start="2024-01-01 08:00:00", end="2024-01-01 12:00:00", freq=f"{frequency_seconds}s"), dtype="datetime64[ns]"
        )
        nof_observations = time_axis.size
        print(f"Numer of observation: {nof_observations}")
        reference_latitude = 0
        reference_longitude = 0
        reference_altitude = 0

        radius = 30
        angles = np.linspace(0, 90, 5)
        sensor_x = radius * np.cos(angles * np.pi / 180)
        sensor_y = radius * np.sin(angles * np.pi / 180)
        sensor_z = np.ones_like(sensor_x) * 5.0

        ENU_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
        ENU_object.from_array(np.vstack([sensor_x, sensor_y, sensor_z]).T)
        LLA_object = ENU_object.to_lla()
        LLA_array = LLA_object.to_array()

        nof_sensors = LLA_array.shape[0]
        sensor_group = SensorGroup()
        for sensor in range(nof_sensors):
            new_sensor = Beam()
            new_sensor.label = f"Beam sensor {sensor}"
            new_sensor.location = LLA(
                latitude=np.array([reference_latitude, LLA_object.latitude[sensor]]),
                longitude=np.array([reference_longitude, LLA_object.longitude[sensor]]),
                altitude=np.array([5.0, LLA_object.altitude[sensor]]),
            )

            new_sensor.time = time_axis
            new_sensor.concentration = np.zeros(nof_observations)
            sensor_group.add_sensor(new_sensor)

        sensor_x = np.array([5, 20])
        sensor_y = np.array([22, 5])
        sensor_z = np.ones_like(sensor_x) * 1.0
        ENU_object = ENU(ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude)
        ENU_object.from_array(np.vstack([sensor_x, sensor_y, sensor_z]).T)
        LLA_object = ENU_object.to_lla()
        LLA_array = LLA_object.to_array()

        nof_sensors = LLA_array.shape[0]
        for sensor in range(nof_sensors):
            new_sensor = Sensor()
            new_sensor.label = f"Point sensor {sensor}"
            new_sensor.location = LLA(
                latitude=np.array([LLA_object.latitude[sensor]]),
                longitude=np.array([LLA_object.longitude[sensor]]),
                altitude=np.array([LLA_object.altitude[sensor]]),
            )

            new_sensor.time = time_axis
            new_sensor.concentration = np.zeros(nof_observations)
            sensor_group.add_sensor(new_sensor)

        met_object = Meteorology()
        random_generator = np.random.default_rng(0)

        met_object.time = time_axis
        met_object.wind_direction = np.linspace(0.0, 90.0, nof_observations) + random_generator.normal(
            loc=0.0, scale=0.1, size=nof_observations
        )
        met_object.wind_speed = 4.0 * np.ones_like(met_object.wind_direction) + random_generator.normal(
            loc=0.0, scale=0.1, size=nof_observations
        )

        met_object.calculate_uv_from_wind_speed_direction()

        met_object.temperature = (273.1 + 15.0) * np.ones_like(met_object.wind_direction)
        met_object.pressure = 101.325 * np.ones_like(met_object.wind_direction)

        met_object.wind_turbulence_horizontal = 5.0 * np.ones_like(met_object.wind_direction)
        met_object.wind_turbulence_vertical = 5.0 * np.ones_like(met_object.wind_direction)

        for nof_sources in [2, 5, 10]:
            print(f"Number of sources: {nof_sources}")
            source_map = SourceMap()
            site_limits = np.array([[0, 30], [0, 30], [0, 3]])
            location_object = ENU(
                ref_latitude=reference_latitude, ref_longitude=reference_longitude, ref_altitude=reference_altitude
            )

            source_map.generate_sources(
                coordinate_object=location_object, sourcemap_limits=site_limits, sourcemap_type="hypercube", nof_sources=nof_sources
            )

            gas_object = CH4()
            print("Creating Dispersion Model")
            temp_start_time = datetime.datetime.now()
            dispersion_model_original = GaussianPlume(source_map=deepcopy(source_map))
            temp_end_time = datetime.datetime.now()
            print(f"Dispersion model created in {temp_end_time - temp_start_time}")
            temp_start_time = datetime.datetime.now()
            dispersion_model = JaxGaussianPlume(source_map=deepcopy(source_map))
            temp_end_time = datetime.datetime.now()
            print(f"Jax dispersion model created in {temp_end_time - temp_start_time}")
            true_emission_rates = np.array([[15], [10]])
            print("creating observations")


            for nof_repeats in [1, 10, 100]:
                temp_start_time = datetime.datetime.now()
                for _ in range(nof_repeats):
                    for current_sensor in sensor_group.values():
                        coupling_matrix = dispersion_model_original.compute_coupling(
                            sensor_object=current_sensor,
                            meteorology_object=met_object,
                            gas_object=gas_object,
                            output_stacked=False,
                            run_interpolation=False,
                        )
                temp_end_time = datetime.datetime.now()
                normal_time = temp_end_time - temp_start_time
                print(f"Observations created in {normal_time}")

                temp_start_time = datetime.datetime.now()
                for _ in range(nof_repeats):
                    for current_sensor in sensor_group.values():
                        coupling_matrix = dispersion_model.compute_coupling(
                            sensor_object=current_sensor,
                            meteorology_object=met_object,
                            gas_object=gas_object,
                            output_stacked=False,
                            run_interpolation=False,
                        )
                temp_end_time = datetime.datetime.now()
                jax_time = temp_end_time - temp_start_time
                print(f"Jax Observations created in {jax_time}")
                overall_results_array.append((nof_observations, nof_sources, nof_repeats, normal_time.total_seconds(), jax_time.total_seconds()))

    overall_results_df = pd.DataFrame(data=overall_results_array, columns=["# observations", "# sources", "# repeats", "Normal time [s]", "Jax time [s]"])
    print(overall_results_df)
