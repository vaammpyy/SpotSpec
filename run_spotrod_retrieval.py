import numpy as np
from mpi4py import MPI
import pymultinest
import spotrod
from astropy.units import R_sun, M_sun, day, R_earth, M_earth
from astropy.constants import G
import configparser as cfg
import argparse
import os
from scipy.special import ndtri
import glob
import time as t

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(prog='Runs spot parameter retrievals.')

parser.add_argument('-d', '--directory', type=str, help='Directory to perform retrievals on')
parser.add_argument('-l', "--local", type=str2bool, default=False, help='Running on the local device')
parser.add_argument('-s', '--starname', type=str, help="Star name")

args = parser.parse_args()

#==================================
# Planetary radius prior parameters
#==================================
if args.starname == 'GJ-1132':
    planet_radius_mean = 0.04943
    planet_radius_std = 0.00015
if args.starname == 'TOI-540':
    planet_radius_mean = 0.0436
    planet_radius_std = 0.0012
if args.starname == 'TOI-5205':
    planet_radius_log_lower_limit = -5
    planet_radius_log_upper_limit = 0

#======================
# PYMULTINEST VARIABLES
#======================
N_L = 1000
evidence_tolerance = 0.5
sampling_efficiency = 'parameter'
multimodal = True

# ---------
# MPI setup
# ---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------
# BASE_DIR
# ----------
if args.local:
    BASE_DIR = "/home/vampy/acads/projects/Spot_Spectrum_Ariel/Codes/SpotSpec"
else:
    BASE_DIR = "/home/krohan/SpotSpec"

def get_out_of_transit(T14_hours):
    """
    T14 will be in hours.
    """
    T14_seconds = T14_hours * 3600
    time = np.arange(-T14_seconds, T14_seconds, 60) / (3600*24) # in days

    mask = (time < (-T14_seconds/(2*3600*24))) | (time > (T14_seconds/(2*3600*24)))
    return mask

def transform_polar_to_cartesian(spot_lat, spot_lon):
    """
    Transforms spot coordinates from polar to cartesian.

    Parameters
    ----------
    spot_lat : float
        Latitude of the spot.
    spot_long : float
        Longitude of the spot.
    
    Returns
    -------
    (spot_x, spot_y) : (float, float)
        Spot's coordinates in sky-projected cartesian system in R_star units.
    """
    return (np.cos(np.deg2rad(spot_lat))*np.sin(np.deg2rad(spot_lon)), np.sin(np.deg2rad(spot_lat)))

def get_spot_radius(deg):
    """
    Takes in spot radius in degrees and converts it into spot radius in stellar radius units.
    """
    return deg * np.pi/180

def get_orbital_semimajor_axis(period):
    """
    Takes in the orbital period of the planet in days and gives out the semimajor axis of the orbit in stellar radius.

    Parameters
    ----------
    period : float
        Orbital period of the star in days.

    Returns
    -------
    a : float
        Semimajor axis of the planet in R_star.
    """
    numerator = G*(stellar_mass+planet_mass)*period*period
    denominator = 4*np.pi*np.pi
    a = np.cbrt(numerator/denominator).to(stellar_radius)
    return a

BASE_FOLDER = f"{BASE_DIR}/{args.directory}"

forward_model_config_file = f"{BASE_FOLDER}/forward_model.cfg"

forward_model_config = cfg.ConfigParser()
forward_model_config.read(forward_model_config_file)

stellar_parameters_file = forward_model_config['STAR']['stellar_parameters_file']

stellar_config = cfg.ConfigParser()
stellar_config.read(f"{BASE_DIR}{stellar_parameters_file}")

#===============================
# Creating Stellar Planet System
#===============================

stellar_radius = stellar_config.getfloat('STAR', 'radius') * R_sun
stellar_mass = stellar_config.getfloat('STAR', 'mass') * M_sun

porb = stellar_config.getfloat('PLANET', 'porb') * day
# planet_radius = stellar_config.getfloat('PLANET', 'radius') *

planet_mass = stellar_config.getfloat('PLANET', 'mass') * M_earth
planet_semimajor_axis = get_orbital_semimajor_axis(period = porb)
planet_impactparameter = stellar_config.getfloat('PLANET', 'b')
planet_T14 = stellar_config.getfloat('PLANET', 'T14')
planet_T14_minutes = planet_T14 * 60

k = 0
h = 0

# --------------
# TO BE RUN ONCE
# --------------

def quadraticlimbdarkening(r, mu1, mu2):
    answer = np.zeros_like(r)
    mask = r <= 1.0
    oneminusmu = 1.0 - np.sqrt(1.0 - np.power(r[mask], 2))
    answer[mask] = 1.0 - mu1 * oneminusmu - mu2 * np.power(oneminusmu, 2)
    return answer

time = np.arange(-planet_T14_minutes, planet_T14_minutes)/(60*24)

eta, xi = spotrod.elements(time, porb.value, planet_semimajor_axis.value, k, h)
planety = np.ascontiguousarray(planet_impactparameter * eta / planet_semimajor_axis.value)
planetx = np.ascontiguousarray(-xi, dtype=np.float64)

z = np.ascontiguousarray(np.sqrt(np.power(planetx, 2) + np.power(planety,2)), dtype = np.float64)

n = 1000
r = np.ascontiguousarray(np.linspace(1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n))

# This function will be called multiple times by the retrieval code.
def compute_model_LC_spotrod(theta):
    """
    This module computes the model lightcurve using spotrod
    """

    params = dict(zip(param_names, theta))


    #==========
    # LC OFFSET
    #==========
    offset = params['offset']

    #===============
    # LIMB DARKENING
    #===============
    LD_profile = np.ascontiguousarray(quadraticlimbdarkening(r, params['u1'], params['u2']))


    #==============
    # Planet radius
    #==============
    if args.starname == 'GJ-1132' or args.starname == 'TOI-540':
        planet_radius = params['planet_radius']
    if args.starname == 'TOI-5205':
        planet_radius = 10**params['planet_radius']
    planetangle = np.array([spotrod.circleangle(r, planet_radius, z[i]) for i in range(z.size)], dtype=np.float64)

    #===========
    # STAR SPOTS
    #===========
    if modelname == '1-SPOT':
        spot_contrast_starry = params['c1']
        spot_radius_deg = 10**params['r1']
        spot_lat = np.degrees(np.arcsin(params['lat1']))
        spot_lon = np.degrees(np.arcsin(params['lon1']))

        spot_x1, spot_y1 = transform_polar_to_cartesian(spot_lat=spot_lat, spot_lon=spot_lon)
        spot_radius_rstar1 = get_spot_radius(spot_radius_deg)
        spot_contrast_spotrod = 1 - spot_contrast_starry
        # print(f"Spot parameters, SPOTlat{spot_lat}, SPOTlon{spot_lon}, SPOTRad{spot_radius_deg}, SPOTContrast{spot_contrast_starry}")
        # print(f"Spot parameters transformed, SPOTX{spot_x1}, SPOTY{spot_y1}, SPOTRad{spot_radius_rstar1}, SPOTContrast{spot_contrast_spotrod}")

        model_transit = spotrod.integratetransit(
            planetx,
            planety,
            z,
            params['planet_radius'],
            r,
            LD_profile,
            np.array([spot_x1]),
            np.array([spot_y1]),
            np.array([spot_radius_rstar1]),
            np.array([spot_contrast_spotrod]),
            planetangle
        )
        return model_transit-offset

    if modelname == '2-SPOT':
        spot_contrast_starry_1 = params['c1']
        spot_radius_deg_1 = 10**params['r1']
        spot_lat_1 = np.degrees(np.arcsin(params['lat1']))
        spot_lon_1 = np.degrees(np.arcsin(params['lon1']))

        spot_x1, spot_y1 = transform_polar_to_cartesian(spot_lat=spot_lat_1, spot_lon=spot_lon_1)
        spot_radius_rstar1 = get_spot_radius(spot_radius_deg_1)
        spot_contrast_spotrod_1 = 1 - spot_contrast_starry_1

        spot_contrast_starry_2 = params['c2']
        spot_radius_deg_2 = 10**params['r2']
        spot_lat_2 = np.degrees(np.arcsin(params['lat2']))
        spot_lon_2 = np.degrees(np.arcsin(params['lon2']))

        spot_x2, spot_y2 = transform_polar_to_cartesian(spot_lat=spot_lat_2, spot_lon=spot_lon_2)
        spot_radius_rstar2 = get_spot_radius(spot_radius_deg_2)
        spot_contrast_spotrod_2 = 1 - spot_contrast_starry_2

        base_transit = spotrod.integratetransit(
            planetx, planety, z, planet_radius, r, LD_profile,
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            planetangle
        )

        spot_transit1 = spotrod.integratetransit(
            planetx,
            planety,
            z,
            params['planet_radius'],
            r,
            LD_profile,
            np.array([spot_x1], dtype=np.float64),
            np.array([spot_y1], dtype=np.float64),
            np.array([spot_radius_rstar1], dtype=np.float64),
            np.array([spot_contrast_spotrod_1], dtype=np.float64),
            planetangle
        )

        spot_transit2 = spotrod.integratetransit(
            planetx,
            planety,
            z,
            params['planet_radius'],
            r,
            LD_profile,
            np.array([spot_x2], dtype=np.float64),
            np.array([spot_y2], dtype=np.float64),
            np.array([spot_radius_rstar2], dtype=np.float64),
            np.array([spot_contrast_spotrod_2], dtype=np.float64),
            planetangle
        )

        model_transit = base_transit + (spot_transit1 - base_transit) + (spot_transit2 - base_transit)

        return model_transit-offset

def prior(cube, ndim, nparams):
    """Transforms the unit hypercube cube[0..ndim-1] in [0, 1]

    to the physical parameter space in-place for PyMultiNest.
    """
    #-------------------------
    # Offset of the Lightcurve
    #-------------------------
    cube[0] = 0 + PPE*ndtri(cube[0]) # N(0, Photometric Precision Error)

    # ----------------------------------------------------
    # 1. Quadratic Limb Darkening
    # ----------------------------------------------------
    # Sampling from a normal distribution centered at the value from the table
    cube[1] = MEAN_MU1 + 0.07*ndtri(cube[1]) # standard deviation of 0.07 comes from np.std of the U1 coeffs.
    cube[2] = MEAN_MU2 + 0.05*ndtri(cube[2]) # standard deviation of 0.05 comes from np.std of the U2 coeffs.

    # ----------------------------------------------------
    # 2. Planetary radius
    # ----------------------------------------------------
    if args.starname == 'GJ-1132' or args.starname == 'TOI-540':
        cube[3] = planet_radius_mean + planet_radius_std * ndtri(cube[3]) # N(r_mean, r_stddev) here planet radius is in R_planet/R_star
    if args.starname == 'TOI-5205':
        # sampling in log-uniform space for TOI-5205
        cube[3] = planet_radius_log_lower_limit + (planet_radius_log_upper_limit - planet_radius_log_lower_limit) * cube[3]

    if modelname == '1-SPOT':
        # ----------------------------------------------------
        # 3. Spot Contrast: c1, starry contrast
        # ----------------------------------------------------
        cube[4] = -2 + 3*cube[4] # U[-2,1] contrast cannot be greater than 1, it'll result in spot having negative intentsity.

        # ----------------------------------------------------
        # 4. Spot Angular Radius: r1 (in degrees)
        # ----------------------------------------------------
        cube[5] = -1 + 2.95*cube[5] # log_r = U[-1, 1.95]

        # ----------------------------------------------------
        # 5. Spot Latitude: lat1 (in degrees)
        #    Isotropic sphere prior: Uniform in sin(latitude)
        # ----------------------------------------------------
        # sin_lat = cube[6]  # Uniform [0, 1]
        # cube[6] = np.degrees(np.arcsin(sin_lat))  # lat1: [0, 90.0] deg, sampling only in the positive half
        cube[6] = cube[6]

        # ----------------------------------------------------
        # 6. Spot Longitude: lon1 (in degrees)
        #    Uniform across visible/transit longitude span
        # ----------------------------------------------------
        cube[7] = 2 * cube[7] - 1 # Uniform [-1, 1]
        # cube[7] = np.degrees(np.arcsin(sin_lon))  # lon1: [-90, 90.0] deg
    
    if modelname == '2-SPOT':
        # ----------------------------------------------------
        # 3. Spot Contrast: c1, starry contrast
        # ----------------------------------------------------
        cube[4] = -2 + 3*cube[4] # U[-2,1] contrast cannot be greater than 1, it'll result in spot having negative intentsity.

        # ----------------------------------------------------
        # 4. Spot Angular Radius: r1 (in degrees)
        # ----------------------------------------------------
        cube[5] = -1 + 2.95*cube[5] # log_r = U[-1, 1.95]

        # ----------------------------------------------------
        # 5. Spot Latitude: lat1 (in degrees)
        #    Isotropic sphere prior: Uniform in sin(latitude)
        # ----------------------------------------------------
        # sin_lat = cube[6]  # Uniform [0, 1]
        # cube[6] = np.degrees(np.arcsin(sin_lat))  # lat1: [0, 90.0] deg, sampling only in the positive half
        cube[6] = cube[6]

        # ----------------------------------------------------
        # 6. Spot Longitude: lon1 (in degrees)
        #    Uniform across visible/transit longitude span
        # ----------------------------------------------------
        cube[7] = 2 * cube[7] - 1 # Uniform [-1, 1]
        # cube[7] = np.degrees(np.arcsin(sin_lon))  # lon1: [-90, 90.0] deg

        # ----------------------------------------------------
        # 7. Spot Contrast: c1, starry contrast
        # ----------------------------------------------------
        cube[8] = -2 + 3*cube[8] # U[-2,1] contrast cannot be greater than 1, it'll result in spot having negative intentsity.

        # ----------------------------------------------------
        # 9. Spot Angular Radius: r1 (in degrees)
        # ----------------------------------------------------
        cube[9] = -1 + 2.95*cube[9] # log_r = U[-1, 1.95]

        # ----------------------------------------------------
        # 10. Spot Latitude: lat1 (in degrees)
        #    Isotropic sphere prior: Uniform in sin(latitude)
        # ----------------------------------------------------
        cube[10] = cube[10]  # Uniform [0, 1]

        # ----------------------------------------------------
        # 11. Spot Longitude: lon1 (in degrees)
        #    Uniform across visible/transit longitude span
        # ----------------------------------------------------
        cube[11] = 2 * cube[11] -1  # Uniform [-1, 1]

eval_count = 0

def loglikelihood(cube, ndim, nparams):
    """
    Computes the log-likelihood function for a given set of parameters.
    """
    global eval_count
    eval_count += 1
    if eval_count == 1:
        print(f"[Rank {rank}/{size}] Actively evaluating likelihoods!", flush=True)
    theta = [cube[i] for i in range(ndim)]

    u1, u2 = theta[1], theta[2]
    if (u1 + u2 >= 1.0) or (u1 <= 0.0) or (u1 + 2.0 * u2 <= 0.0):
        return -np.inf

    model_flux = compute_model_LC_spotrod(theta)
    residuals = channel_flux - model_flux
    chi2 = np.sum((residuals/channel_flux_err)**2)
    return -0.5 * chi2  # Assuming Gaussian errors, ignoring constant terms

forward_model_directory = f"{BASE_DIR}/{args.directory}"
synthetic_data_files = glob.glob(f"{forward_model_directory}/synthetic*.csv")
modelname_list = np.loadtxt(f"{forward_model_directory}/preffered_model.txt", delimiter=',', dtype=str)

PPE_file_path = f"{BASE_DIR}{stellar_config['CHROMATIC']['photometric_precision']}"
LD_file_path = f"{BASE_DIR}{stellar_config['CHROMATIC']['LD_file']}"

PPE_list = np.loadtxt(PPE_file_path, delimiter=',')
LD = np.loadtxt(LD_file_path, delimiter=',').T

mask_onespot = modelname_list == '1-SPOT'
mask_twospot = modelname_list == '2-SPOT'

chains_path = f"{BASE_FOLDER}/chains"

if rank == 0:
    os.makedirs(chains_path, exist_ok=True)
comm.Barrier()

mask_oot = get_out_of_transit(T14_hours=planet_T14)

if sum(mask_onespot) == 0 and sum(mask_twospot) == 0:
    if rank == 0:
        print("No spot was found!")
else:
    for synthetic_data_file in synthetic_data_files:
        synthetic_observation = np.loadtxt(synthetic_data_file, delimiter=',')
        SYN_NUM = int(synthetic_data_file.split("/")[-1].split(".")[0].split("_")[-1])
        # Only analyzing the first 5 synthetic noise realizations.
        if SYN_NUM<=5:
            continue
        if rank == 0:
            print("****************************")
            print(f"Synthetic Observation: {SYN_NUM:02d}.")
            print("****************************")

        onespot_idx = np.where(mask_onespot)[0]
        # Not fitting any of the Two spot models.
        # twospot_idx = np.where(mask_twospot)[0]

        # for idx in twospot_idx:
        #     modelname = '2-SPOT'
        #     param_names = ['offset', 'u1', 'u2', 'planet_radius', 'c1', 'r1', 'lat1', 'lon1', 'c2', 'r2', 'lat2', 'lon2']

        #     PPE = PPE_list[idx]
        #     MEAN_MU1 = LD[0][idx]
        #     MEAN_MU2 = LD[1][idx]
        #     # if rank==0:
        #     #     print(MEAN_MU1)
        #     #     print(MEAN_MU2)
        #     #     input()

        #     channel_flux = synthetic_observation[idx][:-1]
        #     mean_oot_channel_flux = np.mean(channel_flux[mask_oot])
        #     data_offset = 1 - mean_oot_channel_flux

        #     channel_flux = channel_flux + data_offset
        #     channel_flux_err = PPE
        #     if rank == 0:
        #         print("============================")
        #         print(f"Channel number: {idx:03d}.")
        #         print("============================")
        #     #====================
        #     # Running pymultinest
        #     #====================
        #     start_time = t.time()
        #     pymultinest.run(
        #         loglikelihood,
        #         prior,
        #         n_dims=len(param_names),
        #         outputfiles_basename=f"{chains_path}/SYN{SYN_NUM:02d}_{modelname}_CH{idx:03d}_",
        #         n_live_points = N_L,
        #         sampling_efficiency = sampling_efficiency,
        #         evidence_tolerance = evidence_tolerance,
        #         multimodal = multimodal,
        #         resume = False,
        #         verbose = (rank == 0),
        #         init_MPI = False, # MPI has been initialized manually
        #     )
        #     stop_time = t.time()
        #     if rank == 0:
        #         print(f"MODEL_NAME:{modelname} /\/\/\ TIME TAKEN:{stop_time-start_time:.2f}")

        for idx in onespot_idx:
            modelname = '1-SPOT'
            param_names = ['offset', 'u1', 'u2', 'planet_radius', 'c1', 'r1', 'lat1', 'lon1']

            PPE = PPE_list[idx]
            MEAN_MU1 = LD[0][idx]
            MEAN_MU2 = LD[1][idx]

            channel_flux = synthetic_observation[idx][:-1]
            mean_oot_channel_flux = np.mean(channel_flux[mask_oot])
            data_offset = 1 - mean_oot_channel_flux

            channel_flux = channel_flux + data_offset
            channel_flux_err = PPE
            if rank == 0:
                print("============================")
                print(f"Channel number: {idx:03d}.")
                print("============================")
            #====================
            # Running pymultinest
            #====================
            start_time = t.time()
            pymultinest.run(
                loglikelihood,
                prior,
                n_dims=len(param_names),
                outputfiles_basename=f"{chains_path}/SYN{SYN_NUM:02d}_{modelname}_CH{idx:03d}_",
                n_live_points = N_L,
                sampling_efficiency = sampling_efficiency,
                evidence_tolerance = evidence_tolerance,
                multimodal = multimodal,
                resume = False,
                verbose = (rank == 0),
                init_MPI = False, # MPI has been initialized manually
            )
            stop_time = t.time()
            if rank == 0:
                print(f"MODEL_NAME:{modelname} /\/\/\ TIME TAKEN:{stop_time-start_time:.2f}")
