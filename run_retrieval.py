import numpy as np
from mpi4py import MPI
import pymultinest
import starry
from astropy.units import R_sun, M_sun, day, R_earth, M_earth
import configparser as cfg
import argparse
import os

starry.config.lazy = False

#======================
# PYMULTINEST VARIABLES
#======================
N_L = 100
evidence_tolerance = 0.8
sampling_efficiency = 'parameter'
multimodal = True

# ------------------
# Parsing arguments.
# ------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(prog='Runs spot parameter retrievals.')

parser.add_argument('-d', '--datafile', type=str, help="Synthetic lightcurve to run retrieval on.")

parser.add_argument('-s', '--starname', type=str, help='Star name')

parser.add_argument('-m', '--modelname', type=str, help='PRISTINE / 1-SPOT / 2-SPOT model to run.')

parser.add_argument('-n', '--nchannels', type=int, help='Number of channels to run retrieval on starting from 0.')

parser.add_argument('-l', '--local', type=str2bool, default=False, help='Running on local device.')

args = parser.parse_args()


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

# -------------------------------
# Creating initial stellar system
# -------------------------------
if args.starname == 'GJ1132':
    stellar_cfg_file = "/references/GJ-1132b/observed_parameters.cfg"
if args.starname == 'TOI540':
    stellar_cfg_file = "/references/TOI-540b/observed_parameters.cfg"
if args.starname == 'TOI5205':
    stellar_cfg_file = "/references/TOI-5205b/observed_parameters.cfg"

stellar_config_file = BASE_DIR + stellar_cfg_file

system_parameters = cfg.ConfigParser()
system_parameters.read(stellar_config_file)
print(stellar_config_file)

r_star = system_parameters.getfloat('STAR', 'radius')
m_star = system_parameters.getfloat('STAR', 'mass')
prot_star = system_parameters.getfloat('STAR', 'prot')

r_planet = system_parameters.getfloat('PLANET', 'radius')
m_planet = system_parameters.getfloat('PLANET', 'mass')
porb_orbit = system_parameters.getfloat('PLANET', 'porb')
inc_orbit = system_parameters.getfloat('PLANET', 'inc')
T14_orbit_hrs = system_parameters.getfloat('PLANET', 'T14')

omega_orbit = 0
ecc_orbit = 0
t0_orbit = 0
w_orbit = 0

# Creating model specific parameter list and 
if args.modelname == 'PRISTINE':
    param_names = ["u1", "u2", "planet_radius"]
if args.modelname == '1-SPOT':
    param_names = ["u1", "u2", "planet_radius","c1", "r1", "lat1", "lon1"]
if args.modelname == '2-SPOT':
    param_names = ["u1", "u2", "planet_radius", "c1", "r1", "lat1", "lon1", "c2", "r2", "lat2", "lon2"]


# ==========================
# CREATING THE FORWARD MODEL
# ==========================
YDEG = 20 # Keeping it around 20 for faster retrievals

chromatic_stellar_surface = starry.Map(ydeg=YDEG, udeg=2)
planetary_surface = starry.Map(ydeg = 0, amp = 5e-3) # keeping the degree 0 as it's uniform brightness

star = starry.Primary(map=chromatic_stellar_surface,
                        r = r_star,
                        m = m_star,
                        prot = prot_star,
                        length_unit = R_sun,
                        mass_unit = M_sun,
                        time_unit = day)

planet = starry.Secondary(map=planetary_surface,
                            r = r_planet,
                            m = m_planet,
                            length_unit = R_earth,
                            mass_unit = M_earth,
                            time_unit = day,
                            porb = porb_orbit,
                            inc = inc_orbit,
                            omega = omega_orbit,
                            ecc = ecc_orbit,
                            t0 = t0_orbit,
                            w = w_orbit)

system = starry.System(star, planet)

# Creating the time array to evaluate the lightcurve
T14_seconds = T14_orbit_hrs*3600

time = np.arange(-T14_seconds, T14_seconds+60, 60)/(3600*24)

def compute_forward_model(theta):
    """
    This function computes the lightcurve for the given set of theta parameters.

    This function only updates the chromatic stellar surface.
    """
    # RESET THE STELLAR SURFACE
    chromatic_stellar_surface[1:, :] = 0.0
    chromatic_stellar_surface.amp = 1.0 # needs this to avoid blowing up the flux values to millions

    params = dict(zip(param_names, theta))

    #=========================
    # CHROMATIC LIMB DARKENING
    #=========================
    chromatic_stellar_surface[1] = params['u1']
    chromatic_stellar_surface[2] = params['u2']

    #=========================
    # CHROMATIC STELLAR RADIUS
    #=========================
    planet.r = params['planet_radius']

    # Adding inhomogeneity 
    if args.modelname == 'PRISTINE':
        return system.flux(t=time)
    
    if args.modelname == '1-SPOT':
        chromatic_stellar_surface.spot(contrast = params['c1'], radius = params['r1'],
                                        lat = params['lat1'], lon = params['lon1'])
        return system.flux(t=time)
    
    if args.modelname == '2-SPOT':
        chromatic_stellar_surface.spot(contrast = params['c1'], radius = params['r1'],
                                        lat = params['lat1'], lon = params['lon1'])
        chromatic_stellar_surface.spot(contrast = params['c2'], radius = params['r2'],
                                        lat = params['lat2'], lon = params['lon2'])
        return system.flux(t=time)

def prior(cube, ndim, nparams):
    """Transforms the unit hypercube cube[0..ndim-1] in [0, 1]

    to the physical parameter space in-place for PyMultiNest.
    """
    # ----------------------------------------------------
    # 1. Quadratic Limb Darkening: Kipping (2013)
    #    Uniform sampling in (q1, q2) -> physical (u1, u2)
    # ----------------------------------------------------
    # Uniform sampling in Q1 and Q2 and then transforing in to u1 and u2 ensures 
    # only physically valid values are LD values are sampled.
    q1 = cube[0]
    q2 = cube[1]
    cube[0] = 2.0 * np.sqrt(q1) * q2  # u1: [0, 2]
    cube[1] = np.sqrt(q1) * (1.0 - 2.0 * q2)  # u2: [-1, 1]

    # ----------------------------------------------------
    # 2. Planetary radius
    #    Uniform sampling between (0.1, 20.1)
    # ----------------------------------------------------

    cube[2] = cube[2]*20+0.1

    if args.modelname == '1-SPOT':
        # ----------------------------------------------------
        # 3. Spot Contrast: c1
        #    0.0 = completely dark spot, 1.0 = unspotted photosphere
        # ----------------------------------------------------
        cube[3] = cube[3] * 1.0  # c1: Uniform [0.0, 1.0]

        # ----------------------------------------------------
        # 4. Spot Angular Radius: r1 (in degrees)
        #    Adjust range depending on expected spot sizes
        # ----------------------------------------------------
        cube[4] = cube[4] * 60.0 + 1.0  # r1: Uniform [1.0, 61.0] deg

        # ----------------------------------------------------
        # 5. Spot Latitude: lat1 (in degrees)
        #    Isotropic sphere prior: Uniform in sin(latitude)
        # ----------------------------------------------------
        sin_lat = 2.0 * cube[5] - 1.0  # Uniform [-1, 1]
        cube[5] = np.degrees(np.arcsin(sin_lat))  # lat1: [-90.0, 90.0] deg

        # ----------------------------------------------------
        # 6. Spot Longitude: lon1 (in degrees)
        #    Uniform across visible/transit longitude span
        # ----------------------------------------------------
        cube[6] = cube[6] * 360.0 - 180.0  # lon1: Uniform [-180.0, 180.0] deg
    
    if args.modelname == '2-SPOT':
        # ----------------------------------------------------
        # 3. Spot Contrast: c1
        #    0.0 = completely dark spot, 1.0 = unspotted photosphere
        # ----------------------------------------------------
        cube[3] = cube[3] * 1.0  # c1: Uniform [0.0, 1.0]

        # ----------------------------------------------------
        # 4. Spot Angular Radius: r1 (in degrees)
        #    Adjust range depending on expected spot sizes
        # ----------------------------------------------------
        cube[4] = cube[4] * 60.0 + 1.0  # r1: Uniform [1.0, 61.0] deg

        # ----------------------------------------------------
        # 5. Spot Latitude: lat1 (in degrees)
        #    Isotropic sphere prior: Uniform in sin(latitude)
        # ----------------------------------------------------
        sin_lat = 2.0 * cube[5] - 1.0  # Uniform [-1, 1]
        cube[5] = np.degrees(np.arcsin(sin_lat))  # lat1: [-90.0, 90.0] deg

        # ----------------------------------------------------
        # 6. Spot Longitude: lon1 (in degrees)
        #    Uniform across visible/transit longitude span
        # ----------------------------------------------------
        cube[6] = cube[6] * 360.0 - 180.0  # lon1: Uniform [-180.0, 180.0] deg

        # ----------------------------------------------------
        # 7. Spot Contrast: c2
        #    0.0 = completely dark spot, 1.0 = unspotted photosphere
        # ----------------------------------------------------
        cube[7] = cube[7] * 1.0  # c2: Uniform [0.0, 1.0]

        # ----------------------------------------------------
        # 8. Spot Angular Radius: r2 (in degrees)
        #    Adjust range depending on expected spot sizes
        # ----------------------------------------------------
        cube[8] = cube[8] * 60.0 + 1.0  # r2: Uniform [1.0, 61.0] deg

        # ----------------------------------------------------
        # 9. Spot Latitude: lat2 (in degrees)
        #    Isotropic sphere prior: Uniform in sin(latitude)
        # ----------------------------------------------------
        sin_lat = 2.0 * cube[9] - 1.0  # Uniform [-1, 1]
        cube[9] = np.degrees(np.arcsin(sin_lat))  # lat2: [-90.0, 90.0] deg

        # ----------------------------------------------------
        # 10. Spot Longitude: lon2 (in degrees)
        #    Uniform across visible/transit longitude span
        # ----------------------------------------------------
        cube[10] = cube[10] * 360.0 - 180.0  # lon2: Uniform [-180.0, 180.0] deg

def loglikelihood(cube, ndim, nparams):
    """
    Computes the log-likelihood function for a given set of parameters.
    """
    theta = [cube[i] for i in range(ndim)]
    model_flux = compute_forward_model(theta)
    residuals = channel_flux - model_flux
    chi2 = np.sum((residuals/channel_flux_err)**2)
    return -0.5 * chi2  # Assuming Gaussian errors, ignoring constant terms

#=================
# READING DATAFILE
#=================
synthetic_flux = np.loadtxt(args.datafile, delimiter=',')
synthetic_flux_err = np.loadtxt(BASE_DIR + "/" + system_parameters['CHROMATIC']['photometric_precision'])

chains_path = "/".join(args.datafile.split("/")[:-1]) + "/chains"
SYN_NUM = int(args.datafile.split("/")[-1].split(".")[0].split("_")[-1])

if rank==0:
    os.makedirs(chains_path, exist_ok=True)
comm.Barrier()

if rank == 0:
    print(f"Running retrieval for {args.modelname}.")

#==========================================
# RUNNING PYMULTINEST ONE CHANNEL AT A TIME
#==========================================

for ch in range(args.nchannels):
    channel_flux = synthetic_flux[ch]
    channel_flux_err = synthetic_flux_err[ch]
    if rank == 0:
        print("============================")
        print(f"Channel number: {ch:03d}.")
        print("============================")

    #====================
    # Running pymultinest
    #====================
    pymultinest.run(
        loglikelihood,
        prior,
        n_dims=len(param_names),
        outputfiles_basename=f"{chains_path}/SYN{SYN_NUM:02d}_{args.modelname}_CH{ch:03d}_",
        n_live_points = N_L,
        sampling_efficiency = sampling_efficiency,
        evidence_tolerance = evidence_tolerance,
        multimodal = multimodal,
        resume = False,
        verbose = (rank == 0),
        init_MPI = False, # MPI has been initialized manually
    )