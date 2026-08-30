import numpy as np
import matplotlib.pyplot as plt
import starry
from forward_model import make_star, make_planet, make_star_planet_system, generate_lightcurve
from config_utils import config_reader
import glob
from collections.abc import Sequence
import argparse
import configparser as cfg
from pathlib import Path
import os

starry.config.lazy = False

def generate_forward_model(parameter_dict):
    """
    Generates the chromatic forward model with given parameter_dict.

    Parameters
    ----------
    parameter_dict: dict
        Dictionary of all the relevant stellar and planetary parameters.
    
    Returns
    -------
    system_list: list
        Chromatic list of starry systems.
    """
    nw = parameter_dict['star']['nw']
    amp = np.ones(nw)

    n_inhom = len(parameter_dict['inhomogeneties']['radius'])

    if n_inhom == 1:

        onespot_map_list = make_chromatic_stellar_surface_for_model_comparison(ydeg = 30,
                                                    limb_darkening = parameter_dict['star']['limb_darkening'],
                                                    amp = amp,
                                                    nw = nw,
                                                    rv = False,
                                                    contrast_list = parameter_dict['inhomogeneties']['contrast_list'],
                                                    radius = parameter_dict['inhomogeneties']['radius'],
                                                    lat = parameter_dict['inhomogeneties']['latitude'],
                                                    lon = parameter_dict['inhomogeneties']['longitude'])
        
        star_list = make_star(maps = onespot_map_list,
                                r = parameter_dict['star']['radius'],
                                m = parameter_dict['star']['mass'],
                                prot = parameter_dict['star']['prot'])
        
        planet_list = make_planet(planet_surface = starry.Map(ydeg=5, amp=5e-3),
                                    radius = parameter_dict['planet']['radius'],
                                    mass = parameter_dict['planet']['mass'],
                                    porb = parameter_dict['planet']['porb'],
                                    prot = parameter_dict['planet']['prot'],
                                    Omega = parameter_dict['planet']['Omega'],
                                    ecc = parameter_dict['planet']['ecc'],
                                    w = parameter_dict['planet']['w'],
                                    t0 = parameter_dict['planet']['t0'],
                                    inc = parameter_dict['planet']['inc'])
        
        onespot_system_list = make_star_planet_system(star_list=star_list, planet_list=planet_list)

        pristine_map_list = make_chromatic_stellar_surface_for_model_comparison(ydeg = 30,
                                                    limb_darkening = parameter_dict['star']['limb_darkening'],
                                                    amp = amp,
                                                    nw = nw,
                                                    contrast_list = parameter_dict['inhomogeneties']['contrast_list'],
                                                    rv = False,
                                                    unspotted = True)
        
        star_list = make_star(maps = pristine_map_list,
                                r = parameter_dict['star']['radius'],
                                m = parameter_dict['star']['mass'],
                                prot = parameter_dict['star']['prot'])
        
        planet_list = make_planet(planet_surface = starry.Map(ydeg=5, amp=5e-3),
                                    radius = parameter_dict['planet']['radius'],
                                    mass = parameter_dict['planet']['mass'],
                                    porb = parameter_dict['planet']['porb'],
                                    prot = parameter_dict['planet']['prot'],
                                    Omega = parameter_dict['planet']['Omega'],
                                    ecc = parameter_dict['planet']['ecc'],
                                    w = parameter_dict['planet']['w'],
                                    t0 = parameter_dict['planet']['t0'],
                                    inc = parameter_dict['planet']['inc'])
        
        pristine_system_list = make_star_planet_system(star_list=star_list, planet_list=planet_list)

        return [pristine_system_list, onespot_system_list]

    if n_inhom == 2:

        firstspot_map_list = make_chromatic_stellar_surface_for_model_comparison(ydeg = 30,
                                                    limb_darkening = parameter_dict['star']['limb_darkening'],
                                                    amp = amp,
                                                    nw = nw,
                                                    rv = False,
                                                    contrast_list = parameter_dict['inhomogeneties']['contrast_list'],
                                                    radius = [parameter_dict['inhomogeneties']['radius'][0]],
                                                    lat = [parameter_dict['inhomogeneties']['latitude'][0]],
                                                    lon = [parameter_dict['inhomogeneties']['longitude'][0]],
                                                    twospotted=True,
                                                    spotindex=0)
        
        star_list = make_star(maps = firstspot_map_list,
                                r = parameter_dict['star']['radius'],
                                m = parameter_dict['star']['mass'],
                                prot = parameter_dict['star']['prot'])
        
        planet_list = make_planet(planet_surface = starry.Map(ydeg=5, amp=5e-3),
                                    radius = parameter_dict['planet']['radius'],
                                    mass = parameter_dict['planet']['mass'],
                                    porb = parameter_dict['planet']['porb'],
                                    prot = parameter_dict['planet']['prot'],
                                    Omega = parameter_dict['planet']['Omega'],
                                    ecc = parameter_dict['planet']['ecc'],
                                    w = parameter_dict['planet']['w'],
                                    t0 = parameter_dict['planet']['t0'],
                                    inc = parameter_dict['planet']['inc'])
        
        firstspot_system_list = make_star_planet_system(star_list=star_list, planet_list=planet_list)

        secondspot_map_list = make_chromatic_stellar_surface_for_model_comparison(ydeg = 30,
                                                    limb_darkening = parameter_dict['star']['limb_darkening'],
                                                    amp = amp,
                                                    nw = nw,
                                                    rv = False,
                                                    contrast_list = parameter_dict['inhomogeneties']['contrast_list'],
                                                    radius = [parameter_dict['inhomogeneties']['radius'][1]],
                                                    lat = [parameter_dict['inhomogeneties']['latitude'][1]],
                                                    lon = [parameter_dict['inhomogeneties']['longitude'][1]],
                                                    twospotted=True,
                                                    spotindex=1)
        
        star_list = make_star(maps = secondspot_map_list,
                                r = parameter_dict['star']['radius'],
                                m = parameter_dict['star']['mass'],
                                prot = parameter_dict['star']['prot'])
        
        planet_list = make_planet(planet_surface = starry.Map(ydeg=5, amp=5e-3),
                                    radius = parameter_dict['planet']['radius'],
                                    mass = parameter_dict['planet']['mass'],
                                    porb = parameter_dict['planet']['porb'],
                                    prot = parameter_dict['planet']['prot'],
                                    Omega = parameter_dict['planet']['Omega'],
                                    ecc = parameter_dict['planet']['ecc'],
                                    w = parameter_dict['planet']['w'],
                                    t0 = parameter_dict['planet']['t0'],
                                    inc = parameter_dict['planet']['inc'])
        
        secondspot_system_list = make_star_planet_system(star_list=star_list, planet_list=planet_list)

        bothspot_map_list = make_chromatic_stellar_surface_for_model_comparison(ydeg = 30,
                                                    limb_darkening = parameter_dict['star']['limb_darkening'],
                                                    amp = amp,
                                                    nw = nw,
                                                    rv = False,
                                                    contrast_list = parameter_dict['inhomogeneties']['contrast_list'],
                                                    radius = parameter_dict['inhomogeneties']['radius'],
                                                    lat = parameter_dict['inhomogeneties']['latitude'],
                                                    lon = parameter_dict['inhomogeneties']['longitude'])
        
        star_list = make_star(maps = bothspot_map_list,
                                r = parameter_dict['star']['radius'],
                                m = parameter_dict['star']['mass'],
                                prot = parameter_dict['star']['prot'])
        
        planet_list = make_planet(planet_surface = starry.Map(ydeg=5, amp=5e-3),
                                    radius = parameter_dict['planet']['radius'],
                                    mass = parameter_dict['planet']['mass'],
                                    porb = parameter_dict['planet']['porb'],
                                    prot = parameter_dict['planet']['prot'],
                                    Omega = parameter_dict['planet']['Omega'],
                                    ecc = parameter_dict['planet']['ecc'],
                                    w = parameter_dict['planet']['w'],
                                    t0 = parameter_dict['planet']['t0'],
                                    inc = parameter_dict['planet']['inc'])
        
        bothspot_system_list = make_star_planet_system(star_list=star_list, planet_list=planet_list)

        pristine_map_list = make_chromatic_stellar_surface_for_model_comparison(ydeg = 30,
                                                    limb_darkening = parameter_dict['star']['limb_darkening'],
                                                    amp = amp,
                                                    nw = nw,
                                                    contrast_list = parameter_dict['inhomogeneties']['contrast_list'],
                                                    rv = False,
                                                    unspotted = True)
        
        star_list = make_star(maps = pristine_map_list,
                                r = parameter_dict['star']['radius'],
                                m = parameter_dict['star']['mass'],
                                prot = parameter_dict['star']['prot'])
        
        planet_list = make_planet(planet_surface = starry.Map(ydeg=5, amp=5e-3),
                                    radius = parameter_dict['planet']['radius'],
                                    mass = parameter_dict['planet']['mass'],
                                    porb = parameter_dict['planet']['porb'],
                                    prot = parameter_dict['planet']['prot'],
                                    Omega = parameter_dict['planet']['Omega'],
                                    ecc = parameter_dict['planet']['ecc'],
                                    w = parameter_dict['planet']['w'],
                                    t0 = parameter_dict['planet']['t0'],
                                    inc = parameter_dict['planet']['inc'])
        
        pristine_system_list = make_star_planet_system(star_list=star_list, planet_list=planet_list)

        return [pristine_system_list, firstspot_system_list, secondspot_system_list, bothspot_system_list]

def generate_forward_model_array(config_file_path):
    """
    Generates the forward models.

    Runs the forward model with a given set of parameters.

    Parameters
    ----------
    config_file_path : str
        Path to config file.
    generate_animation : bool, optional
        Whether to generate an animation of the system. Default is False.
    """
    parameter_dict = config_reader(config_file_path)
    system_lists = generate_forward_model(parameter_dict=parameter_dict)

    # generating the lightcurve
    T14_hrs = parameter_dict['planet']['T14'] # in hours
    T14_seconds = T14_hrs*3600

    time = np.arange(-T14_seconds, T14_seconds+60, 60)/(3600*24)
    lightcurve_lists = [generate_lightcurve(system_list=system_lists[i], time=time) for i in range(len(system_lists))]

    return lightcurve_lists

def compute_log_likelihood(data, data_err, model):

    data_err_2D = np.repeat(data_err[:, np.newaxis], len(data[0]), axis=1)

    residuals = data - model
    chi2 = np.sum((residuals/data_err_2D)**2, axis=1)
    return -0.5 * chi2

def compute_BIC(k, data, data_err, model):
    "I want to compute for the entire 2D lightcurve at once"
    BIC = k*np.log(len(data[0])) - 2*compute_log_likelihood(data, data_err, model)
    return BIC

def get_out_of_transit(T14_hours):
    """
    T14 will be in hours.
    """
    T14_seconds = T14_hours * 3600
    time = np.arange(-T14_seconds, T14_seconds+60, 60) / (3600*24) # in days

    mask = (time < (-T14_seconds/(2*3600*24))) | (time > (T14_seconds/(2*3600*24)))
    return mask

def make_chromatic_stellar_surface_for_model_comparison(
ydeg: int = 30, 
limb_darkening: Sequence[[int, Sequence[float], Sequence[float]]] = (2, (0.5, ), (0.25, )), 
amp: Sequence[float] = (1,),
nw: int = 1, 
rv: bool = False, 
contrast_list: Sequence[Sequence[float]] = ((0.5,),), 
radius: Sequence[float] = (20.0,), 
lat: Sequence[float] = (20.0,), 
lon: Sequence[float] = (20.0,),
unspotted: bool = False,
twospotted: bool = False,
spotindex: int = 0,
):
    """
    Makes a chromatic stelllar surface stellar surface.

    Makes a chromatic stellar surface with star spots.

    Parameters
    ----------
    ydeg : int
        Degree of the spherical harmonic expansion.
    limb_darkening : Sequence[int, Sequence[float], Sequence[float]]
        [<Polynomial degree of limb darkening>, <chromatic u1 parameter list>, <chromatic u2 parameter list>]
    amp : Sequence[float]
        Absolute flux value in the wavelength bin. Used for flux calibration.
    nw : int
        Number of wavelength bins
    rv : bool
        True will turn on radial velocity calculation.
    contrast_list : Sequence[Sequence[float]]
        List of chromatic spot contrast, each array element contains constrast of all the element at a given wavelength.
    radius : Sequence[float]
        List of spot radius.
    lat : Sequence[float]
        List of spot latitude.
    lon : Sequence[float]
        List of spot longitude

    Returns
    -------
    map_list : list(starry.Maps.YlmBase)
        List of stellar surface map, each entry corresponds to map at a given wavelength.
    """
    # chromatic limb darkening is not implemented directly in starry.Maps,
    # also given how normalization works, we need to be careful with flux calibration.
    # amp parameter can deal with this, it's a factor which can be directly multiplied to give the absolute values
    # value of amp is the value in the wavelength bin according to the doc.
    map_list=[]
    # iterating over the wavelength
    for j in range(len(contrast_list)):
        map = starry.Map(ydeg=ydeg, udeg=limb_darkening[0], rv=rv, amp=amp[j])

        # setting up the coefficients for the limb_darkening
        map[1] = limb_darkening[1][j]
        map[2] = limb_darkening[2][j]

        if unspotted:
            map_list.append(map)
        else:
            if len(lat) == 1 and twospotted == False:
                map.spot(contrast=contrast_list[j], radius=radius[0], lat=lat[0], lon=lon[0])
            elif twospotted:
                map.spot(contrast=contrast_list[j][spotindex], radius=radius[0], lat=lat[0], lon=lon[0])
            else:
                for i in range(len(lat)):
                    map.spot(contrast=contrast_list[j][i], radius=radius[i], lat=lat[i], lon=lon[i])
            map_list.append(map)
    
    return map_list

def compute_model_BICs(synthetic_LC_folder, forward_model_lightcurves, T14, chromatic_err):
    """
    Forward_model_lightcurves contains all the model lightcurves.
    """
    # loading synthetic lightcurves
    syntheic_LC_filename = glob.glob(f"{synthetic_LC_folder}/synthetic*.csv")
    synthetic_LCs = np.array([np.loadtxt(filename, delimiter=',') for filename in syntheic_LC_filename])

    mask_out_of_transit = get_out_of_transit(T14_hours=T14)

    n_models = len(forward_model_lightcurves)
    BIC_array = []
    for n in range(n_models):
        if n == 0:
            k=4
        if n==1 or n==2:
            k=8
        if n==3:
            k=12

        model_BIC_array = []
        
        model_LC = np.array(forward_model_lightcurves[n])

        mean_model = np.mean([model_LC[i][mask_out_of_transit] for i in range(len(model_LC))], axis = 1)

        for synthetic_LC in synthetic_LCs:
            mean_data = np.mean([synthetic_LC[i][mask_out_of_transit] for i in range(len(synthetic_LC))], axis = 1)
            offset = mean_model - mean_data
            # plt.plot(synthetic_LC[0],'ok')
            # plt.plot(model_LC[0]-offset[0],'or')
            # plt.show()
            offset_2D = np.repeat(offset[:, np.newaxis], len(synthetic_LC[0]), axis=1)
            model_BIC_array.append(compute_BIC(k=k, data=synthetic_LC, data_err=chromatic_err, model=model_LC-offset_2D))
        
        model_mean_BIC = np.mean(model_BIC_array, axis=0)
        BIC_array.append(model_mean_BIC)
    return BIC_array

def get_preffered_model(BIC_array):
    """
    Performing model selection using the evaluated BIC array

    index 0 = pristine

    index 1 and index 2 = 1 spot

    index 4 = 2 spot
    """
    delta_BIC_threshold = 10


    if len(BIC_array) == 2:
        BIC_pristine = BIC_array[0]
        BIC_1spot = BIC_array[1]
        delta_pristine_vs_1spot = BIC_pristine - BIC_1spot
        mask_1spot = delta_pristine_vs_1spot >= delta_BIC_threshold

        preffered_model = np.select([mask_1spot], ['1-SPOT'],default='PRISTINE')

    if len(BIC_array) == 4:
        BIC_pristine = BIC_array[0]
        BIC_firstspot = BIC_array[1]
        BIC_secondspot = BIC_array[2]
        BIC_best_onespot = np.minimum(BIC_firstspot, BIC_secondspot)
        BIC_twospot = BIC_array[3]

        delta_pristine_vs_onespot = BIC_pristine - BIC_best_onespot
        delta_onespot_vs_twospot = BIC_best_onespot - BIC_twospot

        mask_1spot = delta_pristine_vs_onespot >= delta_BIC_threshold
        mask_2spot = delta_onespot_vs_twospot >= delta_BIC_threshold

        preffered_model = np.select([mask_2spot, mask_1spot], ['2-SPOT', '1-SPOT'], default='PRISTINE')
    
    return preffered_model

def run_preffered_model_test(forward_model_path):
    model_lc = generate_forward_model_array(config_file_path=f"{forward_model_path}/forward_model.cfg")

    config = cfg.ConfigParser()

    config.read(f"{forward_model_path}/forward_model.cfg")

    stellar_parameters_file = config['STAR']['stellar_parameters_file']

    star_config = cfg.ConfigParser()
    star_config.read(f"{BASE_DIR}/{stellar_parameters_file}")
    T14 = star_config.getfloat('PLANET', 'T14')
    err_file_path = star_config['CHROMATIC']['photometric_precision']
    chromatic_err = np.loadtxt(f"{BASE_DIR}/{err_file_path}", delimiter=',')
    # err_file_path = "/home/vampy/acads/projects/Spot_Spectrum_Ariel/Codes/SpotSpec/models/test/model_comparison/GJ1132_photometric_precision.txt"
    # chromatic_err = np.loadtxt(f"{err_file_path}", delimiter=',')

    BIC_array = compute_model_BICs(synthetic_LC_folder = forward_model_path,
                                forward_model_lightcurves = model_lc,
                                T14 = T14,
                                chromatic_err = chromatic_err)
    pref_model = get_preffered_model(BIC_array=BIC_array)
    np.savetxt(f"{forward_model_path}/preffered_model.txt", pref_model, delimiter=',', fmt='%s')
    np.savetxt(f"{forward_model_path}/model_BICs.txt", BIC_array, delimiter=',')

BASE_DIR = str(Path(os.getenv('SPOTSPEC_DIR', '/home/vampy/acads/projects/Spot_Spectrum_Ariel/Codes/SpotSpec')))

parser = argparse.ArgumentParser(prog="Checks the preffered model for retrieval.")

parser.add_argument('-d', '--directory',
                    type=str,
                    help='Directory of the forward model.')

args = parser.parse_args()

run_preffered_model_test(forward_model_path=args.directory)