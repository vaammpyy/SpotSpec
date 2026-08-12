from imports import *
from forward_model import *

import argparse
from config_utils import config_reader
import os
from pathlib import Path

parser = argparse.ArgumentParser(prog='SpotSpec Forward Model',
                                    description='Runs the spot spectrum forward model.')

parser.add_argument('-c', "--configfile",
                    type=str,
                    help='Path to the config file.')

args = parser.parse_args()

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

    map_list = make_chromatic_stellar_surface(ydeg = 30,
                                                limb_darkening = parameter_dict['star']['limb_darkening'],
                                                amp = amp,
                                                nw = nw,
                                                rv = False,
                                                contrast_list = parameter_dict['inhomogeneties']['contrast_list'],
                                                radius = parameter_dict['inhomogeneties']['radius'],
                                                lat = parameter_dict['inhomogeneties']['latitude'],
                                                lon = parameter_dict['inhomogeneties']['longitude'])
    
    star_list = make_star(maps = map_list,
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
    
    system_list = make_star_planet_system(star_list=star_list, planet_list=planet_list)

    return system_list

def run_forward_model(config_file_path):
    """
    Runs the forward model.

    Runs the forward model with a given set of parameters.

    Parameters
    ----------
    config_file_path : str
        Path to config file.
    """
    parameter_dict = config_reader(config_file_path)
    system_list = generate_forward_model(parameter_dict=parameter_dict)

    # generating the lightcurve
    T14_hrs = parameter_dict['planet']['T14'] # in hours
    T14_seconds = T14_hrs*3600

    time = np.arange(-T14_seconds, T14_seconds+60, 60)/(3600*24)
    lightcurve_list = generate_lightcurve(system_list=system_list, time=time, path_to_save_lightcurve=parameter_dict['output']['path_to_lightcurve'])

    # plotting the lightcurve
    stellar_cfg = cfg.ConfigParser()
    stellar_cfg.read(parameter_dict['star']['parameter_file'])

    BASE_DIR = str(Path(os.getenv('SPOTSPEC_DIR', '/home/vampy/acads/projects/Spot_Spectrum_Ariel/Codes/SpotSpec')))
    print(f"BASE_DIR: {BASE_DIR}")
    ARIEL_noise_file = f"{BASE_DIR}{stellar_cfg['CHROMATIC']['photometric_precision']}"
    ARIEL_noise = np.loadtxt(ARIEL_noise_file)
    plot_lightcurve(lightcurve_list, ARIEL_noise, path_to_save=parameter_dict['output']['path_to_lightcurve_plot'])
    
    #generating animation
    #system_list[0].show(t=time, figsize=(8,8), show=False)
    # print(f"Type of anim: {type(anim)}")

    # if anim is None:
    #     # Print system properties to check if parameters are symbolic or multi-wavelength
    #     print("Star map dimensions (nw):", getattr(star.map, "nw", 1))
    #     print("Planet map dimensions (nw):", getattr(planet.map, "nw", 1))
    # anim.save(parameter_dict['output']['path_to_animation'], writer=animation.PillowWriter(fps=4))
    # plt.close()



run_forward_model(config_file_path=args.configfile)
# map_list=make_chromatic_stellar_surface(amp=[1,2], contrast_list=[(0, 0.5, 0.5, 0.3), (0, 0.3, 0.7, 0.3)], limb_darkening=[2,(0.5, 0.8), (0.25, 0.4)], radius=[15, 5, 30, 10], lat=[10, -10, 70, -5], lon=[-30, 15, 0, -5])
# star_list=make_star(map_list, prot=10)
# planet_list=make_planet(porb=2, radius=[10, 15])
# system_list = make_star_planet_system(star_list, planet_list)
# time = np.linspace(-0.5, 0.5, 1000)
# lightcurve_list = generate_lightcurve(system_list=system_list, time=time, path_to_save_lightcurve='./lightcurve.csv')