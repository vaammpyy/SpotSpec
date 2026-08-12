import numpy as np
import configparser as cfg
import os
from pathlib import Path

def read_limb_darkening_file(path_to_limb_darkening_file):
    """
    Reads the limb darkening file.

    Reads the limb darkening file and returns the coefficients for quadratic limb darkening.

    Parameters
    ----------
    path_to_limb_darkening_file : str
        Path to the limb darkening file.

    Returns
    -------
    limb_darkening : list
        List of quadratic limb darkening coefficients, [<list of u1>, <list of u2>]
    """
    limb_darkening = np.loadtxt(path_to_limb_darkening_file, delimiter=',', dtype=float).T
    return limb_darkening

def read_contrast_file(path_to_contrast_file):
    """
    Reads the chromatic contrast file.

    Reads the chromatic contrast file for each of the inhomogeneity.

    Parameters
    ----------
    path_to_contrast_file : str
        Path to contrast file.
    
    Returns
    -------
    contrast : 2D array
        2D array of contrast where each element of the array is list of contrast for each homogeneity.
    """
    contrast = np.loadtxt(path_to_contrast_file, delimiter=',', dtype=float)
    return contrast

def read_planet_radius_file(path_to_planet_radius_file):
    """
    Reads the planetary radius file.

    Reads the chromatic planetary radius file.

    Parameters
    ----------
    path_to_planet_radius_file : str
        Path to chromatic planetary radius.
    
    Returns
    -------
    planet_radius : list
        Chromatic planet radius list.
    """
    planet_radius = np.loadtxt(path_to_planet_radius_file, dtype=float)
    return planet_radius

def config_reader(path_to_config_file):
    """
    This module will read the config file.

    This module will read the config file for the model parameters.

    Parameters
    ----------
    """
    parameter_dict={"star": {},
                    "inhomogeneties": {},
                    "planet": {},
                    "output": {},
                    "plot": {}}
    config_file = cfg.ConfigParser()

    config_file.read(path_to_config_file)

    BASE_DIR = str(Path(os.getenv('SPOTSPEC_DIR', '/home/vampy/acads/projects/Spot_Spectrum_Ariel/Codes/SpotSpec')))
    config_file['STAR']['limb_darkening_file'] = f"{BASE_DIR}{config_file['STAR']['limb_darkening_file']}"
    config_file['STAR']['stellar_parameters_file'] = f"{BASE_DIR}{config_file['STAR']['stellar_parameters_file']}"
    config_file['STAR.INHOM']['contrast_file'] = f"{BASE_DIR}{config_file['STAR.INHOM']['contrast_file']}"
    config_file['PLANET']['radius_file'] = f"{BASE_DIR}{config_file['PLANET']['radius_file']}"

    # populating stellar dictionary
    parameter_dict["star"]["radius"] = config_file.getfloat('STAR', 'radius')
    parameter_dict['star']["mass"] = config_file.getfloat('STAR', 'mass')
    parameter_dict['star']["prot"] = config_file.getfloat('STAR', 'prot')
    parameter_dict['star']["nw"] = config_file.getint('STAR', 'nw')
    c1_list, c2_list = read_limb_darkening_file(config_file['STAR']['limb_darkening_file'])
    parameter_dict['star']['limb_darkening'] = [2, c1_list, c2_list]
    parameter_dict['star']['parameter_file'] = config_file['STAR']['stellar_parameters_file']

    # populating inhomogenity dictionary
    parameter_dict['inhomogeneties']['radius'] = [float(x.strip()) for x in str(config_file['STAR.INHOM']['radius']).split(',')]
    parameter_dict['inhomogeneties']['latitude'] = [float(x.strip()) for x in str(config_file['STAR.INHOM']['latitude']).split(',')]
    parameter_dict['inhomogeneties']['longitude'] = [float(x.strip()) for x in str(config_file['STAR.INHOM']['longitude']).split(',')]
    parameter_dict['inhomogeneties']['contrast_list'] = read_contrast_file(config_file['STAR.INHOM']['contrast_file']) #this will parse the contrast list file to the reading function

    # populating planetary dictionary
    parameter_dict['planet']['radius'] = read_planet_radius_file(config_file['PLANET']['radius_file'])
    parameter_dict['planet']['mass'] = config_file.getfloat('PLANET', 'mass')
    parameter_dict['planet']['prot'] = config_file.getfloat('PLANET', 'prot')
    parameter_dict['planet']['T14'] = config_file.getfloat('PLANET', 'T14')

    parameter_dict['planet']['porb'] = config_file.getfloat('PLANET.ORBIT', 'porb')
    parameter_dict['planet']['Omega'] = config_file.getfloat('PLANET.ORBIT', 'Omega')
    parameter_dict['planet']['ecc'] = config_file.getfloat('PLANET.ORBIT', 'ecc')
    parameter_dict['planet']['w'] = config_file.getfloat('PLANET.ORBIT', 'w')
    parameter_dict['planet']['t0'] = config_file.getfloat('PLANET.ORBIT', 't0')
    parameter_dict['planet']['inc'] = config_file.getfloat('PLANET.ORBIT', 'inc')

    # populating output dictionary
    parameter_dict['output']['path_to_lightcurve'] = f"{BASE_DIR}{config_file['OUTPUT']['path_to_lightcurve']}"
    parameter_dict['output']['path_to_lightcurve_plot'] = f"{BASE_DIR}{config_file['OUTPUT']['path_to_lightcurve_plot']}"
    parameter_dict['output']['path_to_animation'] = f"{BASE_DIR}{config_file['OUTPUT']['path_to_animation']}"

    # populating the plotting dictionary
    parameter_dict['plot']['lightcurves'] = config_file.getboolean('PLOT', 'lightcurves')

    return parameter_dict

def config_writer(star_cfg, inhom_parameter_dict, folder_name):
    """
    This module will write the config file.

    This module will read the config file for the model parameters.

    Parameters
    ----------
    star_cfg: str
        Path to the stars observed parameters.
    inhom_parameter_dict: dict
        Inhomogeneity parameter dictionary.
    folder_name: str
        Path to save the config file.
    """

    BASE_DIR = str(Path(os.getenv('SPOTSPEC_DIR', '/home/vampy/acads/projects/Spot_Spectrum_Ariel/Codes/SpotSpec')))

    # write the config file to the folder
    config_read_star = cfg.ConfigParser()
    config_read_star.read(star_cfg)

    #constructing paths
    config_read_star['CHROMATIC']['spot_contrast'] = f"{BASE_DIR}{config_read_star['CHROMATIC']['spot_contrast']}"
    config_read_star['CHROMATIC']['faculae_contrast'] = f"{BASE_DIR}{config_read_star['CHROMATIC']['faculae_contrast']}"
    # config_read_star['CHROMATIC']['LD_file'] = f"{BASE_DIR}{config_read_star['CHROMATIC']['LD_file']}"
    # config_read_star['CHROMATIC']['planet_radius'] = f"{BASE_DIR}{config_read_star['CHROMATIC']['planet_radius']}"

    path_to_folder = f"{BASE_DIR}/{config_read_star['PATH']['model_root']}{folder_name}"
    path_to_folder_general = f"{config_read_star['PATH']['model_root']}{folder_name}"
    star_cfg_path = f"{config_read_star['PATH']['references_root']}observed_parameters.cfg"

    os.makedirs(path_to_folder, exist_ok=True)

    spot_contrast = np.loadtxt(config_read_star['CHROMATIC']['spot_contrast'])
    faculae_contrast = np.loadtxt(config_read_star['CHROMATIC']['faculae_contrast'])
    contrast_list = []
    for typ in inhom_parameter_dict['type']:
        if typ == "S":
            contrast_list.append(spot_contrast)
        if typ == "F":
            contrast_list.append(faculae_contrast)
    contrast_list = np.array(contrast_list).T
    contrast_list_fname = f"{path_to_folder}/contrast_list.csv"
    np.savetxt(contrast_list_fname, contrast_list, delimiter=',')
    contrast_list_fname = f"{config_read_star['PATH']['model_root']}{folder_name}/contrast_list.csv"

    forward_model_dict = {"STAR":{
                                "radius": config_read_star.getfloat("STAR", "radius"),
                                "mass": config_read_star.getfloat("STAR", "mass"),
                                "prot": config_read_star.getfloat("STAR", "prot"),
                                "nw": 102, # keeping it fixed for ARIEL
                                "limb_darkening_file": config_read_star["CHROMATIC"]["LD_file"],
                                "stellar_parameters_file": star_cfg_path},
                        
                        "STAR.INHOM":{
                                "radius": ", ".join(map(str, inhom_parameter_dict['radius'])),
                                "latitude": ", ".join(map(str, inhom_parameter_dict['latitude'])),
                                "longitude": ", ".join(map(str, inhom_parameter_dict['longitude'])),
                                "contrast_file": contrast_list_fname},
                        
                        "PLANET":{
                                "radius_file": config_read_star["CHROMATIC"]["planet_radius"],
                                "mass": config_read_star.getfloat("PLANET", "mass"),
                                "prot": 1,
                                "T14": config_read_star.getfloat("PLANET", "T14")},
                                
                        "PLANET.ORBIT":{
                                "porb": config_read_star.getfloat("PLANET", "porb"),
                                "Omega": 0,
                                "ecc": 0,
                                "w": 0,
                                "t0": 0,
                                "inc": config_read_star.getfloat("PLANET", "inc")},
                        
                        "OUTPUT":{"path_to_lightcurve": f"{path_to_folder_general}/lightcurve.csv",
                                    "path_to_lightcurve_plot": f"{path_to_folder_general}/lightcuve.png",
                                    "path_to_animation": f"{path_to_folder_general}/animation.gif"},
                        
                        "PLOT":{"lightcurves":inhom_parameter_dict['plot']['lightcurves']}}
    
    config_to_write = cfg.ConfigParser()
    config_to_write.read_dict(forward_model_dict)

    with open(f"{path_to_folder}/forward_model.cfg", 'w') as file:
        config_to_write.write(file)