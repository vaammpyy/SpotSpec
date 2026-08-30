from imports import *

def make_chromatic_stellar_surface(
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
            if len(lat) == 1:
                map.spot(contrast=contrast_list[j], radius=radius[0], lat=lat[0], lon=lon[0])
            else:
                for i in range(len(lat)):
                    map.spot(contrast=contrast_list[j][i], radius=radius[i], lat=lat[i], lon=lon[i])
            map_list.append(map)
    
    return map_list

def make_star(
    maps=(starry.Map(ydeg=20)),
    r: float=1.0,
    m: float=1.0,
    prot: float=1.0
):
    """
    Makes list of star with chromatic maps.

    Parameters
    ----------
    maps : list(maps)
        List of stellar maps.
    r : float
        Radius of the star in solar radius.
    m : float
        Mass of the star in solar mass.
    prot : float
        Rotation period of the star in days.
    
    Returns
    -------
    star_list : list(star)
        List of stars with applied stellar surface.
    """
    star_list = []
    for map_item in maps:
        star = starry.Primary(map=map_item, r=r, m=m, prot=prot)
        star_list.append(star)

    return star_list

def make_planet(
    planet_surface = starry.Map(yedg=5, amp=5e-3),
    radius: Sequence[float] = (1,),
    mass: float = 1,
    porb: float = 1, #not using semi-major axis value or a
    prot: float = 1,
    Omega: float = 30,
    ecc: float = 0.3,
    w: float = 30,
    t0: float = 0,
    inc: float = np.rad2deg(np.arccos(.1/14)),
    length_unit = R_earth,
    mass_unit = M_earth,
    time_unit = day,
    angle_unit = degree
):
    """
    Makes chromatic planet.

    Makes chromatic planet with varying radius.

    Parameters
    ----------
    planet_surface : starry.Map.YlmBase
        Surface map of the planet.
    radius : Sequence[float]
        List of planetary radius in each wavelength bin.
    
    For other parameters check out the documentation: https://starry.readthedocs.io/en/latest/Secondary/

    Returns
    -------
    planet_list: list(planet)
        List of planets with given parameters
    """
    planet_list = []
    for r in radius:
         planet = starry.kepler.Secondary(
            planet_surface,
            m=mass,
            r=r,
            porb=porb,
            prot=prot,
            Omega=Omega,
            ecc=ecc,
            w=w,
            t0=t0,
            inc=inc,
            length_unit=length_unit,
            mass_unit=mass_unit,
            time_unit=time_unit,
            angle_unit=angle_unit
         )
         planet_list.append(planet)
    return planet_list 

def make_star_planet_system(star_list, planet_list):
    """
    Makes star and planet system.

    Makes chromatic star and planet system.

    Parameters
    ----------
    star_list : list
        List of chromatic star.
    planet_list : list
        List of chromatic planet.
    
    Returns
    -------
    chromatic_system : list
        List of systems at different wavelengths.
    """
    chromatic_system = []
    is_chromatic_planet = not(len(planet_list) == 1)

    for i, star in enumerate(star_list):
        if is_chromatic_planet:
            system = starry.System(star, planet_list[i])
        else:
            system = starry.System(star, planet_list[0])
        chromatic_system.append(system)
    return chromatic_system

def generate_lightcurve(system_list, time, path_to_save_lightcurve=''):
    """
    Generates the lightcurve of the system.

    Generate chromatic lightcurve of the system.

    Parameters
    ----------
    system_list : list
        List of systems.
    time : list
        List of time array to evaluate the lightcurve.
    path_to_save_lightcurve : str
        Path to save the lightcurve, empty by default if non empty then the file will be saved in csv format.
    
    Returns
    -------
    lightcurve_list : list
        List of lightcurves, each entry corresponds to a wavelength.
    """
    lightcurve_list = []
    for system in system_list:
        # flux_system = system.flux(time).eval()
        flux_system = system.flux(time)
        lightcurve_list.append(flux_system)
    if path_to_save_lightcurve:
        np.savetxt(path_to_save_lightcurve, np.array(lightcurve_list), delimiter=',')
    return lightcurve_list

def simulate_ARIEL_lc(lightcurve, ARIEL_noise, CHROMATIC=False):
    """
    Takes a lightcurve in one of the ARIEL's channel and simulates an observation with noise.
    The noise will depend on the object.

    Parameters
    ----------
    lightcurve: 1D array or 2D array
        Lightcurve in one of ARIEL's channel
    ARIEL_noise: float or 1D array
        Noise value in the ARIEL's channel for the target.
    CHROMATIC: bool
        Whether to simulate chromatic noise.
    
    Returns
    -------
    ARIEL_lightcurve: 1D array
        Simulated ARIEL observation in the given channel.
    """
    if CHROMATIC:
        # noise_array = []
        # for i, lc in enumerate(lightcurve):
        #     noise_array.append(np.random.normal(loc=0.0, scale=ARIEL_noise[i], size=len(lc)))
        # # noise_array = np.random.normal(loc=0.0, scale=ARIEL_noise, size=len(lightcurve))
        # noise_array = np.array(noise_array)
        lightcurve = np.asarray(lightcurve, dtype=float)
        ARIEL_noise = np.asarray(ARIEL_noise, dtype=float)
        noise_array = np.random.normal(
            loc=0.0,
            scale=ARIEL_noise[:, None],   # shape: (n_wave, 1)
            size=lightcurve.shape          # shape: (n_wave, n_time)
        )
    else:
        noise_array = np.random.normal(loc=0.0, scale=ARIEL_noise, size=len(lightcurve))
    ARIEL_lightcurve = np.array(lightcurve) + noise_array
    return ARIEL_lightcurve

def plot_lightcurve(lightcurve, noise_array, path_to_save):
    """
    Plots the lightcurve

    Parameters
    ----------
    lightcurve: 1D array
        All lightcurves in 102 ARIEL wavelength channels.
    noise_array: 1D array
        All noise values in 102 ARIEL wavelength channels.
    path_to_save: str
        Path to save the lightcurve figure.
    """
    colors = ['tab:blue', 'tab:cyan', 'darkgreen', 'lime', 'tab:orange', 'tab:red']
    label = ['VISPhot', 'FGS-1', 'FGS-2', 'NIRSpec', 'AIRS-Ch0', 'AIRS-Ch1']

    LC_idx = [0, 1, 2, 8, 42, 90]

    # 1. Create subplots dynamically based on the dataset length
    fig, axes = plt.subplots(
        nrows=6, 
        ncols=1, 
        sharey=False, 
        sharex=True,  # Optional: hides X-axis tick labels except on the bottom subplot
        figsize=(8, 5 * 5),
        gridspec_kw={"hspace": 0.05},  
    )

    length_lightcurve = len(lightcurve[0])

    time_axis = np.arange(-length_lightcurve/2, length_lightcurve/2, 1)/60
    
    for i, idx in enumerate(LC_idx):
        ARIEL_channel_lightcurve = simulate_ARIEL_lc(lightcurve=lightcurve[idx], ARIEL_noise=noise_array[idx])
        # axes[i].scatter(time_axis, ARIEL_channel_lightcurve, s=20, edgecolor='k', color=colors[i], label = label[i])
        axes[i].errorbar(
            time_axis, 
            ARIEL_channel_lightcurve, 
            yerr=noise_array[idx],       # Pass the per-point photometric error
            fmt='o',                     # Circle markers (equivalent to scatter)
            markersize=6,                # Marker size
            markerfacecolor=colors[i],   # Interior color of the marker
            markeredgecolor='k',         # Black outline around markers
            ecolor='k',            # Color of the error bar lines
            capsize=2,                   # Size of top/bottom caps on error bars (0 to disable)
            alpha=0.8,                   # Transparency for clearer overlap visibility
            label=label[i],
            zorder=2)
        axes[i].plot(time_axis, lightcurve[idx], c=colors[i], linewidth=2, zorder=1)
        axes[i].legend()
        axes[i].set_ylim(min(ARIEL_channel_lightcurve)*0.9995, max(ARIEL_channel_lightcurve)*1.0005)
    plt.xlabel("Time [hrs]")
    # plt.show()
    plt.savefig(f"{path_to_save}", dpi=600, bbox_inches='tight')

def generate_forward_model_grid():
    """
    This function will generate the forward model grid according to our requirements.
    """
    return