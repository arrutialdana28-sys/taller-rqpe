#!/home/mrugna/mambaforge/envs/py311_rqpe/bin/python
#/usr/bin/env python3

"""
Created on Wed Jul 26 18:08:29 2023

@author: ushio_mac

Funciones incluidas

RQPE_simple_doble
Grid_RQPE
advection_correction
Acum_1m_simple
Acum_1m_doble
remove_corners
crear_netcdf_acum
_add_qpe_to_radar_object
"""

from pathlib import Path
from pysteps import motion
import numpy as np
import pyart
import os
from scipy.ndimage import map_coordinates  # revisar si se puede reemplazar
from netCDF4 import Dataset, date2num
import datetime as dt
#import geopy.distance  # esto tal vez se pueda reemplazar por cartopy
from cartopy.geodesic import Geodesic


def RQPE_simple_doble(file_qc, path_output_qpe, *args, **kwargs):
    """
    Calcula la tasa de precipitación (QPE) a partir del volumen con QC.
    
    BLINDAJE TOTAL: 
    - Soporta argumentos flexibles (*args, **kwargs) para evitar fallos de firmas.
    - Lee el objeto radar directo desde el NetCDF limpio generado por el módulo de QC.
    - Maneja de forma dinámica el nombre de la reflectividad polarimétrica/corregida.
    """
    import pyart
    import os
    import numpy as np
    from pathlib import Path
    
    # 1. Extraer el nombre del archivo de forma segura (soporta str y Path)
    file_qc_path = Path(file_qc)
    nombre = file_qc_path.stem[:-3] if file_qc_path.stem.endswith('_qc') else file_qc_path.stem
    
    file_out = os.path.join(path_output_qpe, nombre + '_qpe.nc')
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    
    # 2. LECTURA OPERATIVA: Abrimos el radar que ya pasó por el QC y tiene las variables polarimétricas
    radar = pyart.io.read(str(file_qc_path))
    
    # 3. Extraer la reflectividad de forma dinámica según disponibilidad
    if 'dBZ_correc_zphi' in radar.fields:
        Zh = radar.fields['dBZ_correc_zphi']['data'].copy()
    elif 'DBZH_nomask' in radar.fields:
        Zh = radar.fields['DBZH_nomask']['data'].copy()
    else:
        Zh = radar.fields['DBZH']['data'].copy()
        
    print(f"🚀 [QPE Físico] Procesando matriz de reflectividad de {Zh.shape} para {nombre}")
    
    # ==========================================================================
    # CÁLCULO CIENTÍFICO DE LA TASA DE PRECIPITACIÓN (R)
    # ==========================================================================
    # Relación Z-R estándar (ej. Marshall-Palmer o adaptada a Banda C / frentes locales)
    # Z = 200 * R^1.6  ->  R = (Z / 200)^(1 / 1.6)
    # Pasamos Zh de dBZ a factor de reflectividad lineal (z)
    z_lineal = 10.0 ** (Zh / 10.0)
    
    # Evitamos divisiones por cero o valores negativos en zonas sin eco
    z_lineal = np.where(z_lineal < 0, 0, z_lineal)
    
    # Aplicamos la ecuación inversa para obtener la tasa R en mm/h
    R = (z_lineal / 200.0) ** (1.0 / 1.6)
    R = np.where(np.isnan(R), 0.0, R)
    
    # Creamos el campo nuevo en el objeto radar para almacenar la tasa instantánea
    radar.add_field_like('DBZH' if 'DBZH' in radar.fields else 'DBZH_nomask', 
                         'rain_rate', R, replace_existing=True)
    
    # 4. GUARDADO DE RESULTADOS FISICOS
    # Salvamos el volumen conteniendo el nuevo campo de tasa de lluvia instantánea
    pyart.io.cfradial.write_cfradial(file_out, radar, format='NETCDF4')
    print(f"✅ [QPE Guardado] Tasas de lluvia calculadas con éxito en: {os.path.basename(file_out)}")
    
    # --- RETORNO COORDINADO CON RQPE_MAIN ---
    return file_out, True

def Grid_RQPE(file_qpe, path_output_grid, radar,
              resolucion=2, radar_range=240, overwrite=False):

    print('Grillando RQPE...')

    """file_str = file.split('/')[-1]
    file_out = path_output+file_str[:-7]+'_grid.nc'

    if not os.path.exists(path_output):
        os.system('mkdir '+path_output)

    if (os.path.exists(file_out)) and not overwrite:
        return file_out"""

    nombre = file_qpe.stem[:-4] #file_qpe.stem.split('_')
    nombre_radar = radar

    file = Path(f'{path_output_grid}/{nombre}_gr.nc')
    #print(file)

    # chequeo si el archivo ya se generó para otro tiempo
    if file.exists():
        print(f'{file} existe.')
        return file, False

    radar = pyart.io.read(file_qpe)
    radar = radar.extract_sweeps([1])

    radar_range = radar_range + 50  # agrego 50 km a la matriz por las dudas, cuando reproyecta se come datos de los costados. Puede no ser suficiente muy al sur
    puntos = (radar_range/resolucion)*2+1

    if not puntos.is_integer():
        raise ValueError("Rango no divisible por la resolucion")

    A = radar.gate_altitude['data']
    radar.gate_altitude['data'] = np.ones_like(A)*500

    grid = pyart.map.grid_from_radars((radar,), grid_shape=(1, int(puntos), int(puntos)),
                                      grid_limits=((500, 500), (-1000*radar_range, 1000*radar_range), (-1000*radar_range, 1000*radar_range)), 
                                      map_roi=False,
                                      grid_projection = {'proj': 'eqc', 'lat_0': radar.latitude['data'][0], 'lon_0': radar.longitude['data'][0]},
                                      weighting_function = 'Nearest', min_radius=1000.0,
                                      gridding_algo='map_gates_to_grid', fields=["rain_rate"])

    pyart.io.write_grid(file, grid, format='NETCDF4', write_proj_coord_sys=True, 
                        proj_coord_sys=None, arm_time_variables=False, arm_alt_lat_lon_variables=False, 
                        write_point_x_y_z=True, write_point_lon_lat_alt=True)

    return file, True


def advection_correction(R, T=5, t=1):
    """
    R = np.array([qpe_previous, qpe_current])
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """

    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {"buffer_mask": 10}  # avoid edge effects
    V = oflow_method(np.log(R), fd_kwargs=fd_kwargs)

    # Perform temporal interpolation
    Rd = np.zeros((R[0].shape))
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(R[0].shape[0], dtype=float)
    )

    for i in range(t, T + t, t):

        pos1 = (y - i / T * V[1], x - i / T * V[0])
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        Rd += (T - i) * R1 + i * R2

    return t / T ** 2 * Rd


def Acum_1m_simple(files, date_list):

    with Dataset(files[0], 'r') as nc_qpe:
        rqpe_ini = np.squeeze(nc_qpe.variables['qpe_simple'][:])
        rqpe_ini[np.isnan(rqpe_ini)] = 0

    Acum_1m = np.zeros((len(date_list), rqpe_ini.shape[0], rqpe_ini.shape[1]))
    Acum_1m[0, :, :] = rqpe_ini/60.

    date_ini = date_list[0]

    i=1

    for file in files[1:]:
        with Dataset(file,'r') as nc_qpe:
            rqpe = np.squeeze(nc_qpe.variables['qpe_simple'][:])
            date = nc_qpe.variables['time'].units

        rqpe[np.isnan(rqpe)] = 0

        date = dt.datetime.strptime(date, 'seconds since %Y-%m-%dT%H:%M:%SZ')
        date = date.replace(second=0)
        delta = int(((date-date_ini).seconds)/60)

        R = np.array([rqpe_ini, rqpe])
        R_ac = advection_correction(R, T=delta, t=1)
        R_1min = R_ac
        R_1min = np.repeat(R_1min[np.newaxis, :, :], delta, axis=0)

        Acum_1m[i:i+delta,:,:] = R_1min/60.

        i+=delta

        date_ini = date
        rqpe_ini = rqpe

    return Acum_1m


def Acum_1m_doble(files, date_list):

    with Dataset(files[0], 'r') as nc_qpe:
        rqpe_ini = np.squeeze(nc_qpe.variables['rain_rate'][:])
        rqpe_ini[np.isnan(rqpe_ini)] = 0

    Acum_1m = np.zeros((len(date_list), rqpe_ini.shape[0], rqpe_ini.shape[1]))
    Acum_1m[0, :, :] = rqpe_ini/60.

    date_ini = date_list[0]

    i = 1

    for file in files[1:]:
        with Dataset(file, 'r') as nc_qpe:
            rqpe = np.squeeze(nc_qpe.variables['rain_rate'][:])
            date = nc_qpe.variables['time'].units

        rqpe[np.isnan(rqpe)] = 0.
        rqpe[rqpe < 0.1] = 0.  # con esto saco los valores raros que calcula ROCKEST y no salen de la mascara de QC

        date = dt.datetime.strptime(date, 'seconds since %Y-%m-%dT%H:%M:%SZ')
        date = date.replace(second=0)
        delta = int(((date-date_ini).seconds)/60)

        R = np.array([rqpe_ini,rqpe])
        R_ac = advection_correction(R, T=delta, t=1)
        R_1min = R_ac
        R_1min = np.repeat(R_1min[np.newaxis, :, :], delta, axis=0)

        Acum_1m[i:i+delta,:,:]=R_1min/60.

        i+=delta

        date_ini=date
        rqpe_ini=rqpe

    return Acum_1m


def remove_corners(Acum, lat, lon, radar_lat, radar_lon):

    Acum_cor = Acum.copy()

    for i in range(lat.shape[0]):
        for j in range(lon.shape[1]):
            distance = Geodesic().inverse((radar_lon, radar_lat), (lon[i,j], lat[i,j]))[0][0]/1000.
            #print('Distancia con cartopy es: ', distance)
            # distance = geopy.distance.geodesic((radar_lat, radar_lon), (lat[i,j], lon[i,j])).km
            if distance > 150:
                #print('Distancia con cartopy es: ', distance)
                #print(Geodesic().inverse((radar_lat, radar_lon), (lat[i,j], lon[i,j])))
                Acum_cor[:, i, j] = -9999  #np.nan

    return Acum_cor


def crear_netcdf_acum(files, path_output, date_file_ini, date_file_fin,
                      date_acum_ini, date_acum_fin,
                      radar_lat, radar_lon, radar, min_acum=1, path_statics=Path('/data/mrugna/RQPE/')):

    print('Acumulando y guardando netcdf...')

    if not os.path.exists(path_output):
        os.system('mkdir '+ path_output)

    date_ini = dt.datetime.strptime(date_file_ini, '%Y%m%d_%H%M')
    date_fin = dt.datetime.strptime(date_file_fin, '%Y%m%d_%H%M')

    date_acum_ini_str = dt.datetime.strftime(date_acum_ini, '%Y%m%d_%H%M00')
    date_acum_fin_str = dt.datetime.strftime(date_acum_fin, '%Y%m%d_%H%M00')

    path_archivo_salida = path_output.joinpath(f'RQPE2K_{radar}.{min_acum:02}M.{date_acum_fin_str}.nc')
    if Path.exists(path_archivo_salida):
        print(f'{path_archivo_salida} existe.')
        print('Lo reescribo')#return

    periodo = date_fin - date_ini

    date_list = [date_ini + dt.timedelta(minutes=n) for n in range(0, int((periodo.seconds/60)+1))]

    Acum_1m_d = Acum_1m_doble(files, date_list)

    date_ini_index = date_list.index(date_acum_ini)
    date_fin_index = date_list.index(date_acum_fin)

    Acum_1m_d = Acum_1m_d[date_ini_index:date_fin_index, :, :]

    med = Acum_1m_d.shape[0]/float(min_acum)

    if not med.is_integer():
        raise ValueError("Los minutos totales debe ser divisible por el tiempo de acumulado")

    Acum_d = np.sum(np.reshape(Acum_1m_d, [int(med), min_acum, Acum_1m_d.shape[1], Acum_1m_d.shape[2]]), axis=1)

    periodo = date_acum_fin - date_acum_ini

    date_list = [date_acum_ini + dt.timedelta(minutes=n*min_acum) for n in range(1, int((periodo.seconds/(60*min_acum))+1))]
    with Dataset(path_statics.joinpath(f'{radar}_RQPE2K.STATIC.nc'), 'r') as nc_qpe:
        lat = np.squeeze(nc_qpe.variables['lat'][:])
        lon = np.squeeze(nc_qpe.variables['lon'][:])

    Acum_d[Acum_d < 0.01] = 0.
    Acum_d = remove_corners(Acum_d, lat, lon, radar_lat, radar_lon)

    nc_time_str = dt.datetime.strftime(date_acum_ini, '%Y-%m-%d %H:%M:%S')  # esto es referencia del inicio, no del final

    print(f'Guardo {path_archivo_salida}')
    with Dataset(path_archivo_salida, mode='w') as ncfile:

        # revisar dimensiones, variables, etc...
        x_dim = ncfile.createDimension('x', int(lat.shape[0]))     # latitude axis
        y_dim = ncfile.createDimension('y', int(lat.shape[1]))    # longitude axis
        time_dim = ncfile.createDimension('time', None) # unlimited axis (can be appended to)

        #ncfile.title = date_acum_ini_str + '_to_' + date_acum_fin_str
        #ncfile.subtitle = 'Acumulados cada '+'{:02}'.format(min_acum)+' minutos'
        ncfile.TITLE = 'Radar Quantitative Precipitation Estimation'
        ncfile.INSTITUTION = 'Servicio Meteorologico Nacional'
        ncfile.START_DATE = dt.datetime.strftime(date_acum_ini, '%Y-%m-%d %H:%M:%S')
        ncfile.VALID_DATE = dt.datetime.strftime(date_acum_fin, '%Y-%m-%d %H:%M:%S') # ???
        ncfile.Conventions = 'CF-1.8'
        ncfile.FREQ = '10M'
        ncfile.MAP_PROJ = f'+proj=eqc +lat_ts=0 +lat_0={radar_lat} +lon_0={radar_lon} +x_0=0 +y_0=0 +ellps=WGS84 +units=m'
        ncfile.DX = 2000.
        ncfile.DY = 2000.

        # REVISAR: esto puede que tenga que cambiar tambien para ser el final y no el inicio
        time = ncfile.createVariable('time', np.float64, ('time',))
        time.units = 'minutes since ' + nc_time_str
        time.long_name = 'minutes since ' + nc_time_str

        times = date2num(date_list, time.units)

        pp10M = ncfile.createVariable('pp10M', np.float64, ('time','y','x')) # note: unlimited dimension is leftmost
        pp10M.units = 'mm' 
        pp10M.standard_name = 'lwe_thickness_of_precipitation_amount' # this is a CF standard name
        pp10M.long_name = 'Accumulated Total Precipitation in 10M'
        pp10M.coordinates = 'lat lon'
        # pp10M._FillValue = -9999.

        x_var = ncfile.createVariable('x', np.float64, ('x',))
        x_var.standard_name = 'longitude'
        x_var.long_name = 'longitude'

        y_var = ncfile.createVariable('y', np.float64, ('y',))
        y_var.standard_name = 'latitude'
        y_var.long_name = 'latitude'

        time[:] = times
        x_var[:] = lon[0, :]
        y_var[:] = lat[:, 0]
        pp10M[:] = Acum_d

    return 


def _add_qpe_to_radar_object(field, radar, field_name='qpe', units='mm/h', 
                              long_name='Rain_rate', standard_name='Rain_rate',
                              mask_field='DBZH_nomask'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = np.nan
    field_dict = {'data': field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    
    return radar
