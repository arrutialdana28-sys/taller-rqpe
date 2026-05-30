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

def Grid_RQPE(file_qpe, path_output_grid, *args, **kwargs):
    """
    Grilla los datos de QPE a una malla cartesiana regular sin destruir datos.
    - Amplía el rango vertical (Z) para asegurar que capture la lluvia del volumen.
    - Limpia las máscaras de Py-ART convirtiéndolas a ceros físicos manejables.
    """
    import pyart
    import os
    import numpy as np
    from pathlib import Path
    
    file_qpe_path = Path(file_qpe)
    nombre = file_qpe_path.stem[:-4] if file_qpe_path.stem.endswith('_qpe') else file_qpe_path.stem
    
    file_out = os.path.join(path_output_grid, nombre + '_grid.nc')
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    
    radar = pyart.io.read(str(file_qpe_path))
    print(f"🗺️ [Grillado Cartesiano] Interpolando volumen a malla regular para {nombre}")
    
    # Rellenar máscaras en el objeto radar antes de grillar para evitar que Barnes extienda el NaN
    if 'rain_rate' in radar.fields:
        data_raw = radar.fields['rain_rate']['data']
        if np.ma.isMaskedArray(data_raw):
            radar.fields['rain_rate']['data'] = data_raw.filled(0.0)
            
    res_km = kwargs.get('res', 2.0)
    grid_shape = (1, int(300 / res_km), int(300 / res_km)) 
    
    try:
        # Ampliamos los límites de Z (de 1km a 4km) para que interpole con datos reales del volumen
        grid = pyart.map.grid_from_radars(
            (radar,),
            grid_shape=grid_shape,
            grid_limits=((1000, 4000), (-150000, 150000), (-150000, 150000)),
            fields=['rain_rate'],
            gridding_algo='map_to_grid',
            weighting_function='Barnes',
            roi_func='dist_beam' # Usa el ancho del haz del radar para ponderar, mucho más físico
        )
        
        # Guardar grilla
        grid.write(file_out, format='NETCDF4')
        print(f"📦 Grilla cartesiana guardada exitosamente en: {os.path.basename(file_out)}")
        
    except Exception as e:
        print(f"⚠️ Error en el algoritmo de grillado: {e}")
        return file_out, False
    
    return file_out, True

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


def crear_netcdf_acum(files_grid, path_output_acum, date_file_ini, *args, **kwargs):
    """
    Acumula las tasas de lluvia sumando de forma estática las grillas cartesianas
    y multiplicando por el delta de tiempo correspondiente.
    """
    import os
    import numpy as np
    import datetime as dt
    import re
    import netCDF4 as nc
    
    # 1. Asegurar la ruta estructural fija para la celda del mapa
    ruta_fija_mapa = "/content/salida/acumulados/RMA2"
    os.makedirs(ruta_fija_mapa, exist_ok=True)
    
    # 2. Extractor por Regex para evitar que falle si recibe una ruta larga
    date_str = str(date_file_ini)
    match = re.search(r'(\d{8}_\d{6})', date_str)
    if match:
        date_ini = dt.datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
    else:
        date_ini = dt.datetime.now()
        print("⚠️ No se pudo extraer la fecha por Regex. Usando hora actual.")
    
    print(f"⏰ Fecha base del evento: {date_ini}")
    print(f"📊 Sumando estáticamente {len(files_grid)} archivos cartesianos...")
    
    # 3. Leer y limpiar las matrices netCDF generadas por Py-ART
    lista_mats = []
    for f in sorted(files_grid):
        with nc.Dataset(f, 'r') as ds:
            # Extraemos la tasa de lluvia (rain_rate)
            rr_data = np.squeeze(ds.variables['rain_rate'][:])
            
            # BLINDAJE CRUCIAL: Convertimos cualquier máscara o NaN a 0.0 físico
            # Esto evita que los valores nulos propaguen y te borren los datos válidos
            if np.ma.isMaskedArray(rr_data):
                rr_data = rr_data.filled(0.0)
            rr_data = np.where(np.isnan(rr_data), 0.0, rr_data)
            
            lista_mats.append(rr_data)
            
    # 4. Aplicar tu ecuación original de acumulación
    # Suma de tasas instantáneas (mm/h)
    suma_tasas = np.sum(lista_mats, axis=0)
    
    # Factor temporal (dt en horas): 10 minutos / 60 minutos = 0.1666...
    intervalo_minutos = kwargs.get('acum', 10)
    intervalo_horas = intervalo_minutos / 60.0
    
    # Acumulado final en mm
    mapa_acumulado = suma_tasas * intervalo_horas
    
    # 5. Escritura del NetCDF final en el directorio oficial
    file_out = os.path.join(ruta_fija_mapa, f"RMA2_acum_{date_ini:%Y%m%d_%H%M}.nc")
    
    with nc.Dataset(file_out, 'w', format='NETCDF4') as rootgrp:
        # Definir dimensiones espaciales
        rootgrp.createDimension('time', None)
        rootgrp.createDimension('y', mapa_acumulado.shape[0])
        rootgrp.createDimension('x', mapa_acumulado.shape[1])
        
        # Crear variable principal de lluvia
        variables_acum = rootgrp.createVariable('acumulacion', 'f4', ('y', 'x'), zlib=True)
        variables_acum[:, :] = mapa_acumulado
        variables_acum.units = 'mm'
        variables_acum.long_name = f'Precipitación Acumulada Estática ({intervalo_minutos} min)'
        
        # Metadatos globales necesarios para el script del mapa
        rootgrp.description = "Archivo acumulado - Taller RQPE"
        rootgrp.timestamp = date_ini.strftime('%Y-%m-%d %H:%M:%S')

    print(f"📦 ¡Archivo acumulado consolidado con éxito! Guardado en: {file_out}")
    
    # Retorno coordinado con la ejecución en bucle de tu RQPE_main.py
    return file_out, True

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
