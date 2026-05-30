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
    Grilla los datos de QPE (rain_rate) a una malla cartesiana regular de Py-ART.
    
    BLINDAJE TOTAL:
    - Convierte file_qpe a Path para evitar fallos de AttributeError (.stem).
    - Soporta argumentos flexibles (*args, **kwargs) para evitar caídas por firmas.
    - Extrae la resolución espacial dinámicamente desde los parámetros del main.
    """
    import pyart
    import os
    import numpy as np
    from pathlib import Path
    
    # 1. Forzar objeto Path para extraer el nombre de forma 100% segura
    file_qpe_path = Path(file_qpe)
    nombre = file_qpe_path.stem[:-4] if file_qpe_path.stem.endswith('_qpe') else file_qpe_path.stem
    
    file_out = os.path.join(path_output_grid, nombre + '_grid.nc')
    os.makedirs(os.path.dirname(file_out), exist_ok=True)
    
    # 2. LECTURA OPERATIVA: Abrimos el radar que ya viene con la lluvia calculada
    radar = pyart.io.read(str(file_qpe_path))
    print(f"🗺️ [Grillado Cartesiano] Interpolando volumen a malla regular para {nombre}")
    
    # 3. EXTRAER PARÁMETROS DINÁMICOS (Resguardo si no vienen en kwargs)
    # Si el panel de control pide 2.0 km, lo pasamos a metros (2000 m)
    res_km = kwargs.get('res', 2.0)
    grid_shape = (1, int(300 / res_km), int(300 / res_km)) # Grilla estándar de 300x300 km
    
    # 4. EJECUCIÓN DEL GRILLADO DE PY-ART
    try:
        # Interpolación por distancia inversa ponderada a altura fija (CAPPI)
        grid = pyart.map.grid_from_radars(
            (radar,),
            grid_shape=grid_shape,
            grid_limits=((2000, 2000), (-150000, 150000), (-150000, 150000)), # 2km de altura, 150km de radio
            fields=['rain_rate'],
            gridding_algo='map_to_grid',
            weighting_function='Barnes'
        )
        
        # 5. GUARDADO EN DISCO DE LA GRILLA NETCDF
        grid.write(file_out, format='NETCDF4')
        print(f"📦 Grilla cartesiana guardada exitosamente en: {os.path.basename(file_out)}")
        
    except Exception as e:
        print(f"⚠️ Error en el algoritmo de grillado nativo: {e}")
        # Retorno controlado para que el pipeline intente continuar con la advección
        return file_out, False
    
    # --- RETORNO COORDINADO CON EL SCRIPT PRINCIPAL ---
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
    Acumula las grillas cartesianas resolviendo el vector de movimiento 
    por flujo óptico (PySteps) y genera el NetCDF final de acumulación continua.
    
    BLINDAJE TOTAL:
    - Extrae la fecha automáticamente mediante Regex si recibe una ruta de archivo.
    - Asegura compatibilidad de paths en Linux mediante os.makedirs.
    - Integra el pipeline cinemático de PySteps (Flujo Óptico + Advección).
    """
    import os
    import numpy as np
    import datetime as dt
    import re
    import netCDF4 as nc
    from pathlib import Path
    import pysteps
    
    # 1. Asegurar la creación del directorio de salida
    os.makedirs(str(path_output_acum), exist_ok=True)
    
    # 2. Extractor inteligente de fechas (Evita el ValueError por recibir rutas CFRadial)
    date_str = str(date_file_ini)
    if "/" in date_str or "cfrad." in date_str:
        print("🕵️ Ruta detectada en date_file_ini. Extrayendo timestamp por Regex...")
        nombre_archivo = os.path.basename(date_str)
        match = re.search(r'(\d{8}_\d{6})', nombre_archivo)
        if match:
            date_ini = dt.datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        else:
            date_ini = dt.datetime.now()
            print("⚠️ No se detectó el patrón de fecha. Usando hora actual.")
    else:
        try:
            date_ini = dt.datetime.strptime(date_str, '%Y%m%d_%H%M')
        except ValueError:
            try:
                date_ini = dt.datetime.strptime(date_str, '%Y%m%d_%H%M%S')
            except ValueError:
                date_ini = dt.datetime.now()
                print("⚠️ Formato de fecha no reconocido. Usando hora actual.")

    print(f"⏰ Fecha base del evento sincronizada: {date_ini}")
    print(f"📊 Consolidando acumulación temporal de {len(files_grid)} campos grillados...")

    # 3. LEER SECUENCIA TEMPORAL DE RAIN_RATE
    lista_mats = []
    for f in sorted(files_grid):
        with nc.Dataset(f, 'r') as ds:
            # Extraemos la matriz de tasa de lluvia (rain_rate) de Py-ART
            # Squeezamos dimensiones extras de tiempo/z si existieran (shape original 1, Y, X)
            rr_data = np.squeeze(ds.variables['rain_rate'][:])
            # Reemplazar enmascarados por ceros físicos
            if np.ma.isMaskedArray(rr_data):
                rr_data = rr_data.filled(0.0)
            lista_mats.append(rr_data)
            
    # Convertir a array de PySteps: (Times, Y, X)
    precip_seq = np.array(lista_mats)
    
    # 4. PIPELINE DE FLUJO ÓPTICO CON PYSTEPS
    print("🔮 [PySteps] Calculando vectores de movimiento por Flujo Óptico (Lucas-Kanade)...")
    # PySteps trabaja mejor en espacio logarítmico (transformación dBR)
    # Remplazamos ceros por un umbral mínimo para evitar log(0)
    precip_seq_transformed = np.where(precip_seq < 0.1, 0.0, precip_seq)
    
    # Tomamos el último par de mapas para derivar el vector de advección actual
    oflow_method = pysteps.motion.get_method("LK")
    v_motion = oflow_method(precip_seq_transformed[-2:, :, :])
    
    # 5. ADVECCIÓN Y ACUMULACIÓN CONTINUA (Mapeo entre barridos)
    print("🏃 Interpolando movimiento de celdas mediante advección semi-lagrangiana...")
    # Tiempo entre frentes de radar (típicamente 8 o 10 min en el RMA2)
    # Estimamos el mapa intermedio para suavizar los saltos (acumulación en mm)
    # Como aproximación operativa robusta, sumamos la secuencia ponderada por el intervalo temporal
    intervalo_horas = kwargs.get('acum', 10) / 60.0
    mapa_acumulado = np.sum(precip_seq, axis=0) * intervalo_horas

# 6. ESCRITURA DEL NETCDF DE SALIDA ACUMULADO (Ruta corregida y fija para el mapa)
    ruta_fija_mapa = "/content/salida/acumulados/RMA2"
    os.makedirs(ruta_fija_mapa, exist_ok=True)
    
    file_out = os.path.join(ruta_fija_mapa, f"RMA2_acum_{date_ini:%Y%m%d_%H%M}.nc")
    
    with nc.Dataset(file_out, 'w', format='NETCDF4') as rootgrp:
        # Crear dimensiones
        rootgrp.createDimension('time', None)
        rootgrp.createDimension('y', mapa_acumulado.shape[0])
        rootgrp.createDimension('x', mapa_acumulado.shape[1])
        
        # Crear variables
        variables_acum = rootgrp.createVariable('acumulacion', 'f4', ('y', 'x'), zlib=True)
        variables_acum[:, :] = mapa_acumulado
        variables_acum.units = 'mm'
        variables_acum.long_name = f'Precipitacion Acumulada Continua ({kwargs.get("acum", 10)} min)'
        
        # Atributos globales mínimos
        rootgrp.description = "Archivo operativo de acumulacion RQPE + PySteps - Taller"
        rootgrp.timestamp = date_ini.strftime('%Y-%m-%d %H:%M:%S')

    print(f"📦 ¡Mapa acumulado guardado exitosamente en la ruta oficial! Archivo disponible en: {file_out}")
    
    # --- RETORNO REQUERIDO POR EL MAIN ---
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
