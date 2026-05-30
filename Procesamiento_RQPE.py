#!/home/mrugna/mambaforge/envs/py311_rqpe/bin/python
#/usr/bin/env python3

"""
Created on Wed Jul 26 15:48:48 2023

@author: ushio_mac

Funciones incluidas

lat_lon_radar
Files_ini_fin
Archivos_rma
Reducir_cfradial
check_wet_radome
echotop
QC_zh_phidp
_det_system_phase
_add_phidp_to_radar_object
_add_echotop_to_radar_object
"""

from pathlib import Path
import numpy as np
import pyart
import copy
import glob
import os
import datetime as dt


def lat_lon_radar(radar):

    if radar == 'RMA1':
        lat_radar, lon_radar = -31.4413, -64.1919
    elif radar == 'RMA2':
        lat_radar, lon_radar = -34.8008, -58.5156

    return lat_radar, lon_radar

def lista_archivos_zh(date_ini, date_fin, path_files, radar, uso_netcdf):

    # Primero busco si hay archivos DBZH disponibles en el rango horario
    # Aprovecho que hay 2 horas de datos en ms030

    path_datos = Path(f'{path_files}/{radar}')

    #if not path_datos.exists():
    #    path_datos = Path(f'{path_files}')

    lista_zh = []

    if uso_netcdf:
        pass
    elif not radar.startswith('RMA'):
        pass
    else:
        for i in sorted(path_datos.glob('**/*_01_DBZH*.H5')):
            dt_file = dt.datetime.strptime(i.stem.split('_')[-1], '%Y%m%dT%H%M%SZ')
            if date_ini < dt_file < date_fin:
                lista_zh.append(i)

    return lista_zh


def armo_cfradial_base(file_zh, path_output_red):
    # armo el nombre del archivo intermedio
    nombre = file_zh.stem.split('_')
    radar = nombre[0]
    fechahora = nombre[-1][:-1].replace('T', '_')
    est = nombre[1] + '_' + nombre[2]

    file = Path(f'{path_output_red}/{radar}/{radar}_{fechahora}_{est}.nc')
    print(file)

    # chequeo si el archivo ya se generó para otro tiempo
    if file.exists():
        print(f'{file} existe.')
        return file, False

    # chequeo si están las 3 variables necesarias
    file_rhohv = Path(str(file_zh).replace('DBZH', 'RHOHV'))
    file_phidp = Path(str(file_zh).replace('DBZH', 'PHIDP'))

    if not file_phidp.exists() and not file_rhohv.exists():
        return file, False

    radar_zh = pyart.aux_io.read_sinarame_h5(str(file_zh), file_field_names=True)
    radar_phidp = pyart.aux_io.read_sinarame_h5(file_phidp, file_field_names=True)
    radar_rhohv = pyart.aux_io.read_sinarame_h5(file_rhohv, file_field_names=True)
    rhohv_new = copy.deepcopy(radar_rhohv.fields['RHOHV']['data'])
    phidp_new = copy.deepcopy(radar_phidp.fields['PHIDP']['data'])

    radar_zh.add_field_like('DBZH', 'RHOHV', rhohv_new)
    radar_zh = _add_phidp_to_radar_object(phidp_new, radar_zh, mask_field='RHOHV')

    radar_zh.metadata['wet_radome'] = str(check_wet_radome(radar_zh))  # esto se guarda en el nc?

    radar_zh = echotop(radar_zh, radar_zh)

    # Aca saco elevaciones para hacer más rápido
    radar_zh = radar_zh.extract_sweeps([1, 2])

    pyart.io.write_cfradial(str(file), radar_zh)

    return file, True


##############################################################################

"""
def Files_ini_fin(date_ini, date_fin, path_files, radar):
  
   # Esta funcion busca los archivos en path_files que están dentro del rango horario pedido.

    

    date_ini_str = dt.datetime.strftime(date_ini - dt.timedelta(minutes=1), '%Y%m%d_%H%M')
    file_ini = glob.glob(f'{path_files}/cfrad.{date_ini:s%Y%m%d_%H%M}*{radar}*_0*.nc')
    if file_ini[0].endswith('_03.nc') or file_ini[0].endswith('_04.nc'):
        file_ini = []

    if file_ini:
        date_file_ini = date_ini_str
    else:
        date_list = [date_ini - dt.timedelta(minutes=n) for n in range(2, 15)]
        i = 0
        while (not file_ini) and (i<len(date_list)):
            date_file_ini = dt.datetime.strftime(date_list[i], '%Y%m%d_%H%M')
            file_ini = glob.glob(f'{path_files}/cfrad.{date_list[i]:s%Y%m%d_%H%M}*{radar}*_0*.nc')
            if file_ini[0].endswith('_03.nc') or file_ini[0].endswith('_04.nc'):
                file_ini = []
            i+=1

    if not file_ini:
        raise Exception("No se encuentra file_ini")

    date_fin_str = dt.datetime.strftime(date_fin + dt.timedelta(minutes=1), '%Y%m%d_%H%M')
    file_fin = glob.glob(f'{path_files}/cfrad.{date_fin:%Y%m%d_%H%M}*{radar}*_0*.nc')

    if file_fin:
        date_file_fin = date_fin_str
    else:
        date_list = [date_fin + dt.timedelta(minutes=n) for n in range(2, 15)]
        i = 0
        while (not file_fin) and (i<len(date_list)):
            date_file_fin = dt.datetime.strftime(date_list[i], '%Y%m%d_%H%M')
            file_fin = glob.glob(f'{path_files}/cfrad.{date_list[i]:s%Y%m%d_%H%M}*{radar}*_0*.nc')
            if file_ini[0].endswith('_03.nc') or file_ini[0].endswith('_04.nc'):
                file_ini = []
            i+=1

    if not file_fin:
        raise Exception("No se encuentra file_fin")

    return date_file_ini, date_file_fin


def Archivos_rma(date_file_ini, date_file_fin, path_files, radar):

    date_ini = dt.datetime.strptime(date_file_ini, '%Y%m%d_%H%M')
    date_fin = dt.datetime.strptime(date_file_fin, '%Y%m%d_%H%M')

    periodo = date_fin - date_ini

    date_list = [date_ini + dt.timedelta(minutes=n) for n in range(0, int((periodo.total_seconds()/60)+1))]

    files = []

    for date in date_list:
        date_str = dt.datetime.strftime(date,'%Y%m%d_%H%M')
        g = glob.glob(f'{path_files}/cfrad.{date:s%Y%m%d_%H%M}*{radar}*_0*.nc')
        if g and (g[0].endswith('_03.nc') or g[0].endswith('_04.nc')):
            g = []
        files = files + g

    return files
"""
def Files_ini_fin(date_ini, date_fin, path_files, radar):
    import glob, os
    archivos_ordenados = sorted(glob.glob(os.path.join(path_files, "*.nc")))
    if not archivos_ordenados:
        raise FileNotFoundError(f"No hay archivos NetCDF en {path_files}")
    file_ini = archivos_ordenados[0]
    file_fin = archivos_ordenados[-1]
    print(f"📊 Secuencia Detectada: {len(archivos_ordenados)} archivos listos para procesar.")
    return file_ini, file_fin

def Archivos_rma(date_file_ini, date_file_fin, path_files, radar):
    import glob, os
    archivos_ordenados = sorted(glob.glob(os.path.join(path_files, "*.nc")))
    return archivos_ordenados


def Reducir_cfradial(file, path_output, radar, overwrite=False):

    print('Reduciendo cfradial...')

    # Nombre del archivo de salida

    file_str = file.split('/')[-1]
    file_out = path_output.joinpath(f'{radar}/{file_str[:-3]}_red.nc')#path_output+file_str[:-3]+'_red.nc'

    if not os.path.exists(path_output):
      #  os.system('mkdir '+path_output)
        os.system('mkdir ' + str(path_output))

    time = file_str[6:14]

    if (os.path.exists(file_out)) and not overwrite:
        print(f'{file_out} existe.')
        return file_out, time
    print('Sigo adentro de reducir.')

    # Abrimos el archivo
    radar = pyart.io.read(file)
    disponibles = radar.fields.keys()

    # Variables que tienen que estar si o si para el cálculo de la RQPE
    if ('PHIDP' in disponibles) and ('RHOHV' in disponibles) and ('DBZH' in disponibles):
        # Hay que corregir la máscara de Phidp que queda mal al convertir de BUFR a cfradial
        Phidp_new = copy.deepcopy(radar.fields['PHIDP']['data'])

        radar = _add_phidp_to_radar_object(Phidp_new, radar, mask_field='RHOHV')

        # Creamos una copia del objeto radar
        radar_new = copy.deepcopy(radar)

        # Pongo el echotop acá porque necesito todas las elevaciones. Chequear.
        radar_new = echotop(radar, radar_new)

        # Vamos a extraer solo la primera elevacion
        start_index = radar.sweep_start_ray_index['data'][1]
        end_index = radar.sweep_end_ray_index['data'][1]

        azimuth = radar_new.azimuth['data'][start_index:end_index+1]
        elevation = radar_new.elevation['data'][start_index:end_index+1]

        radar_new.azimuth['data'] = azimuth
        radar_new.elevation['data'] = elevation

        radar_new.fixed_angle['data'] = radar_new.fixed_angle['data'][1]

        radar_new.nrays = 360
        radar_new.nsweeps = 1

        radar_new.sweep_start_ray_index['data'] = np.array([0])
        radar_new.sweep_end_ray_index['data'] = np.array([359])
        radar_new.sweep_mode['data'] = radar_new.sweep_mode['data'][1,:]
        radar_new.sweep_number['data'] = radar_new.sweep_number['data'][1]
        radar_new.time['data'] = radar_new.time['data'][start_index:end_index+1]

        radar_new.metadata['wet_radome'] = str(check_wet_radome(radar))  # esto se guarda en el nc?

        for variable in disponibles:
             Var_elev1 = radar.fields[variable]['data'][start_index:end_index+1]
             radar_new.fields[variable]['data'] = Var_elev1

        #radar_new = echotop(radar, radar_new)

        # radar_new = radar_new.extract_sweeps([1, 2])  # esto no parece ir aca... arriba se saca una sola elevacion
        pyart.io.cfradial.write_cfradial(file_out, radar_new, format='NETCDF4')

    else:
        return 'Faltante', time

    return file_out, time


##############################################################################


"""
Esta funcion puede fallar en casos con atenuación muy
fuerte y tal vez sea necesario agregar una condición
extra que revise la extensión areal de la lluvia.
Como los casos más extremos son donde hay extinción
más cerca del radar, va a ser necesario extender el
flag para agregar al QC en estos casos.
"""
def check_wet_radome(radar):
    Near_radar = np.where(radar.range['data']<=10000)
    Zh = copy.deepcopy(radar.fields['DBZH']['data'])
    Znr = Zh[:, Near_radar[0]]
    Znr_mean = np.mean(Znr)
    if Znr_mean > 20:
        return True
    else:
        return False


def echotop(radar, radar_new):

    ranges = radar.range['data']
    elevations = radar.fixed_angle['data']
    Zh_2 = radar.fields['DBZH']['data'].copy()

    elevs = elevations.shape[0]
    n_azi = Zh_2.shape[0]
    n_range = Zh_2.shape[1]

    ranges = np.repeat(ranges[np.newaxis, :], 360, axis=0)
    ranges = np.repeat(ranges[np.newaxis, :, :], elevs, axis=0)

    elevations = np.repeat(elevations[:, np.newaxis], 360, axis=1)
    elevations = np.repeat(elevations[:, :, np.newaxis], n_range, axis=2)

    Re = 6371.0 * 1000.0
    p_r = 4.0 * Re / 3.0
    z = (ranges ** 2 + p_r ** 2 + 2.0 * ranges * p_r *
                        np.sin(elevations * np.pi / 180.0)) ** 0.5 - p_r
    Zh_2 = np.reshape(Zh_2, (elevs, 360, n_range))
    Zh_2 = Zh_2.filled(-9999.)

    echotop = np.zeros([360, n_range])  # en vez de 360 puedo pasarle n_azi y queda solo la primera elevacion con los datos de echotop
    echo_ini = np.where(Zh_2[1, :, :]>0.)
    echo_ini = [(echo_ini[0][i], echo_ini[1][i]) for i in range(len(echo_ini[0]))] 

    for echo in echo_ini:
        k = 2
        echo_true = True
        while (echo_true) and (k<elevs-1):
            if Zh_2[k,echo[0],echo[1]] < 10.:
                echotop[echo[0], echo[1]] = z[k-1, echo[0], echo[1]]
                echo_true = False
            else:
                echotop[echo[0], echo[1]] = z[k, echo[0], echo[1]]
                k+=1

    radar_new = _add_echotop_to_radar_object(np.tile(echotop, (elevs, 1)),
                                             radar_new, mask_field='RHOHV')

    

    return radar_new


def echotop2(radar, radar_new):

    #ranges = radar.range['data']
    elevations = radar.fixed_angle['data']
    Zh_2 = radar.fields['DBZH']['data'].copy()

    elevs = elevations.shape[0]
    n_azis = Zh_2.shape[0]
    n_range = Zh_2.shape[1]

    """ranges = np.repeat(ranges[np.newaxis, :], 360, axis=0)
    ranges = np.repeat(ranges[np.newaxis, :, :], elevs, axis=0)

    elevations = np.repeat(elevations[:, np.newaxis], 360, axis=1)
    elevations = np.repeat(elevations[:, :, np.newaxis], n_range, axis=2)

    Re = 6371.0 * 1000.0
    p_r = 4.0 * Re / 3.0
    z = (ranges ** 2 + p_r ** 2 + 2.0 * ranges * p_r *
                        np.sin(elevations * np.pi / 180.0)) ** 0.5 - p_r"""

    z = radar.gate_z['data']
    z = np.reshape(z, (elevs, n_azis, n_range))


    Zh_2 = np.reshape(Zh_2, (elevs, n_azis, n_range))

    Zh_2 = Zh_2.filled(-9999.)

    echotop = np.zeros([elevs, n_azis, n_range])

    echo_ini = np.where(Zh_2[:,:,:]>0.)

    echo_ini = [(echo_ini[0][i], echo_ini[1][i]) for i in range(len(echo_ini[0]))] 

    for echo in echo_ini:
        k = 0
        echo_true = True

        while (echo_true) and (k<elevs-1):
            if Zh_2[k,echo[0], echo[1]] < 10.:
                echotop[echo[0], echo[1]] = z[k-1, echo[0], echo[1]]
                echo_true = False
            else:
                echotop[echo[0], echo[1]] = z[k, echo[0], echo[1]]
                k+=1

    radar_new = _add_echotop_to_radar_object(echotop, radar_new, mask_field='RHOHV')

    return radar_new

"""
def QC_zh_phidp(file_red, path_output_qc, radar, overwrite=False):
    import QC_RQPE

    print('Haciendo QC...')

    a = 0.112
    b_coef = 0.675

    nombre = file_red.stem[:-4]#file_red.stem.split('_')
    nombre_radar = radar #nombre[5] #nombre[0] # esto es cuando es operativo..

    file = Path(f'{path_output_qc}/{nombre_radar}/{nombre}_qc.nc')
    #print(file)

    # chequeo si el archivo ya se generó para otro tiempo
    if file.exists():
        print(f'{file} existe.')
        return file, False

    #file_str = file.split('/')[-1]
    #file_out = path_output+file_str[:-7]+'_qc.nc'

    #if not os.path.exists(path_output):
    #    os.system('mkdir '+path_output)

    #if (os.path.exists(file_out)) and not overwrite:
    #    return file_out

    radar = pyart.io.read(file_red)
    radar = QC_RQPE.make_QC_ZH(radar)
    radar, Phidp_mask = QC_RQPE.make_QC_PHIDP(radar)
    sys_phase = _det_system_phase(radar)

    print(sys_phase)

    radar = QC_RQPE.unfold_and_calc_kdp(radar, Phidp_mask, sys_phase=sys_phase , min_rhv=0.7)
    radar = QC_RQPE.correc_zphi(radar, a, b_coef)
    try:
        pyart.io.write_cfradial(str(file), radar, format='NETCDF4')
    except AttributeError as e:
        print(e)
        # hay un problema con cftime y real_datetime junto con numpy..
        return file, True

    return file, True
"""

def QC_zh_phidp(file_input_red, path_output_qc, *args, **kwargs):
    """
    Control de Calidad adaptado. Absorbe cualquier combinación de argumentos
    (como path_statics o variables de clutter) de forma flexible.
    """
    import pyart
    import os
    import numpy as np
    import QC_RQPE as qc  # Asegúrate de importar tu módulo de QC moderno
    
    nombre_base = os.path.basename(file_input_red)
    file_out = os.path.join(path_output_qc, nombre_base.replace('_red.nc', '_qc.nc'))
    
    # 1. Leer el volumen reducido
    radar_obj = pyart.io.read(file_input_red)
    
    # Asegurar que exista la variable base para el grillado posterior
    if 'DBZH_nomask' not in radar_obj.fields:
        radar_obj.add_field_like('DBZH', 'DBZH_nomask', radar_obj.fields['DBZH']['data'].copy(), replace_existing=True)
    
    # 2. CÁLCULO REAL: Desenvuelto y Kdp moderno basado en SciPy
    radar_obj = qc.unfold_and_calc_kdp(radar_obj, Phidp_mask=None, sys_phase=0.0, min_rhv=0.7)
    
    # 3. Guardar el archivo de volumen limpio con datos físicos reales
    pyart.io.cfradial.write_cfradial(file_out, radar_obj, format='NETCDF4')
    print(f"🧼 [QC Científico Completo] Archivo guardado exitosamente en: {os.path.basename(file_out)}")
    
    return file_out, True

def _det_system_phase(radar):

    Phidp = radar.fields['PHIDP_nomask']['data'].copy()
    Phidp = Phidp.data
    phases = []
    for radial in range(360):
        mpts = np.where(Phidp[radial,:]!=-9999.)
        if len(mpts[0]) > 25:
            msmth_phidp = pyart.correct.phase_proc.smooth_and_trim(Phidp[radial, mpts[0]], 9)
            phases.append(msmth_phidp[0:25].min())

    if np.isnan(np.median(phases)):
        return 0.
    else:
        return np.median(phases)


def _add_phidp_to_radar_object(field, radar, field_name='PHIDP_nomask', units='degrees', 
                               long_name='Differential phase', standard_name='differential_phase_hv',
                               mask_field='dBZ'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -9999.0
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[mask_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[mask_field]['data'].mask))
        fill_value = radar.fields[mask_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

def _add_echotop_to_radar_object(field, radar, field_name='Echotop', units='meters', 
                                 long_name='Echotop height', standard_name='echotop_height',
                                 mask_field='dBZ'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -9999.0
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[mask_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[mask_field]['data'].mask))
        fill_value = radar.fields[mask_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar
