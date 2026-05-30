#!/home/mrugna/mambaforge/envs/py311_rqpe/bin/python
#/usr/bin/env python3
"""
Created on Fri Jul 28 12:58:12 2023

@author: ushio_mac
"""

from pathlib import Path
import datetime as dt
import argparse
import pickle
import numpy as np

import Procesamiento_RQPE as Proc
import Calc_RQPE as Calc


parser = argparse.ArgumentParser(description="Calculo RQPE",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--path_archivos",
                    default='/ms030/mrugna/RMA/datos/', help="Ruta de archivos de radar")
parser.add_argument("-nc", action="store_true",
                    help="Usar archivos netcdf en vez de H5. Funciona en diferido.")
parser.add_argument("-o", "--path_output",
                    default='/data/mrugna/RQPE', help="Ruta de salida de archivos")
parser.add_argument("-res", default=2, type=float,
                    help="Resolución espacial en kilometros del grillado")
parser.add_argument("-rango",default=150, type=float,
                    help="Rango maximo en kilometros")
parser.add_argument("-acum", default=1, type=int, help="Minutos de acumulado")
parser.add_argument("-radar", required=True, help="Nombre del radar")
parser.add_argument("-fecha_ini", required=True,
                    type=lambda d: dt.datetime.strptime(d, '%Y%m%d_%H%M'),
                    help="Tiempo inicial de acumulacion en formato YYYYMMDD_HHMM")
parser.add_argument("-fecha_fin", required=True,
                    type=lambda d: dt.datetime.strptime(d, '%Y%m%d_%H%M'),
                    help="Tiempo final de acumulacion en formato YYYYMMDD_HHMM")
parser.add_argument("-qc", action="store_true",
                    help="Sobreescribir QC si el archivo ya existe")
parser.add_argument("-red", action="store_true",
                    help="Sobreescribir reduccion si el archivo ya existe")
parser.add_argument("-gr", action="store_true",
                    help="Sobreescribir grillado si el archivo ya existe")
parser.add_argument("-qpe", action="store_true",
                    help="Sobreescribir calculo de qpe si el archivo ya existe")
args = parser.parse_args()
config = vars(args)

path_files = config['path_archivos']
uso_netcdf = config['nc']
path_output = Path(config['path_output'])

path_output_red = path_output.joinpath('datos/')
path_output_qc = path_output.joinpath('datos_corregidos/')
path_output_qpe = path_output.joinpath('QPE/')
path_output_grid = path_output.joinpath('QPE_grid/')
path_output_acum = path_output.joinpath('QPE_acum/')

date_ini = config['fecha_ini']
date_fin = config['fecha_fin']

min_acum = config['acum']

rango = config['rango']

res = config['res']

radar = config['radar']

lat_radar, lon_radar = Proc.lat_lon_radar(radar)

#print(lat_radar, lon_radar)
#print(path_output_red)




if uso_netcdf:
    """
    Esta funcion hace un glob en el path_files en conjunto con los rangos temporales
    pero por ahora solo busca netcdf. Hay que reciclar la funcion que arma el
    objeto radar con los H5.
    """
    date_file_ini, date_file_fin = Proc.Files_ini_fin(date_ini, date_fin, path_files)
    #print(date_file_fin)
    #print(config['path_archivos'])
    """
    Esta funcion busca todos los archivos, minuto por minuto.
    Se podria hacer todo en la primera funcion. Devuelve una lista de archivos.
    - El tema acá es que hay que levantar una excepción cuando hay muchos archivos
    faltantes dentro de la ventana que uno pide.
    """
    files = Proc.Archivos_rma(date_file_ini, date_file_fin, path_files)
else:
    files = Proc.lista_archivos_zh(date_ini, date_fin, path_files, radar, uso_netcdf)

files_grid = []

if config['red']:
    config['qc'] = True
    config['qpe'] = True
    config['gr'] = True

elif config['qc']:
    config['qpe'] = True
    config['gr'] = True

elif config['qpe']:
    config['gr'] = True

#print(files)
print(config)
print()

for file in files:
    print(file)


    if uso_netcdf:
        """
        Teniendo el netcdf del backup diario se genera uno con menos variables
        y menos elevaciones. Se agrega también el flag de radomo mojado y el echotop.
        """
        file_output_red, time = Proc.Reducir_cfradial(file, path_output_red, radar,
                                                      overwrite=config['red'])
        if file_output_red == 'Faltante':
            continue
    else:
        """
        Esta función arma un archivo nuevo
        se revisa si están las 3 variables necesarias, phi, rho y zh
        Aca se agrega una variable de radomo mojado y se usa la funcion echotop
        
        wet_radome revisa si en los 10 km más cercanos al radar el promedio de
        reflectividad es mayor o menor a 20 dBZ

        echotop busca en la matriz 3D de ZH iterando por filas y columnas
        el primer valor de 10 dBZ.
        """
        file_output_red, flag = Proc.armo_cfradial_base(file, path_output_red)


    """
    Esta funcion crea otro archivo a partir del archivo que se creo con
    Reducir_cfradial. 
    Adentro llama a QC_RQPE.make_QC_ZH que levanta echotop y wet_radome para
    hacer un primer filtro. Parece algo que hizo Diego.
    Luego llama a QC_RQPE.make_QC_PHIDP que tiene valores levemente diferentes
    para hacer el QC pero que devuelve algo muy parecido a make_QC_ZH.
    Finalmente calcula la fase del sistema y corrige con funciones
    de pyart modificadas.
    - Hay que volver a bajar el archivo de pyart modificado.
    - Revisar en el resto que hay líneas que agregar.
    """
    file_output_qc, flag = Proc.QC_zh_phidp(file_output_red, path_output_qc, radar,
                                            overwrite=config['qc'])

    """
    Esta función crea otro archivo y calcula al mismo tiempo el QPE con
    ZH y luego con KDP, ambos corregidos con la funcion anterior,
    usando relaciones. Enmascara con RHOHV que viene sin máscara desde el radar.
    Recordar que acá se está usando el Z-R del RMA1 al momento de hacer las
    comparaciones con la versión de Diego.
    """
    file_output_qpe, flag = Calc.RQPE_simple_doble(file_output_qc, path_output_qpe, radar,
                                                   overwrite=config['qpe'])

    """
    Con el archivo de la función anterior escribe un netcdf grillado.
    Genera una grilla de 2 km de forma automática a partir de los 2 campos
    de QPE generados previamente.
    """
    file_out_grid, flag = Calc.Grid_RQPE(file_output_qpe, path_output_grid, radar,
                                         resolucion=res, radar_range=rango,
                                         overwrite=config['gr'])

    files_grid.append(file_out_grid)

    print()

"""
Esta funcion agarra la lista de archivos grillados y los limites temporales
definidos más arriba.
Adentro usa las funciones Acum_1m_[simple doble] que adentro usan funciones
para corregir por advección.
Revisar el tema del periodo temporal y el pasaje de mm/h a mm.
Luego elimina los datos que estén por fuera del rango del radar con remove_corners
Finalmente crea un último archivo solo con la QPE.
"""
Calc.crear_netcdf_acum(files_grid, path_output_acum, date_file_ini,
                       date_file_fin, date_ini, date_fin,
                       lat_radar, lon_radar, radar, min_acum=min_acum)
