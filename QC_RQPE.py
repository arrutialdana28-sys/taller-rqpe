#!/home/mrugna/mambaforge/envs/py311_rqpe/bin/python
#/usr/bin/env python3

"""
Created on Wed Jul 26 17:45:43 2023

@author: ushio_mac

Funciones incluidas

make_QC_ZH
make_QC_PHIDP
unfold_and_calc_kdp
correc_zphi
_add_phidp_to_radar_object
_add_zh_to_radar_object
_add_kdp_to_radar_object
"""

import numpy as np
import pyart
import wradlib as wrl 
import copy
import pyart_funciones_modificadas as pyart_mod


def make_QC_ZH(radar_qc):

    Zh = radar_qc.fields['DBZH']['data'].copy()
    radar_qc = _add_zh_to_radar_object(Zh, radar_qc, mask_field='RHOHV')

    echotop = radar_qc.fields['Echotop']['data'][:]

    wet_rad = radar_qc.metadata['wet_radome']

    Rhohv = radar_qc.fields['RHOHV']['data'].copy()

    if wet_rad in ['False']:  # ???
        Zh[echotop<1500] = np.nan

    # Sacamos los pixeles con rhohv bajos y regiones de reflectividad baja y rhohv menores a 0.8
    Zh[(Rhohv<0.7)] = np.nan
    Zh[(Rhohv<0.8) & (Zh<10)] = np.nan

    # Calculamos la textura de rhohv y sacamos en Phidp y en Zh los valores altos de textura
    rhot = wrl.dp.texture(Rhohv)
    rhot_thr = 0.3
    Zh[rhot > rhot_thr] = np.nan

    # Aplicamos el filtro de gabella en Zh y removemos en Phidp
    clmap = wrl.classify.filter_gabella(Zh, wsize=5, thrsnorain=0.0, tr1=20., n_p=6, tr2=1.3, rm_nans=False)
    Zh[clmap] = np.nan

    # Aplicamos depeckle
    Zh = wrl.util.despeckle(Zh, n=3, copy=False)

    # Enmascaramos y guardamos en el objeto radar
    Zh[np.isnan(Zh)] = -9999.
    Zh = np.ma.masked_where(Zh==-9999., Zh)

    radar_qc.fields['DBZH_nomask']['data'][:] = Zh

    return radar_qc


def make_QC_PHIDP(radar, sys_phase=0):

    wet_rad=radar.metadata['wet_radome']

    if wet_rad in ['False']:
        Phidp = radar.fields['PHIDP_nomask']['data'].copy()
        radar = _add_phidp_to_radar_object(Phidp, radar,mask_field='DBZH')

    Phidp = radar.fields['PHIDP_nomask']['data'][:].copy()
    Rhohv = radar.fields['RHOHV']['data'][:].copy()
    Zh = radar.fields['DBZH']['data'][:].copy()

    Phidp = Phidp.filled(np.nan)
    Zh = Zh.filled(np.nan)

    Phidp[Phidp==-9999.] = np.nan
    Zh[Zh==-9999.] = np.nan

    echotop = radar.fields['Echotop']['data'][:]

    Zh[echotop<2000] = np.nan
    Phidp[echotop<2000] = np.nan

    Phidp_m = Phidp.copy()

    # Sacamos los pixeles con rhohv bajos y regiones de reflectividad baja y rhohv menores a 0.95
    Phidp[(Rhohv<0.7)] = np.nan
    Phidp[(Rhohv<0.8) & (Zh<10)] = np.nan

    Phidp_m[(Rhohv<0.7)] = np.nan
    Phidp_m[(Rhohv<0.8) & (Zh<10)] = np.nan

    # Calculamos la textura de rhohv y sacamos en Phidp y en Zh los valores altos de textura
    rhot = wrl.dp.texture(Rhohv)

    if wet_rad in ['True']:
        rhot_thr = 0.1
    else:
        rhot_thr = 0.15

    Phidp[rhot > rhot_thr] = np.nan
    Phidp_m[rhot > rhot_thr] = np.nan
    Zh[rhot > rhot_thr] = np.nan

    # Aplicamos el filtro de gabella en Zh y removemos en Phidp
    clmap = wrl.classify.filter_gabella(Zh, wsize=5, thrsnorain=0.0, tr1=20., n_p=6, tr2=1.3, rm_nans=False)
    Phidp[clmap] = np.nan
    Phidp_m[clmap] = np.nan

    Phidp_mask = np.zeros_like(Phidp_m)
    Phidp_mask[Phidp_m>0] = 1

    # Aplicamos depeckle
    Phidp = wrl.util.despeckle(Phidp, n=3, copy=False)

    # Enmascaramos y guardamos en el objeto radar
    Phidp[np.isnan(Phidp)] = -9999.

    masked_field = np.ma.asanyarray(Phidp)
    masked_field.mask = masked_field == -9999.

    radar.fields['PHIDP_nomask']['data'] = masked_field

    return radar, Phidp_mask


def unfold_and_calc_kdp(radar, Phidp_mask, sys_phase=0., min_rhv=0.7):

    phidp, kdp = pyart_mod.phase_proc_lp(radar, 0,sys_phase=sys_phase, overide_sys_phase=True, refl_field='DBZH_nomask', 
                                             ncp_field='RHOHV', rhv_field='RHOHV', phidp_field='PHIDP_nomask', 
                                             min_rhv=min_rhv, nowrap=None, fzl=15000., LP_solver='pyglpk', 
                                             window_len=15,high_z=70., coef=0.84)

    phidp['valid_max'] = np.nanmax(phidp['data'][:])

    radar.add_field('corrected_differential_phase', phidp, replace_existing=True)
    radar.add_field('corrected_kdp', kdp, replace_existing=True)

    #start_index = radar.sweep_start_ray_index['data'][0]
    #end_index = radar.sweep_end_ray_index['data'][0] + 1

    kdp_values = radar.fields['corrected_kdp']['data'].copy()
    kdp_values[Phidp_mask==0] = -9999.

    masked_field = np.ma.asanyarray(kdp_values)
    masked_field.mask = masked_field == -9999.

    radar.fields['corrected_kdp']['data'] = masked_field

    return radar

def correc_zphi(radar, a, b):

    Atenua_espec, Z_correc_zphi = pyart_mod.calculate_attenuation(radar, 0.0, refl_field='DBZH_nomask', ncp_field='RHOHV', rhv_field='RHOHV',
                                                                  phidp_field='corrected_differential_phase', fzl=5000, a_coef=a, beta=b, rhv_min=0.7)

    radar.add_field('dBZ_correc_zphi', Z_correc_zphi, replace_existing=True)
    radar.add_field('spec_attenuation', Atenua_espec, replace_existing=True)

    return radar


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

def _add_zh_to_radar_object(field, radar, field_name='DBZH_nomask', units='dBZ', 
                            long_name='Reflectivity_2', standard_name='equivalent_reflectivity_factor_2',
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

def _add_kdp_to_radar_object(field, radar, field_name='KDP_nomask', units='degress/km', 
                             long_name='Specific dofferential phase (KDP)', standard_name='specific_differential_phase_hv',
                             mask_field='PHIDP_nomask'):
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

