from pathlib import Path
from datetime import datetime, timedelta
import subprocess

conda_env = 'py311_rqpe'
py = f'/home/mrugna/mambaforge/envs/{conda_env}/bin/python'
main = '/ms030/mrugna/RMA/scripts/rqpe_maite/RQPE_main.py'
radar = 'RMA1'
path_in = f'/data/mrugna/RQPE/entrada/{radar}'
fecha_cero = datetime(2024, 11, 27, 12, 0)
#fecha_cero = datetime(2024, 3, 18, 20, 30)
acum = 10

rango = 24*6*4 + 7*3  # 24 horas * X minutos * X cantidad de días + X+1 divisiones * X cantidad
# minutos: si el acum es 30 entonces es 2, si el acum es 10 entonces es 6
# divisiones: 2+1 si es 30 minutos, 6+1 si es 10 minutos

for i in range(rango):
    fecha_ini = fecha_cero + timedelta(minutes=acum*i)
    fecha_fin = fecha_ini + timedelta(minutes=acum)
    interprete = f'mamba run -n {conda_env} python {main}'  # f'time {py} {main}'
    comando = f'{interprete} -nc -radar {radar} -i {path_in} -fecha_ini {fecha_ini:%Y%m%d_%H%M} -fecha_fin {fecha_fin:%Y%m%d_%H%M} -acum {acum}'
    print(comando)
    subprocess.run(comando, shell=True)
    #print()

"""fecha_ini = fecha_cero #+ timedelta(minutes=30*i)
fecha_fin = fecha_ini + timedelta(minutes=60*24)
comando = f'time {py} {main} -nc -radar RMA1 -i {path_in} -fecha_ini {fecha_ini:%Y%m%d_%H%M} -fecha_fin {fecha_fin:%Y%m%d_%H%M} -acum {60*24}'
print(comando)
subprocess.run(comando, shell=True)"""
