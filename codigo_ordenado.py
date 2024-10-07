from NasaSpaceApp.funciones import *

# Preferably run Google Collab
nombre_planeta = 'HD 209458 b'
dec, ra = coordenada_exoplaneta(nombre_planeta)
host_coord = obtener_anfitrion(ra, dec)
df = obtener_estrellas(*host_coord)
graficar(df, host_coord)
guardar_achivo(nombre_planeta+".csv", df, host_coord)
