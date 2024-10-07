import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

def coordenada_exoplaneta(nombre_planeta):
    """
    Retorna las coordenadas celestiales del exoplaneta solicitado
    :param nombre_planeta: nombre del exoplaneta según cómo aparece
    en el NASA Exoplanet Archive
    :return: declensión y ascensión recta del exoplaneta (tupla de dos valores)
    """
    query = NasaExoplanetArchive.query_criteria(
        table="PS",
        select="pl_name, hostname, ra, dec",
        where=f"pl_name='{nombre_planeta}'"
    )
    ra = query['ra'][0]  # RA obtenida de la consulta en el Exoplanet Archive
    dec = query['dec'][0]  # DEC obtenida de la consulta en el Exoplanet Archive
    return dec, ra

def obtener_anfitrion(ra, dec, radio=0.0001):
    """
    Retorna las coordenadas de la estrella anfitriona de un exoplaneta ubicado en las
    coordenadas celestiales ra (ascensión recta) y dec (declinación).
    Se recomienda utilizar el radio por defecto siempre que sea posible. En caso de no detectarse
    ninguna estrella con las coordenadas dadas, aumentar el radio de búsqueda, si se encontrase más
    de una, disminuirlo.
    :param ra: ascensión recta del exoplaneta (en grados)
    :param dec: declinación del exoplaneta (en grados)
    :param radio: radio de busqueda en torno a la ubicación del exoplaneta (en grados)
    :return: coordenadas cartesianas de la estrella anfitriona
    """
    coord = SkyCoord(ra, dec, frame='icrs')
    radius = radio * u.degree  # Definimos un pequeño radio para la búsqueda

    # Escribimos la consulta ADQL a Gaia para obtener el source_id y más información
    query_gaia = f"""
    SELECT source_id, ra, dec, 1000/parallax AS distance_parsec
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
      POINT('ICRS', ra, dec),
      CIRCLE('ICRS', {coord.ra.degree}, {coord.dec.degree}, {radius.to(u.degree).value})
    ) = 1
    """

    # Ejecuta la consulta
    job = Gaia.launch_job(query_gaia)

    # Obtiene los resultados
    results = job.get_results()

    if len(results) == 0:
        print(f"No se encontraron estrellas en esta coordenada. "
              f"Se recomienda aumentar el radio de búsqueda")
    elif len(results) > 1:
        print(f"Se encontraron {len(results)} estrellas. "
              f"Se recomienda disminuir el radio de búsqueda")

    # Extraigo los datos de interes del objeto results
    ra_huesped = results['ra'][0]  # Cambia a la RA de la estrella huésped, en grados
    dec_huesped = results['dec'][0]  # Cambia a la Dec de la estrella huésped, en grados
    distancia_huesped = results['distance_parsec'][0]  # Distancia de la estrella huésped en parsecs

    # Convertir las coordenadas de la estrella huésped a cartesianas
    x_anfitrion = distancia_huesped * np.cos(np.radians(dec_huesped)) * np.cos(np.radians(ra_huesped))
    y_anfitrion = distancia_huesped * np.cos(np.radians(dec_huesped)) * np.sin(np.radians(ra_huesped))
    z_anfitrion = distancia_huesped * np.sin(np.radians(dec_huesped))

    return x_anfitrion, y_anfitrion, z_anfitrion

def obtener_estrellas(x_host, y_host, z_host, radio_busqueda=800):
    """
    Se busca en la base de datos Gaia (datos obtenidos por el satélite europeo del
    mismo nombre), estrellas que estén cerca de la estrella anfitriona con las
    coordenadas cartesianas suministradas (en parsecs).
    El radio de búsqueda está dada en parsecs.
    :param x_host: coordenada x de la estrella anfitriona, en parsecs, relativo a la Tierra
    :param y_host: coordenada y de la estrella anfitriona, en parsecs, relativo a la Tierra
    :param z_host: coordenada z de la estrella anfitriona, en parsecs, relativo a la Tierra
    :param radio_busqueda: distancia en parsec, a medir desde la estrella anfitriona
    :return: dataframe con las estrellas
    """
    #radio_esfera = 800  # parsecs -> 2609.25 años luz

    query_gaia = f"""
    SELECT source_id, ra, dec, parallax,
           parallax_over_error, phot_g_mean_mag,
           (1000 / parallax) AS distance_parsec, teff_gspphot
    FROM gaiadr3.gaia_source
    WHERE
        SQRT(POWER((1000 / parallax) * COS(RADIANS(ra)) * COS(RADIANS(dec)) - {x_host}, 2) +
             POWER((1000 / parallax) * SIN(RADIANS(ra)) * COS(RADIANS(dec)) - {y_host}, 2) +
             POWER((1000 / parallax) * SIN(RADIANS(dec)) - {z_host}, 2)) <= {radio_busqueda}
    AND phot_g_mean_mag < 9
    AND teff_gspphot > 1500
    """

    job = Gaia.launch_job_async(query_gaia)
    results = job.get_results()

    df = results.to_pandas()
    df["x_pc"] = df["distance_parsec"] * np.cos(np.radians(df["dec"])) * np.cos(np.radians(df["ra"]))
    df["y_pc"] = df["distance_parsec"] * np.cos(np.radians(df["dec"])) * np.sin(np.radians(df["ra"]))
    df["z_pc"] = df["distance_parsec"] * np.sin(np.radians(df["dec"]))

    print(f"Se encontraron {len(df)} estrellas")

    return df

def graficar(df, host_coord=None):
    """
    Graficación en 3D de las estrellas en el dataframe suministrado.
    Si se suministra coordenadas de la estrella anfitriona, la grafica en rojo
    en el mismo gráfico.
    :param df: dataframe con las estrellas cercanas a la estrella anfitriona
    :param host_coord: tupla de coordenadas cartesianas de la estrella anfitriona
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["x_pc"], df["y_pc"], df["z_pc"], c='b', marker='o', s=5)

    ax.set_xlabel('X (pc)')
    ax.set_ylabel('Y (pc)')
    ax.set_zlabel('Z (pc)')
    ax.set_title('Estrellas dentro de una esfera alrededor de la estrella anfitriona')

    if host_coord is not None:
        ax.scatter(host_coord[0], host_coord[1], host_coord[2], color='r', s=500,
                   depthshade=False, label="Punto")

    plt.show()

def guardar_achivo(filename, df, host_coord=None):
    """
    Guarda en un archivo csv con el nombre indicado los datos de las estrellas en el
    dataframe suministrado. Si se pasan las coordenadas de la estrella anfitriona, se
    agregan sus coordenadas como primera línea en el archivo.
    :param filename: nombre del archivo
    :param df: dataframe con las estrellas
    :param host_coord: coordenadas de la estrella anfitriona
    :return: None
    """
    df = df[["x_pc", "y_pc", "z_pc", "phot_g_mean_mag", "teff_gspphot"]]

    if host_coord is not None:
        data_centro = {
            'X': [host_coord[0]],
            'Y': [host_coord[1]],
            'Z': [host_coord[2]],
            'Luminosity': [0],
            'Temp': [0]
        }
        df_c = pd.DataFrame(data_centro)
        df = pd.concat([df_c, df])

    # Guardar el DataFrame en un archivo CSV
    df.to_csv(filename, index=False, header=False)
