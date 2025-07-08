import requests
import xarray as xr
import csv
import zipfile
import io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import math
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from PIL import Image

# --- SYNOP dict ---
synop_stations = {
    "Szécsény (SYNOP: 12756)": (48.10667, 19.51556),
    "Jósvafő (SYNOP: 12766)": (48.49528, 20.53583),
    "Miskolc (SYNOP: 12772)": (48.09444, 20.72667),
    "Záhony (SYNOP: 12786)": (48.39833, 22.17722),
    "Sopron (SYNOP: 12805)": (47.67806, 16.60194),
    "Szombathely (SYNOP: 12812)": (47.19703, 16.64778),
    "Mosonmagyaróvár (SYNOP: 12815)": (47.88944, 17.26694),
    "Pér repülőtér (SYNOP: 12821)": (47.62417, 17.81167),
    "Győr (SYNOP: 12822)": (47.71000, 17.67444),
    "Pápa repülőtér (dél) (SYNOP: 12824)": (47.35667, 17.50333),
    "Veszprém / Szentkirályszabadja (SYNOP: 12830)": (47.08278, 17.97056),
    "Tata (SYNOP: 12836)": (47.65028, 18.30750),
    "Budapest / Lőrinc (SYNOP: 12843)": (47.42917, 19.18194),
    "Agárd (SYNOP: 12846)": (47.18972, 18.58333),
    "Tát (SYNOP: 12847)": (47.75639, 18.60583),
    "Kékestető (SYNOP: 12851)": (47.87194, 20.01278),
    "Szolnok (SYNOP: 12860)": (47.11694, 20.23083),
    "Poroszló (SYNOP: 12866)": (47.65583, 20.65333),
    "Eger (SYNOP: 12870)": (47.90389, 20.38889),
    "Debrecen (SYNOP: 12882)": (47.48417, 21.60556),
    #"Nyíregyháza/Napkor (SYNOP: 12892)": (47.96194, 21.88667),
    "Szentgotthárd / Farkasfa (SYNOP: 12910)": (46.91028, 16.30917),
    "Sármellék (SYNOP: 12922)": (46.69417, 17.15722),
    "Nagykanizsa (SYNOP: 12925)": (46.45583, 16.97056),
    "Siófok (SYNOP: 12935)": (46.91056, 18.04056),
    "Paks (SYNOP: 12950)": (46.57333, 18.84556),
    "Baja (SYNOP: 12960)": (46.17944, 19.01056),
    "Kecskemét (SYNOP: 12970)": (46.91194, 19.75944),
    "Szeged (SYNOP: 12982)": (46.25583, 20.09056),
    #"Békéscsaba (SYNOP: 12992)": (46.67944, 21.16056)
}

# ------------------------------------------------------------------------------------------------------------------------------------
# Global variables
inittime = datetime(2025, 3, 10, 0) # Ezt módosítsd a kívánt kezdőidőpontra!

forecast_hours_aifs = list(range(0, 361, 6))

# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
class ECMWF:  # parent class
    def __init__(self, name, inittime, forecast_hours, base_url):
        self.name = name
        self.inittime = inittime
        self.forecast_hours = forecast_hours
        self.base_url = base_url

        self.times = []
        self.mslps = []
        self.winds10 = []
        self.precipitations = []
        self.temperatures = []
        self.dewpoints = []

        self.dlon, self.dlat, self.lon0, self.lat0 = 0.25, -0.25, -180, 90 #grid paraméterei
# ------------------------------------------------------------------------------------------------------------------------------------
    def download_grib(self, filename, hour):
        file_url = f"{self.base_url}{self.inittime.strftime('%Y%m%d%H%M%S')}-{hour}h-oper-fc.grib2"
        save_dir = f'ecmwf_grib_data_{self.inittime.strftime("%Y%m%d")}/{self.name}-0p25'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {filename}, Status Code: {response.status_code}")
# ------------------------------------------------------------------------------------------------------------------------------------
    def download_all_grib_files(self):
        save_dir = f'ecmwf_grib_data_{self.inittime.strftime("%Y%m%d")}/{self.name}-0p25'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print(f"Downloading GRIB files for {self.name}...")
        for hour in self.forecast_hours:
            filename = f"{self.inittime.strftime('%Y%m%d%H%M%S')}-{hour}h-oper-fc.grib2"
            file_path = os.path.join(save_dir, filename)

            if not os.path.exists(file_path):
                self.download_grib(file_path, hour)
            else:
                print(f"File already exists: {filename}")
# ------------------------------------------------------------------------------------------------------------------------------------
    def extractData(self, filename, j, i):
        try:
            ds = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'meanSea'}, decode_timedelta=True)
            mslp = ds["msl"].values[j, i] / 100
        except Exception:
            mslp = None

        try:
            ds_wind10 = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround', 'level': 10}, decode_timedelta=True)
            u10, v10 = ds_wind10["u10"].values[j, i], ds_wind10["v10"].values[j, i]
            wind10 = np.sqrt(u10**2 + v10**2)
        except Exception:
            wind10 = None

        try:
            ds_precip = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={'stepType': 'accum', 'typeOfLevel': 'surface'}, decode_timedelta=True)
            if self.name == 'aifs-single':
                precipitation = ds_precip["tp"].values[j, i]
            elif self.name == 'ifs':
                precipitation = ds_precip["tp"].values[j, i] * 1000
        except Exception:
            precipitation = None

        try:
            ds_temp = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround', 'level': 2}, decode_timedelta=True)
            temp = ds_temp["t2m"].values[j, i] - 273.15
        except Exception:
            temp = None

        try:
            ds_dewpoint = xr.open_dataset(filename, engine="cfgrib", filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround', 'level': 2}, decode_timedelta=True)
            dewpoint = ds_dewpoint["d2m"].values[j, i] - 273.15
        except Exception:
            dewpoint = None

        return mslp, wind10, precipitation, temp, dewpoint
# ------------------------------------------------------------------------------------------------------------------------------------
    def extract_and_store_data_for_station(self, target_lat, target_lon):
        self.times = []
        self.mslps = []
        self.winds10 = []
        self.precipitations = []
        self.temperatures = []
        self.dewpoints = []

        i, j = int(np.round((target_lon - self.lon0) / self.dlon)), int(np.round((target_lat - self.lat0) / self.dlat))
        save_dir = f'ecmwf_grib_data_{self.inittime.strftime("%Y%m%d")}/{self.name}-0p25'

        last_precip_accum = None

        for hour in self.forecast_hours:
            valid_time = self.inittime + timedelta(hours=hour)
            filename = f"{self.inittime.strftime('%Y%m%d%H%M%S')}-{hour}h-oper-fc.grib2"
            file_path = os.path.join(save_dir, filename)
            if not os.path.exists(file_path):
                print(f"Error: GRIB file not found for extraction: {file_path}")
                mslp, wind10, precipitation, temp, dewpoint = None, None, None, None, None
            else:
                mslp, wind10, precipitation, temp, dewpoint = self.extractData(file_path, j, i)

            self.mslps.append(mslp)
            self.winds10.append(wind10)
            self.temperatures.append(temp)
            self.dewpoints.append(dewpoint)
            self.times.append(valid_time)

            if last_precip_accum is None:
                six_hourly_precip = 0.0
            else:
                six_hourly_precip = precipitation - last_precip_accum if precipitation is not None and last_precip_accum is not None else None

            self.precipitations.append(six_hourly_precip)
            last_precip_accum = precipitation
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
class AIFS(ECMWF):
    def __init__(self, inittime):
        super().__init__("aifs-single", inittime, forecast_hours_aifs,
                         f"https://data.ecmwf.int/forecasts/{inittime.strftime('%Y%m%d')}/{inittime.strftime('%Hz')}/aifs-single/0p25/oper/")

class IFS(ECMWF):
    def __init__(self, inittime):
        super().__init__("ifs", inittime, forecast_hours_aifs,
                         f"https://data.ecmwf.int/forecasts/{inittime.strftime('%Y%m%d')}/{inittime.strftime('%Hz')}/ifs/0p25/oper/")

# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
class MeasuredData:
    def __init__(self, target_lat, target_lon):
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.metadata = []
        self.station_code = None
        self.timestamps = []
        self.precipitation = []
        self.temperature = []
        self.pressure = []
        self.humidity = []
        self.wind_speed = []
        self.dewpoint = []
        self.elevation = []
        self.mslp = []

# ------------------------------------------------------------------------------------------------------------------------------------
    def download_metadata(self):
        metadata_url = 'https://odp.met.hu/climate/observations_hungary/hourly/station_meta_auto.csv'
        output_filename = "stations_meta_auto.csv"

        if os.path.exists(output_filename):
            print(f"A fájl már létezik: {output_filename}")
            return output_filename

        response = requests.get(metadata_url)
        if response.status_code == 200:
            with open(output_filename, "wb") as file:
                file.write(response.content)
            print(f"Fájl sikeresen letöltve: {output_filename}")
            return output_filename
        else:
            print(f"Hiba történt a letöltés során: {response.status_code}")
            return None

# ------------------------------------------------------------------------------------------------------------------------------------
    def load_metadata(self):
        filename = "stations_meta_auto.csv"
        self.metadata = [] # Tisztítás, ha esetleg már volt benne valami
        with open(filename, mode="r", encoding="utf-8") as file:
            csv_metadata = csv.reader(file, delimiter=";")
            next(csv_metadata)  #fejléc átugrása
            
            for row in csv_metadata:
                try:
                    lat = float(row[3])  #latitude a 4. oszlopban van
                    lon = float(row[4])  #longitude az 5.-ben
                    station_code = row[0].strip()  #station number az 1.-ben, whitespace eltávolítás
                    station_name = row[6].strip()  #állomásnév a 7.-ben, whitespace eltávolítás
                    self.metadata.append((lat, lon, station_code, station_name)) #eltárolja a metadata-t
                except (ValueError, IndexError) as e:
                    print(f"Hiba a metaadat sor feldolgozásakor: {row}. Hiba: {e}")
                    continue

# ------------------------------------------------------------------------------------------------------------------------------------    
    def find_nearest_station(self):
        if not self.metadata:
            print("Hiba: Nincsenek betöltött metaadatok a legközelebbi állomás kereséséhez.")
            return None, None

        #legkisebb négyzetes eltérés
        distances = [(np.square(lat - self.target_lat) + np.square(lon - self.target_lon), station_code, station_name)
                     for lat, lon, station_code, station_name in self.metadata]
                
        nearest_station = min(distances, key=lambda x: x[0])  #key=lambda x: x[0] rész: a distances-ben lévő tuple-ök első elemét adja vissza (ami a távolság)
        #tehát a nearest_station is egy tuple: nulladik eleme a távolság, első a station code, második pedig a station name
        nearest_station_code = nearest_station[1]
        nearest_station_name = nearest_station[2]
        
        return nearest_station_code, nearest_station_name

# ------------------------------------------------------------------------------------------------------------------------------------
    def download_and_extract_zip(self, omsz_station_code):
        # Itt a paraméter már az OMSZ állomás kódja!
        zip_url = f"https://odp.met.hu/climate/observations_hungary/hourly/recent/HABP_1H_{omsz_station_code}_akt.zip"
        
        print(f"Próbálom letölteni a ZIP-et erről az URL-ről: {zip_url}")
        
        response = requests.get(zip_url) #letölti a zip-et
        
        if response.status_code == 200: #ha OK, kicsomagolja
            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    csv_file = zip_ref.namelist()[0]  #a fájl neve
                    zip_ref.extract(csv_file, "station_data") #kicsomagolás
                    print(f"Fájl sikeresen letöltve és kicsomagolva: {omsz_station_code}")
                    print(f"Kicsomagolt fájl: {csv_file}")
                    return os.path.join("station_data", csv_file) #visszatérési érték: a csv fájl elérési útja
            except zipfile.BadZipFile:
                print(f"Hiba: A letöltött fájl nem érvényes ZIP fájl az {omsz_station_code} kódhoz.")
                return None
        else:
            print(f"Hiba történt a ZIP fájl letöltésekor az {omsz_station_code} kódhoz: {response.status_code}")
            return None

# ------------------------------------------------------------------------------------------------------------------------------------
# TETENS-FORMULA INVERZE ALAPJÁN        
    def calculate_dewpoint(self, T, RH):
        if T is None or RH is None or RH == 0:
            return None
        if RH <= 0: 
            return None
        gamma = (7.5 * T) / (T + 237.3) + math.log10(RH / 100)
        if (7.5 - gamma) == 0:
            return None
        Td = (237.3 * gamma) / (7.5 - gamma)
        return Td

    # ------------------------------------------------------------------------------------------------------------------------------------

    def process_csv(self, omsz_station_code):
        file_path = self.download_and_extract_zip(omsz_station_code)  # ZIP fájl letöltése és kibontása
        if not file_path:
            print(f'Nem sikerült letölteni vagy kibontani a CSV fájlt az {omsz_station_code} kódhoz!')
            return
        
        self.timestamps = []
        self.precipitation = []
        self.temperature = []
        self.pressure = []
        self.humidity = []
        self.wind_speed = []
        self.dewpoint = []
        self.mslp = []

        start_date = inittime  # Kezdő dátum (datetime objektum)
        # 0, 6, 12, ..., 360 óra
        forecast_hours_measure = list(range(0, 361, 6))
        valid_times = {start_date + timedelta(hours=h) for h in forecast_hours_measure}

        with open(file_path, mode="r", encoding="utf-8") as file:
            csv_file = csv.reader(file, delimiter=";")
            # Fejléc és az első adatblokk átlépése
            for _ in range(2):
                next(csv_file)
            row = next(csv_file)

            elevation = row[5]
            
            for _ in range(6):
                next(csv_file)

            hourly_data = []
            for row in csv_file:
                try:
                    timestamp_str = row[1]
                    # Ensure timestamp_str has enough characters for parsing
                    if len(timestamp_str) >= 12:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                    else:
                        continue

                    if not (start_date <= timestamp <= start_date + timedelta(hours=360)):
                        continue

                    hourly_data.append({
                        "timestamp": timestamp,
                        "precipitation": float(row[2]) if row[2] else 0.0,
                        "temp": float(row[4]) if row[4] else None,
                        "press": float(row[14]) if row[14] else None,
                        "hum": float(row[16]) if row[16] else None,
                        "ws": float(row[24]) if row[24] else None
                    })
                except ValueError as ve:
                    print(f"Data conversion error in row: {row}. Error: {ve}")
                except IndexError as ie:
                    print(f"Index error in row (missing column?): {row}. Error: {ie}")
                except Exception as e:
                    print(f"Unexpected error processing row: {row}. Error: {e}")

            # Process hourly data to get 6-hourly accumulated precipitation and other values at 6-hour intervals
            current_precip_6h_acc = 0.0
            last_timestamp = None
            
            for i, data in enumerate(hourly_data):
                timestamp = data["timestamp"]
                precipitation = data["precipitation"]

                if last_timestamp is None:
                    last_timestamp = timestamp
                
                if (timestamp.hour % 6 == 0 and (timestamp - start_date).total_seconds() / 3600 % 6 == 0) or \
                   (i == len(hourly_data) - 1 and (timestamp - start_date).total_seconds() / 3600 % 6 != 0): # For the last point if it's not exactly on a 6-hour mark
                    
                    if timestamp in valid_times:
                        self.timestamps.append(timestamp)
                        self.precipitation.append(current_precip_6h_acc + precipitation) # Add current hour's precip to accumulation
                        self.temperature.append(data["temp"])
                        self.pressure.append(data["press"])
                        self.humidity.append(data["hum"])
                        self.dewpoint.append(self.calculate_dewpoint(data["temp"], data["hum"]))
                        self.wind_speed.append(data["ws"])
                        current_precip_6h_acc = 0.0 # Reset for next 6-hour block
                    else:
                        current_precip_6h_acc += precipitation

                last_timestamp = timestamp
        
        # MSLP átszámítása
        M = 0.0289644
        L = 0.0065
        T0 = 288.15
        g = 9.80665
        R = 8.31432
        h = float(elevation)

        if elevation is None:
            self.mslp = [np.nan] * len(self.pressure)
        else:
            h = float(elevation)
            self.mslp = []  # inicializáljuk a listát
            for p, t in zip(self.pressure, self.temperature):
                if p is None or np.isnan(p) or p <= 0:
                    mslp = np.nan
                elif t is None or np.isnan(t):
                    mslp = np.nan
                else:
                    if (T0 - L * h) == 0:
                        mslp = np.nan
                    else:
                        t_kelvin = t + 273.15 
                        factor = 1 - (L * h / t_kelvin) 

                        if factor <= 0 or t_kelvin <= 0:
                            mslp = np.nan
                        else:
                            exponent = g * M / (R * L)
                            exponent_original = -1 * exponent
                            mslp = p * (factor ** exponent_original)
                self.mslp.append(mslp)
# ------------------------------------------------------------------------------------------------------------------------------------

def plotAll(city_name, nearest_station_name, aifs_data, ifs_data, measured_data):
    # Diagramok mentési mappájának beállítása
    output_dir = "SYNOP_meteograms"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    times_6hourly = aifs_data.times  # ECMWF időpontok (6 óránként)

    # Csapadék
    axs[0].plot(times_6hourly, measured_data.precipitation, label="Mért (6 óránként)", color='green',marker='o')
    axs[0].plot(times_6hourly, aifs_data.precipitations, label="AIFS", color='blue', marker='o')
    axs[0].plot(times_6hourly, ifs_data.precipitations, label="IFS", color='red', marker='o')
    line_5mm = [5] * len(times_6hourly) # 5mm vonal minden időponthoz
    axs[0].plot(times_6hourly, line_5mm, label='5 mm', color='black', linestyle='--')
    axs[0].set_ylabel("Csapadék (mm)")
    axs[0].set_ylim(bottom=0)
    axs[0].set_title(f"Csapadék adatok, {city_name}, mérőhely: {nearest_station_name}")
    axs[0].grid(True)        
    axs[0].legend()

    # Hőmérséklet
    axs[1].plot(times_6hourly, measured_data.temperature, label="Mért (6 óránként)", color='green', marker='o')
    axs[1].plot(times_6hourly, aifs_data.temperatures, label="AIFS", color='blue', marker='o')
    axs[1].plot(times_6hourly, ifs_data.temperatures, label="IFS", color='red', marker='o')
    axs[1].set_ylabel("Hőmérséklet (°C)")
    axs[1].set_title(f"Hőmérséklet adatok, {city_name}, mérőhely: {nearest_station_name}")
    axs[1].grid(True)
    axs[1].legend()

    # Légnyomás
    axs[2].plot(times_6hourly, measured_data.mslp, label="Mért (6 óránként)", color='green', marker='o')
    axs[2].plot(times_6hourly, aifs_data.mslps, label="AIFS", color='blue', marker='o')
    axs[2].plot(times_6hourly, ifs_data.mslps, label="IFS", color='red', marker='o')
    axs[2].set_ylabel("Légnyomás (hPa)")
    axs[2].set_title(f"Légnyomás adatok, {city_name}, mérőhely: {nearest_station_name}")
    axs[2].grid(True)
    axs[2].legend()

    # Szélsebesség
    axs[3].plot(times_6hourly, measured_data.wind_speed, label="Mért (6 óránként)", color='green', marker='o')
    axs[3].plot(times_6hourly, aifs_data.winds10, label="AIFS", color='blue', marker='o')
    axs[3].plot(times_6hourly, ifs_data.winds10, label="IFS", color='red', marker='o')
    axs[3].set_ylabel("Szélsebesség (m/s)")
    axs[3].set_title(f"Szélsebesség adatok, {city_name}, mérőhely: {nearest_station_name}")
    axs[3].grid(True)
    axs[3].legend()

    # Harmatpont
    axs[4].plot(times_6hourly, measured_data.dewpoint, label="Mért (6 óránként)", color='green', marker='o')
    axs[4].plot(times_6hourly, aifs_data.dewpoints, label="AIFS", color='blue', marker='o')
    axs[4].plot(times_6hourly, ifs_data.dewpoints, label="IFS", color='red', marker='o')
    axs[4].set_ylabel("Harmatpont (°C)")
    axs[4].set_title(f"Harmatpont adatok, {city_name}, mérőhely: {nearest_station_name}")
    axs[4].grid(True)
    axs[4].legend()

    # X tengely beállítások
    axs[4].set_xlabel("Idő")
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Képek mentése a megadott abszolút útvonalra
    plot_filename = os.path.join(output_dir, f"{city_name.replace(' ', '_').replace(':', '_')}_{inittime.strftime('%Y%m%d%H')}_forecasts.png")
    plt.savefig(plot_filename)
    print(f"Grafikon elmentve: {plot_filename}")
    plt.close(fig)

param_units = {
    'temperature': '°C',
    'precipitation': 'mm',
    'mslp': 'hPa',
    'wind_speed': 'm/s',
    'dewpoint': '°C'
}

param_titles_hu = {
    'temperature': 'hőmérséklet',
    'precipitation': 'csapadék',
    'mslp': 'MSLP',
    'wind_speed': 'szélsebesség',
    'dewpoint': 'harmatpont'
}

# --- ÚJ FÜGGVÉNY A TÉRKÉPES MEGJELENÍTÉSHEZ ---
def plotMeasuredDataOnMap(all_measured_data, plot_time_index, param_to_plot):
    """
    Térképen jeleníti meg a mért adatokat a Basemap segítségével.
    Ez a függvény most a Figure és Axes objektumokat adja vissza, és az M (Basemap) objektumot is.

    Args:
        all_measured_data (dict): Egy dictionary, ahol a kulcs a SYNOP állomás neve,
                                  értéke pedig a hozzá tartozó MeasuredData objektum.
        plot_time_index (int): Az időbélyeg indexe, amelynek adatait ábrázolni szeretnénk
                                (pl. 0 a kezdeti időpont, 1 a 6 órás előrejelzés stb.).
        param_to_plot (str): A megjelenítendő paraméter neve (pl. 'temperature', 'precipitation', 'mslp', 'wind_speed', 'dewpoint').
    Returns:
        tuple: (fig, ax, m, scatters, texts) Matplotlib Figure, Axes és Basemap objektumok,
               valamint a scatter plot és szöveg objektumok listája az animáció frissítéséhez.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    min_lon, min_lat = 15.5, 45.5
    max_lon, max_lat = 22.8, 48.8

    m = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat,
                urcrnrlon=max_lon, urcrnrlat=max_lat,
                resolution='i', projection='lcc',
                lat_0=(min_lat + max_lat) / 2, lon_0=(min_lon + max_lon) / 2,
                ax=ax)

    m.drawcoastlines()
    m.drawcountries()
    m.drawrivers(color='blue')
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='lightgreen', lake_color='aqua')

    # Rácsok és meridiánok
    m.drawparallels(np.arange(46., 49., 0.5), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(16., 23., 0.5), labels=[0, 0, 0, 1], fontsize=10)

    # Cím
    current_time = inittime + timedelta(hours=forecast_hours_aifs[plot_time_index])
    param_title = param_titles_hu.get(param_to_plot, param_to_plot) # Ha nincs meg a dictionaryben, akkor marad az angol
    ax.set_title(f"Mért {param_title} - {current_time.strftime('%Y-%m-%d %H:%M')} UTC")

    # Adatok előkészítése
    lons = []
    lats = []
    values = []
    station_display_names = [] # Ez fogja tartalmazni a térképen megjelenő állomásneveket

    for synop_station_name, coords in synop_stations.items(): # Iterálunk a SYNOP állomásokon
        lat, lon = coords
        
        # Az `all_measured_data` dictionary már a SYNOP állomásnevekkel van kulcsolva
        if synop_station_name in all_measured_data:
            measured_data_obj = all_measured_data[synop_station_name]
            if plot_time_index < len(getattr(measured_data_obj, param_to_plot)):
                value = getattr(measured_data_obj, param_to_plot)[plot_time_index]
                if value is not None:
                    lons.append(lon)
                    lats.append(lat)
                    values.append(value)
                    # Csak az állomás neve
                    station_display_names.append(synop_station_name.split(' ')[0]) 

    x, y = m(lons, lats)

    # Színskála beállítása a paraméter alapján
    cmap = 'viridis'
    norm = None
    if param_to_plot == 'temperature' or param_to_plot == 'dewpoint':
        cmap = 'coolwarm'
        vmin, vmax = -10, 30 # Példa tartomány
    elif param_to_plot == 'precipitation':
        cmap = 'Blues'
        vmin, vmax = 0, 20 # Példa tartomány
    elif param_to_plot == 'mslp':
        cmap = 'Spectral'
        #vmin, vmax = 990, 1030 # Példa tartomány
    elif param_to_plot == 'wind_speed':
        cmap = 'YlGnBu'
        #vmin, vmax = 0, 15 # Példa tartomány

    scatters = m.scatter(x, y, c=values, cmap=cmap, s=50, edgecolors='k', zorder=5, norm=norm) # `norm` hozzáadva
    
    # Szöveges címkék (állomás neve és érték)
    texts = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        x_text, y_text = m(lon + 0.05, lat + 0.05) # Kis eltolás, hogy ne fedje a pontot
        text = ax.text(x_text, y_text, f"{station_display_names[i]}\n{values[i]:.1f}{param_units[param_to_plot]}",
                       fontsize=8, ha='left', va='bottom', zorder=6)
        texts.append(text)

    # Színskála (colorbar)
    cbar = m.colorbar(scatters, location='right', pad="5%")
    cbar.set_label(f"{param_title} ({param_units[param_to_plot]})") 

    plt.tight_layout()
    return fig, ax, m, scatters, texts # Visszaadjuk az 'm' objektumot is

def update_map(frame, all_measured_data, param_to_plot, m, scatters, texts, ax):
    """
    Frissíti a térképet minden egyes animációs képkockához.
    """
    lons = []
    lats = []
    values = []
    station_display_names = []

    current_time = inittime + timedelta(hours=forecast_hours_aifs[frame])
    param_title = param_titles_hu.get(param_to_plot, param_to_plot) # Ha nincs meg a dictionaryben, akkor marad az angol
    ax.set_title(f"Mért {param_title} - {current_time.strftime('%Y-%m-%d %H:%M')} UTC")


    for synop_station_name, coords in synop_stations.items(): # Iterálunk a SYNOP állomásokon
        lat, lon = coords
        
        if synop_station_name in all_measured_data:
            measured_data_obj = all_measured_data[synop_station_name]
            if frame < len(getattr(measured_data_obj, param_to_plot)):
                value = getattr(measured_data_obj, param_to_plot)[frame]
                if value is not None:
                    lons.append(lon)
                    lats.append(lat)
                    values.append(value)
                    station_display_names.append(synop_station_name.split(' ')[0])

    x, y = m(lons, lats) # Itt használjuk a paraméterként kapott 'm' objektumot

    # Frissíti a scatter plot adatait
    scatters.set_offsets(np.c_[x, y])
    scatters.set_array(np.array(values))

    # Frissíti a szöveges címkéket vagy újra létrehozza őket
    for text in texts:
        text.remove() # Eltávolítja az előző képkocka szövegeit
    texts[:] = [] # Üríti a listát

    for i, (lon, lat) in enumerate(zip(lons, lats)):
        x_text, y_text = m(lon + 0.05, lat + 0.05)
        text = ax.text(x_text, y_text, f"{station_display_names[i]}\n{values[i]:.1f}{param_units[param_to_plot]}",
                       fontsize=8, ha='left', va='bottom', zorder=6)
        texts.append(text)
    
    return [scatters] + texts # Visszaadja az összes frissített művészi elemet


def create_map_animation(all_measured_data, param_to_plot, output_filename="map_animation.gif", fps=1):
    """
    Létrehoz egy animációt a térképes adatokból.

    Args:
        all_measured_data (dict): A mért adatokat tartalmazó dictionary.
        param_to_plot (str): A megjelenítendő paraméter (pl. 'temperature').
        output_filename (str): A kimeneti GIF fájl neve.
        fps (int): Képkockák másodpercenként (sebesség).
    """
    print(f"Creating animation for {param_to_plot}...")
    
    # Inicializálja az első képkockát, most már az 'm' objektumot is visszakapjuk
    fig, ax, m, scatters, texts = plotMeasuredDataOnMap(all_measured_data, 0, param_to_plot)

    # Animáció létrehozása
    # Az `fargs` paraméterrel adjuk át az `update_map` függvénynek szükséges további argumentumokat,
    # beleértve az 'm' Basemap objektumot is.
    anim = animation.FuncAnimation(fig, update_map, frames=len(forecast_hours_aifs),
                                   fargs=(all_measured_data, param_to_plot, m, scatters, texts, ax),
                                   interval=1000 // fps, blit=True, repeat=True)

    # Mentés PillowWriter-rel
    writer = animation.PillowWriter(fps=fps)
    animation_output_path = os.path.join(output_dir_base, output_filename)
    anim.save(animation_output_path, writer=writer)
    plt.close(fig) # Bezárja az utolsó ábrát
    print(f"Animation saved to {animation_output_path}")


# --- Fő futtatási logika ---
if __name__ == "__main__":
    # A fő kimeneti mappa beállítása
    output_dir_base = r"C:\Users\opago\Documents\Projektek\TDK\Z_animations_diagrams\anims\measured"
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    # 1. SYNOP metaadatok letöltése és betöltése
    # Ez az objektum csak arra kell, hogy egyszer letöltse és betöltse az összes állomás metaadatát.
    master_measured_data_obj = MeasuredData(0, 0) 
    master_measured_data_obj.download_metadata()
    master_measured_data_obj.load_metadata() # Betölti a metadata-t a master objektumba

    # 2. Adatok gyűjtése minden SYNOP állomásra
    all_measured_data_for_map = {}
    print("\n--- SYNOP adatok feldolgozása minden állomásra ---")
    for synop_station_name_in_dict, coords in synop_stations.items():
        lat, lon = coords
        
        # Létrehozunk egy MeasuredData objektumot az aktuális SYNOP állomás koordinátáival
        # Fontos: Minden egyes SYNOP állomáshoz külön MeasuredData objektumot hozunk létre
        # annak érdekében, hogy a tárolt adatok (precipitation, temperature stb.)
        # az adott SYNOP állomáshoz tartozzanak a dictionary-ben.
        current_synop_measured_data_obj = MeasuredData(lat, lon)
        
        # Átadjuk a master objektumból a már betöltött metaadatokat az új objektumnak,
        # hogy a find_nearest_station metódus tudja használni.
        current_synop_measured_data_obj.metadata = master_measured_data_obj.metadata
        
        # Megkeressük a legközelebbi OMSZ automata mérőállomást az adott SYNOP állomás koordinátáihoz
        omsz_station_code_for_zip, nearest_station_name_from_omsz = current_synop_measured_data_obj.find_nearest_station()
        
        if omsz_station_code_for_zip: # Ha találtunk érvényes OMSZ kódot
            print(f"\nFeldolgozandó SYNOP állomás: {synop_station_name_in_dict}")
            print(f"Legközelebbi OMSZ mérőhely: {nearest_station_name_from_omsz} (Kód: {omsz_station_code_for_zip})")
            
            # Ezzel az OMSZ kóddal töltjük le és dolgozzuk fel az adatokat
            current_synop_measured_data_obj.process_csv(omsz_station_code_for_zip)
            
            # Csak akkor tároljuk, ha a process_csv sikeres volt (pl. van benne időbélyeg)
            if current_synop_measured_data_obj.timestamps:
                all_measured_data_for_map[synop_station_name_in_dict] = current_synop_measured_data_obj
                print(f"Adatok sikeresen feldolgozva és tárolva ehhez: {synop_station_name_in_dict}")
            else:
                print(f"Nincsenek adatok a CSV-ben ehhez az OMSZ kódhoz: {omsz_station_code_for_zip}. Kihagyva.")
        else:
            print(f"Nem található megfelelő OMSZ mérőhely kód a {synop_station_name_in_dict} állomáshoz. Kihagyva.")


    # 3. Animációk generálása minden paraméterre
    print("\n--- Térképes animációk generálása ---")
    if not all_measured_data_for_map:
        print("Nincsenek feldolgozott mért adatok, animációk nem generálhatók.")
    else:
        for param_name in param_units.keys():
            create_map_animation(all_measured_data_for_map, param_name,
                                 output_filename=f"{param_name}_animation_{inittime.strftime('%Y%m%d%H')}.gif",
                                 fps=0.5) # 2 képkocka/másodperc

    print("\n--- Animációk kész! ---")
