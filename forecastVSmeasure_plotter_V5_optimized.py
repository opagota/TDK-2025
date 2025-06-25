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

# --- SYNOP dict ---
synop_stations = {
    # "Szécsény (SYNOP: 12756)": (48.10667, 19.51556),
    # "Jósvafő (SYNOP: 12766)": (48.49528, 20.53583),
    # "Miskolc (SYNOP: 12772)": (48.09444, 20.72667),
    # "Záhony (SYNOP: 12786)": (48.39833, 22.17722),
    # "Sopron (SYNOP: 12805)": (47.67806, 16.60194),
    # "Szombathely (SYNOP: 12812)": (47.19703, 16.64778),
    # "Mosonmagyaróvár (SYNOP: 12815)": (47.88944, 17.26694),
    # "Pér repülőtér (SYNOP: 12821)": (47.62417, 17.81167),
    # "Győr (SYNOP: 12822)": (47.71000, 17.67444),
    # "Pápa repülőtér (dél) (SYNOP: 12824)": (47.35667, 17.50333),
    # "Veszprém - Szentkirályszabadja (SYNOP: 12830)": (47.08278, 17.97056),
    # "Tata (SYNOP: 12836)": (47.65028, 18.30750),
    # "Budapest - Lőrinc (SYNOP: 12843)": (47.42917, 19.18194),
    # "Agárd (SYNOP: 12846)": (47.18972, 18.58333),
    # "Tát (SYNOP: 12847)": (47.75639, 18.60583),
    # "Kékestető (SYNOP: 12851)": (47.87194, 20.01278),
    # "Szolnok (SYNOP: 12860)": (47.11694, 20.23083),
    # "Poroszló (SYNOP: 12866)": (47.65583, 20.65333),
    # "Eger (SYNOP: 12870)": (47.90389, 20.38889),
    # "Debrecen (SYNOP: 12882)": (47.48417, 21.60556),
    # #"Nyíregyháza - Napkor (SYNOP: 12892)": (47.96194, 21.88667), #adathiany!
    # "Szentgotthárd - Farkasfa (SYNOP: 12910)": (46.91028, 16.30917),
    "Sármellék (SYNOP: 12922)": (46.69417, 17.15722),
    "Nagykanizsa (SYNOP: 12925)": (46.45583, 16.97056),
    "Siófok (SYNOP: 12935)": (46.91056, 18.04056),
    "Paks (SYNOP: 12950)": (46.57333, 18.84556),
    "Baja (SYNOP: 12960)": (46.17944, 19.01056),
    "Kecskemét (SYNOP: 12970)": (46.91194, 19.75944),
    "Szeged (SYNOP: 12982)": (46.25583, 20.09056),
    #"Békéscsaba (SYNOP: 12992)": (46.67944, 21.16056) #adathiany!
}

# ------------------------------------------------------------------------------------------------------------------------------------
# Global variables
inittime = datetime(2025, 3, 10, 0)

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
        """Letölti az összes szükséges GRIB fájlt a megadott inittime-hoz és forecast_hours-hoz."""
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
    def extractData(self, filename, j, i):  # metódus az adatok feldolgozására
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
    def extract_and_store_data_for_station(self, target_lat, target_lon):  # metódus az adatok feldolgozásához
        self.times = []
        self.mslps = []
        self.winds10 = []
        self.precipitations = []
        self.temperatures = []
        self.dewpoints = []

        i, j = int(np.round((target_lon - self.lon0) / self.dlon)), int(np.round((target_lat - self.lat0) / self.dlat))
        save_dir = f'ecmwf_grib_data_{self.inittime.strftime("%Y%m%d")}/{self.name}-0p25'

        last_precip_accum = None

        for hour in self.forecast_hours:  # az összes előrejelzési órát feldolgozzuk
            valid_time = self.inittime + timedelta(hours=hour)
            filename = f"{self.inittime.strftime('%Y%m%d%H%M%S')}-{hour}h-oper-fc.grib2"
            file_path = os.path.join(save_dir, filename)

            # Itt már feltételezzük, hogy a fájl létezik a download_all_grib_files() hívás miatt
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
        with open(filename, mode="r", encoding="utf-8") as file:
            csv_metadata = csv.reader(file, delimiter=";")
            next(csv_metadata)  #fejléc átugrása
            
            for row in csv_metadata:
                lat = float(row[3])  #latitude a 4. oszlopban van
                lon = float(row[4])  #longitude az 5.-ben
                station_code = row[0]  #station number az 1.-ben
                station_name = row[6]  #állomásnév a 7.-ben
                self.metadata.append((lat, lon, station_code, station_name)) #eltárolja a metadata-t

# ------------------------------------------------------------------------------------------------------------------------------------    
    def find_nearest_station(self): #metódus a legközelebbi állomás megkeresésére a target koordináták alapján
        #legkisebb négyzetes eltérés
        distances = [(np.square(lat - self.target_lat) + np.square(lon - self.target_lon), station_code, station_name)
                     for lat, lon, station_code, station_name in self.metadata]
                
        nearest_station = min(distances, key=lambda x: x[0])  #key=lambda x: x[0] rész: a distances-ben lévő tuple-ök első elemét adja vissza (ami a távolság)
        #tehát a nearest_station is egy tuple: nulladik eleme a távolság, első a station code, második pedig a station name
        nearest_station_code = nearest_station[1]
        nearest_station_code = nearest_station_code.strip()  #Szóközök eltávolítása
        nearest_station_name = nearest_station[2]
        
        return nearest_station_code, nearest_station_name

# ------------------------------------------------------------------------------------------------------------------------------------
    def download_and_extract_zip(self, station_code): #metódus a zip feldolgozására

        zip_url = f"https://odp.met.hu/climate/observations_hungary/hourly/recent/HABP_1H_{station_code}_akt.zip"
        response = requests.get(zip_url) #letölti a zip-et
        
        if response.status_code == 200: #ha OK, kicsomagolja
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                csv_file = zip_ref.namelist()[0]  #a fájl neve
                zip_ref.extract(csv_file, "station_data") #kicsomagolás
                print(f"Fájl sikeresen letöltve és kicsomagolva")
                print(f"Kicsomagolt fájl: {csv_file}")
                return os.path.join("station_data", csv_file) #visszatérési érték: a csv fájl elérési útja
        else:
            print(f"Hiba történt a ZIP fájl letöltésekor: {response.status_code}")
            return None

# ------------------------------------------------------------------------------------------------------------------------------------
# TETENS-FORMULA INVERZE ALAPJÁN        
    def calculate_dewpoint(self, T, RH):
        if T is None or RH is None or RH == 0:
            return None
        if RH <= 0: #nem pozitív szám log-a
            return None
        gamma = (7.5 * T) / (T + 237.3) + math.log10(RH / 100)
        if (7.5 - gamma) == 0: #0-val osztás
            return None
        Td = (237.3 * gamma) / (7.5 - gamma)
        return Td

    # ------------------------------------------------------------------------------------------------------------------------------------

    def process_csv(self, station_code):
        file_path = self.download_and_extract_zip(station_code)  # ZIP fájl letöltése és kibontása
        if not file_path:
            print('Nem sikerült letölteni vagy kibontani a CSV fájlt!')
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
            print('Az állomás tengerszint feletti magassága:', elevation, 'm')
            for _ in range(6):
                next(csv_file)

            # Store all hourly precipitation values first
            hourly_data = []
            for row in csv_file:
                try:
                    timestamp_str = row[1]
                    # Ensure timestamp_str has enough characters for parsing
                    if len(timestamp_str) >= 12:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                    else:
                        print(f"Skipping row due to invalid timestamp format: {row}")
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
                
                # Check if it's time for a 6-hourly accumulation point
                # This logic ensures we accumulate correctly over 6 hours and store at the 6-hour marks
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
                        current_precip_6h_acc += precipitation # Continue accumulating if not a valid_time for storing
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

        for p, t in zip(self.pressure, self.temperature):
            if p is not None and t is not None:
                if (T0 - L * h) == 0: #zero division
                    mslp = None
                else:
                    factor = 1 - (L * h / T0)
                    if factor <= 0: #log
                        mslp = None
                    else:
                        exponent = -g * M / (R * L)
                        mslp = p * (factor ** exponent)
                self.mslp.append(mslp)
            else:
                self.mslp.append(None)
# ------------------------------------------------------------------------------------------------------------------------------------

def plotAll(city_name, nearest_station_name, aifs_data, ifs_data, measured_data):
    # Diagramok mentési mappájának beállítása
    output_dir = r"C:/Users/opago/Documents/Projektek/TDK/Z_animations_diagrams"
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

# 1. Létrehozzuk az AIFS és IFS objektumokat egyszer
aifs = AIFS(inittime)
ifs = IFS(inittime)

# 2. Letöltjük az ÖSSZES GRIB fájlt mindkét modellhez, csak egyszer.
# Ez jelentősen csökkenti a futási időt, mivel a hálózati letöltések csak egyszer történnek meg.
print("\n--- ECMWF GRIB fájlok előzetes letöltése és ellenőrzése ---")
aifs.download_all_grib_files()
ifs.download_all_grib_files()
print("--- ECMWF GRIB fájlok letöltése/ellenőrzése befejezve ---")

# 3. Fő loop futtatása az összes SYNOP állomásra
for city_name, coords in synop_stations.items():
    target_lat, target_lon = coords
    print(f"\n--- Feldolgozás indítása: {city_name} (Lat: {target_lat}, Lon: {target_lon}) ---")

    # MeasuredData példányosítás
    measured_data = MeasuredData(target_lat, target_lon)

    # Metadata letöltése és betöltése (ez is csak egyszer történik meg a fájl letöltésével)
    measured_data.download_metadata() 
    measured_data.load_metadata()

    # Legközelebbi állomás keresése
    nearest_station_code, nearest_station_name = measured_data.find_nearest_station()
    print(f"A legközelebbi mérőállomás ehhez a helyhez: {nearest_station_name} (Kód: {nearest_station_code})")
    measured_data.process_csv(nearest_station_code) # Ez letölti és feldolgozza a mért adatokat

    # Az AIFS és IFS adatok kinyerése és tárolása az aktuális állomásra
    # Itt már nem töltjük le a fájlokat, csak feldolgozzuk a meglévőket a diszkről
    print(f"Kinyerem az AIFS adatokat a(z) {city_name} állomáshoz...")
    aifs.extract_and_store_data_for_station(target_lat, target_lon)
    print(f"Kinyerem az IFS adatokat a(z) {city_name} állomáshoz...")
    ifs.extract_and_store_data_for_station(target_lat, target_lon) # Hibás hívás javítva, lásd alább!

    # Grafikonok rajzolása és mentése
    plotAll(city_name, nearest_station_name, aifs, ifs, measured_data)

print("\n--- Minden állomás feldolgozása befejeződött! ---")