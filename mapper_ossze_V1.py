from mapper_imports import *

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
    #"Nyíregyháza/Napkor (SYNOP: 12892)": (47.96194, 21.88667), #adathiány
    "Szentgotthárd / Farkasfa (SYNOP: 12910)": (46.91028, 16.30917),
    "Sármellék (SYNOP: 12922)": (46.69417, 17.15722),
    "Nagykanizsa (SYNOP: 12925)": (46.45583, 16.97056),
    "Siófok (SYNOP: 12935)": (46.91056, 18.04056),
    "Paks (SYNOP: 12950)": (46.57333, 18.84556),
    "Baja (SYNOP: 12960)": (46.17944, 19.01056),
    "Kecskemét (SYNOP: 12970)": (46.91194, 19.75944),
    "Szeged (SYNOP: 12982)": (46.25583, 20.09056),
    #"Békéscsaba (SYNOP: 12992)": (46.67944, 21.16056) #adathiány
}

# --- global variables ---
inittime = datetime(2025, 3, 10, 0)
forecast_hours = list(range(0, 361, 6))

# --- koordináták Magyarországra ---
min_lat, max_lat = 45.5, 48.8
min_lon, max_lon = 15.5, 22.8

# rács
grid_lats = np.arange(min_lat, max_lat + 0.001, 0.25) # Az .001 hozzáadás a max értékekhez a lebegőpontos pontatlanság elkerülése végett van
grid_lons = np.arange(min_lon, max_lon + 0.001, 0.25) # ez volt a basemaphez a rács

#pcolormesh-nek az edge-ek. nem a cellák középpontjait, hanem a cellák határait várja
lon_edges = np.linspace(min_lon - 0.25/2, max_lon + 0.25/2, len(grid_lons) + 1) #határvonalakból N+1 kell, ha N adatpont van
lat_edges = np.linspace(min_lat - 0.25/2, max_lat + 0.25/2, len(grid_lats) + 1)

#--------------------------------------------------------------------
# mértékegység dict
param_units = {
    'temperature': '°C',
    'precipitation': 'mm',
    'mslp': 'hPa',
    'wind_speed': 'm/s',
    'dewpoint': '°C'
}

# magyar nevek
param_titles_hu = {
    'temperature': 'hőmérséklet',
    'precipitation': 'csapadék',
    'mslp': 'MSLP',
    'wind_speed': 'szélsebesség',
    'dewpoint': 'harmatpont'
}

# színskála dict
cmap_dict = {
    'temperature': 'coolwarm',
    'mslp': 'Spectral',
    'wind_speed': 'YlGnBu',
    'dewpoint': 'coolwarm',
    'precipitation': 'Blues'
}

# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

class ECMWF:
    def __init__(self, name, inittime, forecast_hours, base_url):
        self.name = name
        self.inittime = inittime
        self.forecast_hours = forecast_hours
        self.base_url = base_url
        self.dlon, self.dlat, self.lon0, self.lat0 = 0.25, -0.25, -180, 90 
# ------------------------------------------------------------------------------------------------------------------------------------
    def downloadGrib(self, filename, hour):
        """Downloads an ECMWF GRIB forecast file.

            This method constructs the URL for a specific ECMWF GRIB2 forecast file
            based on the initialization time and forecast hour, then downloads it
            to a predefined local directory. It creates the directory if it doesn't
            already exist.

            Parameters
            ----------
            filename : str
                The full local path and name where the GRIB file should be saved
                (e.g., 'path/to/ecmwf_data/YYYYMMDDHHMMSS-Xh-oper-fc.grib2').
            hour : int
                The forecast hour (in hours from initialization time) for which
                the GRIB data is to be downloaded (e.g., 0, 6, 12, 360).

            Returns
            -------
            None
                The method downloads the file directly to the specified `filename` path.
                It prints an error message to the console if the download fails.

            Notes
            -----
            The `self.base_url`, `self.inittime`, and `self.name` attributes
            are expected to be set as part of the class instance.
            """

        file_url = f"{self.base_url}{self.inittime.strftime('%Y%m%d%H%M%S')}-{hour}h-oper-fc.grib2"
        save_dir = f'ecmwf_grib_data_{self.inittime.strftime("%Y%m%d")}/{self.name}-0p25'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        response = requests.get(file_url)

        if response.status_code == 200: #ha OK
            with open(filename, "wb") as f:
                f.write(response.content)

        else:
            print(f"Failed to download: {filename}, Status Code: {response.status_code}")

# ------------------------------------------------------------------------------------------------------------------------------------
    def extractData(self, filename, target_lats, target_lons):
        """Extracts selected meteorological parameters from an ECMWF GRIB2 file.

        Parameters extracted include 2-meter temperature (t2m), mean sea level pressure (msl),
        10-meter wind speed (from u10 and v10), total precipitation (tp), and 2-meter dewpoint (d2m).
        The method filters the data to a given lat-lon area and performs basic unit conversions
        (e.g., K to °C, Pa to hPa, m to mm for IFS precipitation).

        Parameters
        ----------
        filename : str
            Path to the GRIB2 file to read.
        target_lats : np.ndarray
            Array of target latitude values (used to define selection bounds).
        target_lons : np.ndarray
            Array of target longitude values.

        Returns
        -------
        tuple
            Tuple of extracted 2D arrays: (temperature, mslp, wind_speed, precipitation, dewpoint).
            Each can be None if not found in the GRIB file.

        Notes
        -----
        - Precipitation is converted from meters to millimeters if `self.name == 'ifs'`.
        - All fields are optional; if a variable is missing or fails to load, its value is None.
        """
        
        temp_data, mslp_data, wind10_data, precip_data, dew_data = [None]*5
        
        try:
            ds_temp = xr.open_dataset(filename, engine="cfgrib",
                                      filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround', 'level': 2},
                                      decode_timedelta=True)
            temp_data = ds_temp["t2m"].sel(latitude=slice(target_lats.max(), target_lats.min()), 
                                            longitude=slice(target_lons.min(), target_lons.max())).values - 273.15
                #sel: dimenzió label alapján subsetet választ ki 
                #megadjuk az egyes labeleket (pl latitude) és e mentén sliceoljuk a kívánt területre a datasetet (Mo.)
                #.values: numerikus adatot np arrayként nyeri ki
        except Exception:
            pass # temp_data marad None

        try:
            ds_mslp = xr.open_dataset(filename, engine="cfgrib",
                                      filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'meanSea'},
                                      decode_timedelta=True)
            mslp_data = ds_mslp["msl"].sel(latitude=slice(target_lats.max(), target_lats.min()), 
                                            longitude=slice(target_lons.min(), target_lons.max())).values / 100 # Pa -> hPa
        except Exception:
            pass

        try:
            ds_wind10 = xr.open_dataset(filename, engine="cfgrib",
                                         filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround', 'level': 10},
                                         decode_timedelta=True)
            u10_data = ds_wind10["u10"].sel(latitude=slice(target_lats.max(), target_lats.min()), 
                                             longitude=slice(target_lons.min(), target_lons.max())).values
            v10_data = ds_wind10["v10"].sel(latitude=slice(target_lats.max(), target_lats.min()), 
                                             longitude=slice(target_lons.min(), target_lons.max())).values
            wind10_data = np.sqrt(u10_data**2 + v10_data**2)
        except Exception:
            pass

        try:
            ds_precip = xr.open_dataset(filename, engine="cfgrib",
                                         filter_by_keys={'stepType': 'accum', 'typeOfLevel': 'surface'},
                                         decode_timedelta=True)
            tp_data = ds_precip["tp"].sel(latitude=slice(target_lats.max(), target_lats.min()), 
                                          longitude=slice(target_lons.min(), target_lons.max())).values
            if self.name == 'aifs-single':
                precip_data = tp_data
            elif self.name == 'ifs':
                precip_data = tp_data * 1000 # m -> mm :DD
        except Exception:
            pass

        try:
            ds_dew = xr.open_dataset(filename, engine="cfgrib",
                                      filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'heightAboveGround', 'level': 2},
                                      decode_timedelta=True)
            dew_data = ds_dew["d2m"].sel(latitude=slice(target_lats.max(), target_lats.min()), 
                                         longitude=slice(target_lons.min(), target_lons.max())).values - 273.15
        except Exception:
            pass

        return temp_data, mslp_data, wind10_data, precip_data, dew_data
    
# ------------------------------------------------------------------------------------------------------------------------------------
    def processForecast(self, target_lats, target_lons):
        """Processes ECMWF forecast data for selected latitude and longitude grids.

            This method loops through the specified forecast hours, downloading or accessing
            local GRIB2 files for each time step, and extracts relevant weather variables
            (mean sea level pressure, 10m wind speed, 2m temperature, 2m dewpoint, 6-hour precipitation).
            It stores the values into 3D NumPy arrays with dimensions [time, lat, lon].

            Parameters
            ----------
            target_lats : array-like
                Array of latitude values corresponding to the desired grid points.
            target_lons : array-like
                Array of longitude values corresponding to the desired grid points.

            Returns
            -------
            mslps : np.ndarray
                3D array of mean sea level pressure values [time, lat, lon].
            wind_speeds : np.ndarray
                3D array of 10-meter wind speed values [time, lat, lon].
            temperatures : np.ndarray
                3D array of 2-meter temperature values [time, lat, lon].
            dewpoints : np.ndarray
                3D array of 2-meter dewpoint temperature values [time, lat, lon].
            precipitations : np.ndarray
                3D array of 6-hourly accumulated precipitation values [time, lat, lon].
            times : list of datetime.datetime
                List of valid forecast times corresponding to each time step in the arrays.

            Notes
            -----
            - The method assumes that forecast data is available in GRIB2 format with filenames
            constructed from the initialization time and forecast hour.
            - Precipitation values are computed as differences between cumulative forecast steps,
            effectively yielding the precipitation amount for each 6-hour interval.
            - Care should be taken when comparing model parameters with observations, because observations
            are often local to a particular point in space and time, rather than representing averages over a model grid box.
            """

        num_lats = len(target_lats)
        num_lons = len(target_lons)
        
        mslps = np.full((len(self.forecast_hours), num_lats, num_lons), np.nan) # nan értékekkel feltöltött np array. np.full(shape, fill_value)
        wind_speeds= np.full((len(self.forecast_hours), num_lats, num_lons), np.nan)
        temperatures = np.full((len(self.forecast_hours), num_lats, num_lons), np.nan)
        dewpoints = np.full((len(self.forecast_hours), num_lats, num_lons), np.nan)
        precipitations = np.full((len(self.forecast_hours), num_lats, num_lons), np.nan)
        times = []

        save_dir = f"ecmwf_grib_data_{self.inittime.strftime("%Y%m%d")}/{self.name}-0p25"
        if not os.path.exists(save_dir): # ha nem létezik még
            os.makedirs(save_dir)

        last_precip_accum = np.full((num_lats, num_lons), np.nan) # last_precip_accum az előző 6 órás időpontig felhalmozott csapadék

        for t_idx, hour in enumerate(tqdm(self.forecast_hours, desc=f"Processing {self.name} forecast hours")): #desc: szöveges leírás tqdm-nek
            valid_time = self.inittime + timedelta(hours=hour) #ahol éppen jár a loop
            times.append(valid_time) #a times az adatokkal rendelkező időpontokat fogja tartalmazni

            filename = f"{self.inittime.strftime('%Y%m%d%H%M%S')}-{hour}h-oper-fc.grib2"
            file_path = os.path.join(save_dir, filename) #teljes elérési utat adja vissza

            if not os.path.exists(file_path): #létezik-e már a megadott úton ilyen grib?
                self.downloadGrib(file_path, hour) #ha nem, letölti a downloadGrib metódussal

            temp_grid, mslp_grid, wind10_grid, precip_grid, dew_grid = self.extractData(file_path, target_lats, target_lons) #meghívja az extractData-t
            # kinyeri a releváns időjárási paramétereket a letöltött GRIB fájlból a megadott szélességi és hosszúsági tartományra

            # Ellenőrizzük, hogy a kinyert adatok érvényesek-e (nem None) és illeszkednek-e a várt rácsmérethez
            if temp_grid is not None and temp_grid.shape == (num_lats, num_lons): #ha minden Ok...
                temperatures[t_idx] = temp_grid #az aktuális t-höz tartozó 2D-s rácsot (lat, lon) bemásoljuk a 3D-s adattömbbe (time, lat, lon)
            if mslp_grid is not None and mslp_grid.shape == (num_lats, num_lons):
                mslps[t_idx] = mslp_grid
            if wind10_grid is not None and wind10_grid.shape == (num_lats, num_lons):
                wind_speeds[t_idx] = wind10_grid
            if dew_grid is not None and dew_grid.shape == (num_lats, num_lons):
                dewpoints[t_idx] = dew_grid

            # csapadék külön...
            if precip_grid is not None and precip_grid.shape == (num_lats, num_lons): # precip_grid az aktuális 6 órás időpontig felhalmozott csapadék
                hourly_precip_grid = np.zeros_like(precip_grid) #egy ugyanolyan alakú, nulla értékkel feltöltött tömb, amibe a 6 óránkénti csapadékértékeket számoljuk
                mask_nan_precip = np.isnan(last_precip_accum) #logikai maszk: True, ahol last_precip_accum nan (azaz ez az első időpont, vagy az előző órában hiányzott az adat)
                hourly_precip_grid[mask_nan_precip] = 0.0 # azokon a rácspontokon, ahol ez az első csapadék adat (vagy az előző hiányzott), az óránkénti csapadékot 0.0-ra állítja
                valid_indices = ~mask_nan_precip # Létrehoz egy maszkot azokra a rácspontokra, ahol az last_precip_accum nem NaN (azaz van előző akkumulált adat)
            
                # Ha az hour pl. 12, akkor precip_grid a 0-12h közötti ÖSSZES csapadék.
                # last_precip_accum a 6h-nál a 0-6h között leesett ÖSSZES csapadék.
                # Eredmény: 6-12h között leesett csapadék
                hourly_precip_grid[valid_indices] = precip_grid[valid_indices] - last_precip_accum[valid_indices]

                precipitations[t_idx] = hourly_precip_grid
                last_precip_accum = precip_grid # Az aktuális akkumulált csapadék (precip_grid) lesz az új last_precip_accum a következő iterációhoz

        return mslps, wind_speeds, temperatures, dewpoints, precipitations, times
# ------------------------------------------------------------------------------------------------------------------------------------
class AIFS(ECMWF):
    def __init__(self, inittime):
        super().__init__("aifs-single", inittime, forecast_hours,
                         f"https://data.ecmwf.int/forecasts/{inittime.strftime('%Y%m%d')}/{inittime.strftime('%Hz')}/aifs-single/0p25/oper/")

class IFS(ECMWF):
    def __init__(self, inittime):
        super().__init__("ifs", inittime, forecast_hours,
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
        """Downloads metadata for Hungarian meteorological stations from OMSZ.

        This method downloads the station metadata CSV file from the official OMSZ
        data portal if it is not already present locally. The file contains automatic 
        station metadata such as station ID, name, coordinates, elevation, etc.

        Returns
        -------
        str or None
            The output filename if the download was successful or the file already exists.
            Returns None if the download fails.

        Notes
        -----
        - The method checks for local existence of the file before downloading.
        - The file is downloaded as 'stations_meta_auto.csv' from:
        https://odp.met.hu/climate/observations_hungary/hourly/station_meta_auto.csv
        """

        metadata_url = "https://odp.met.hu/climate/observations_hungary/hourly/station_meta_auto.csv"
        output_filename = "stations_meta_auto.csv"

        if os.path.exists(output_filename):
            print(f"File already exists: {output_filename}")
            return output_filename

        response = requests.get(metadata_url)
        if response.status_code == 200:
            with open(output_filename, "wb") as file:
                file.write(response.content)
            print(f"File downloaded successfully: {output_filename}")
            return output_filename
        else:
            print(f"An error occurred while downloading, status code: {response.status_code}")
            return None

# ------------------------------------------------------------------------------------------------------------------------------------
    def load_metadata(self):
        """ Loads station metadata from the local CSV file into memory.

        Reads 'stations_meta_auto.csv' (previously downloaded from OMSZ),
        parses each row, and extracts the station's latitude, longitude,
        station code, and name. The result is stored in `self.metadata` as a list of tuples.

        Each tuple has the form:
        (lat: float, lon: float, station_code: str, station_name: str)

        Returns
        -------
        None

        Notes
        -----
        - Assumes the CSV is encoded in UTF-8 and uses semicolon (;) as the delimiter.
        - Skips the header row.
        - Invalid or incomplete rows are skipped with a printed error message.
        """

        filename = "stations_meta_auto.csv"
        self.metadata = [] # Tisztítás, ha esetleg már volt benne valami

        with open(filename, mode="r", encoding="utf-8") as file:
            csv_metadata = csv.reader(file, delimiter=";")
            next(csv_metadata)  # fejléc átugrása
            
            for row in csv_metadata:
                try:
                    lat = float(row[3])  # latitude (szélesség) a 4. oszlopban van
                    lon = float(row[4])  # longitude (hosszúság) az 5.-ben
                    station_code = row[0].strip()  # station number az 1.-ben, whitespace eltávolítás; string.strip(characters)
                    station_name = row[6].strip()  # állomásnév a 7.-ben, whitespace eltávolítás
                    self.metadata.append((lat, lon, station_code, station_name)) #eltárolja a metadata-t
                except (ValueError, IndexError) as e:
                    print(f"Error processing metadata row {row}. Error: {e}")
                    continue

# ------------------------------------------------------------------------------------------------------------------------------------    
    def find_nearest_station(self):
        """ Finds the nearest meteorological station to the target coordinates.

        Uses the loaded station metadata to find the station with the smallest
        squared Euclidean distance to the target point (self.target_lat, self.target_lon).
        Returns the station code and station name of the closest station.

        Returns
        -------
        tuple[str, str] or (None, None)
            A tuple containing (station_code, station_name) of the nearest station.
            Returns (None, None) if metadata is not loaded.

        Notes
        -----
        - Requires that `self.metadata` is already populated via `load_metadata()`.
        - Distance is computed as (lat_diff)^2 + (lon_diff)^2 (no projection).
        - Assumes `self.target_lat` and `self.target_lon` are set on the instance.
        """

        if not self.metadata:
            print("Error: No metadata loaded to find the nearest station.")
            return None, None
        else: #legkisebb négyzetes eltéréssel keressük meg a legközelebbit
            distances = [(np.square(lat - self.target_lat) + np.square(lon - self.target_lon), station_code, station_name) #minden iterációkor ez a kifejezés kiértékelődik
                                for lat, lon, station_code, station_name in self.metadata] #iteráció minden állomáson
                    
            nearest_station = min(distances, key=lambda x: x[0])  #key=lambda x: x[0] rész: a distances-ben lévő tuple-ök első elemét adja vissza (ami a távolság)
            #tehát a nearest_station is egy tuple: nulladik eleme a távolság, első a station code, második pedig a station name
            nearest_station_code = nearest_station[1]
            nearest_station_name = nearest_station[2]
            
            return nearest_station_code, nearest_station_name

# ------------------------------------------------------------------------------------------------------------------------------------
    def download_and_extract_zip(self, omsz_station_code):
        """Downloads and extracts the latest hourly data ZIP file for a given OMSZ station.

        Given a station code, the method downloads a ZIP archive from the OMSZ server,
        which contains a single CSV file with recent hourly meteorological observations.
        The CSV file is extracted into the 'station_data' directory.

        Parameters
        ----------
        omsz_station_code : str
            The unique identifier code of the OMSZ meteorological station.

        Returns
        -------
        str or None
            The file path to the extracted CSV file if successful, or None if the download
            or extraction fails.

        Raises
        ------
        zipfile.BadZipFile
            If the downloaded file is not a valid ZIP archive.

        Notes
        -----
        - The ZIP file is expected to contain exactly one CSV file.
        - Uses in-memory BytesIO to avoid saving the ZIP file to disk before extraction.
        - Creates or uses the 'station_data' directory for extraction.
        """

        zip_url = f"https://odp.met.hu/climate/observations_hungary/hourly/recent/HABP_1H_{omsz_station_code}_akt.zip"
        
        response = requests.get(zip_url)
        if response.status_code == 200:
            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref: #BytesIO is like a virtual file that exists in the computer's memory. It lets you perform operations on these bytes, such as reading and writing, as if you were interacting with a regular file
                    csv_file = zip_ref.namelist()[0]  # zip fájl egyetlen CSV fájlt tartalmaz csak!
                    zip_ref.extract(csv_file, "station_data") # kicsomagolja a megtalált CSV fájlt a megadott "station_data" mappába
                    return os.path.join("station_data", csv_file) # visszatérési érték: a csv fájl elérési útja
            
            except zipfile.BadZipFile: # nem érvényes zip fájl (pl. sérült)
                print(f"Error: The downloaded file is not a valid ZIP file for {omsz_station_code}.")
                return None
       
        else:
            print(f"Error downloading ZIP file for {omsz_station_code}, status code: {response.status_code}")
            return None

# ------------------------------------------------------------------------------------------------------------------------------------
# Tetens-formula inverze
    def calculate_dewpoint(self, T, RH):
        """Calculates the dew point temperature (Td) from air temperature and relative humidity.

        Uses the inverse of Tetens' formula to estimate dew point in degrees Celsius,
        given temperature and relative humidity.

        Parameters
        ----------
        T : float
            Air temperature in degrees Celsius.
        RH : float
            Relative humidity in percent (0 to 100).

        Returns
        -------
        float or None
            Dew point temperature in degrees Celsius, or None if input is invalid or calculation fails.

        Notes
        -----
        - Returns None if RH is 0 (dew point undefined) or out of bounds.
        - Handles potential math errors such as division by zero or invalid logarithms.
        - Reference: https://glossary.ametsoc.org/wiki/Tetens%27s_formula
        """

        if T is None or RH is None:
            print("Error: temperature or relative humidity is None.")
            return None
        if not (0 <= RH <= 100):
            print("Error: relative humidity must be between 0 and 100.")
            return None
        if RH == 0: #log-nál hiba lenne
            return None
        try:
            es_T = 0.611 * (10**((17.3 * T) / (T + 237.3))) 
        except ZeroDivisionError:
            return None
        
        ea = (RH / 100.0) * es_T #aktuális gőznyomás

        try:
            gamma = math.log10(ea / 0.611)
        except ValueError:
            return None
        denominator = 17.3 - gamma
        if denominator == 0:
            return None
        Td = (237.3 * gamma) / denominator
        return Td

    # ------------------------------------------------------------------------------------------------------------------------------------

    def process_csv(self, omsz_station_code):
        """Processes hourly meteorological CSV data for a given OMSZ station code.

        Downloads and extracts the hourly observation ZIP file for the specified
        station, then reads and processes the CSV data. Extracts timestamped hourly
        values of precipitation, temperature, pressure, humidity, wind speed, and
        calculates dew point where possible. Data are aggregated into 6-hour intervals
        matching the forecast hours.

        Additionally, converts station pressure measurements to mean sea level pressure
        (MSLP) using the non-isothermal barometric formula with station elevation.

        Attributes populated on `self`:
        - timestamps: list of datetime objects for each forecast hour
        - precipitation: list of accumulated precipitation sums (6-hour totals)
        - temperature: list of temperatures at forecast times
        - pressure: list of station pressure values at forecast times
        - humidity: list of relative humidity percentages at forecast times
        - wind_speed: list of wind speeds at forecast times
        - dewpoint: list of calculated dew point temperatures at forecast times
        - mslp: list of mean sea level pressure values calculated from station pressure and elevation

        Parameters
        ----------
        omsz_station_code : str
            OMSZ station identifier used to download the appropriate data file.

        Returns
        -------
        None
            The processed data are stored as instance attributes; no direct return value.

        Notes
        -----
        - Expects `inittime` and `forecast_hours` to be defined in the global scope or
        accessible in the method's context.
        - Skips rows with invalid or missing data.
        - Assumes CSV delimiter is ';' and encoding is UTF-8.
        - The 6-hour precipitation is computed as the sum over the preceding 6 hours.
        - Dew point is calculated only when temperature and humidity data are valid.
        - Mean sea level pressure (MSLP) is calculated using the non-isothermal barometric formula: https://en.wikipedia.org/wiki/Barometric_formula
        - If elevation data is missing, MSLP values are set to NaN.
        """
        file_path = self.download_and_extract_zip(omsz_station_code)  # ZIP fájl letöltése és kibontása, visszatérési érték: elérési út
        
        if not file_path:
            return None
        
        self.timestamps = []
        self.precipitation = []
        self.temperature = []
        self.pressure = []
        self.humidity = []
        self.wind_speed = []
        self.dewpoint = []
        self.mslp = []

        start_date = inittime
        valid_times = {start_date + timedelta(hours=h) for h in forecast_hours} #inittime = datetime(2025, 3, 10, 0), forecast_hours = list(range(0, 361, 6))

        with open(file_path, mode="r", encoding="utf-8") as file:
            csv_file = csv.reader(file, delimiter=";")
            for _ in range(2): #0. sor: ##Meta; 1. sor: #StationNumber;StartDate;EndDate ;Latitude;Longitude;Elevation;StationName;EOR
                next(csv_file)
            row = next(csv_file)
            try:
                elevation = float(row[5]) # a 2. sor 5. oszlopa a tszfm
            except (ValueError, IndexError):
                elevation = None

            for _ in range(3): # 3. sor: ##Meta END; 4. sor: üres, 5. sor: fejléc
                next(csv_file)

            hourly_data = []
            for row in csv_file:
                try:
                    timestamp_str = row[1] # a sor 2. oszlopában van a dátum
                    if len(timestamp_str) >= 12: #12 karakter egy időpont, pl: 202503101200
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M") #megfelelő formátumra alakítás
                    else:
                        continue

                    # Csak a releváns időablakot dolgozzuk fel
                    if not (start_date <= timestamp <= start_date + timedelta(hours=forecast_hours[-1])):
                        continue

                    hourly_data.append({
                        "timestamp": timestamp,
                        "precipitation": float(row[2]) if row[2] else np.nan, # órás csapadékösszeg: 2. oszlop (python indexeléssel számolva)
                        "temp": float(row[4]) if row[4] else np.nan, # órás pillanatnyi hőmérséklet: 4. oszlop
                        "press": float(row[14]) if row[14] else np.nan, # órás pillanatnyi műszerszinti légnyomás: 14. oszlop
                        "hum": float(row[16]) if row[16] else np.nan, # órás pillanatnyi relatív nedvesség: 16. oszlop
                        "ws": float(row[24]) if row[24] else np.nan # órás szinoptikus szélsebesség: 24. oszlop
                    })
                except ValueError as ve:
                    continue
                except IndexError as ie:
                    continue
                except Exception as e:
                    continue
          
            hourly_data.sort(key=lambda x: x["timestamp"]) # Az órás adatok rendezése időrendben (ha esetleg nem lenne az)

            # A 6 órás léptékekre aggregáljuk az adatokat
            for target_hour in forecast_hours:
                target_time = start_date + timedelta(hours=target_hour)
                
                target_omsz_data = None
                for d in hourly_data: # egyező timestamp megkeresése
                    if d["timestamp"] == target_time:
                        target_omsz_data = d
                        break # Megtaláltuk, kilépünk

                # Gyűjtsük össze az aktuális 6 órás intervallum adatait
                interval_data = [d for d in hourly_data if target_time - timedelta(hours=6) < d["timestamp"] <= target_time]

                if target_omsz_data is not None and interval_data:
                    accum_precip = np.nansum([d["precipitation"] for d in interval_data]) # akkumulálva a kívánt intervallumra (6 órás összeg), nan-t kihagyva
                    self.precipitation.append(accum_precip)
                    self.timestamps.append(target_time)
                    self.temperature.append(target_omsz_data["temp"])
                    self.pressure.append(target_omsz_data["press"] if "press" in target_omsz_data else np.nan)
                    self.humidity.append(target_omsz_data["hum"] if "hum" in target_omsz_data else np.nan)
                    self.wind_speed.append(target_omsz_data["ws"] if "ws" in target_omsz_data else np.nan)

                    if not np.isnan(self.temperature[-1]) and not np.isnan(self.humidity[-1]): # ha van adat, harmatpont számítás
                        self.dewpoint.append(self.calculate_dewpoint(self.temperature[-1], self.humidity[-1]))
                    else:
                        self.dewpoint.append(np.nan)
                else:
                    self.timestamps.append(target_time)
                    self.precipitation.append(np.nan)
                    self.temperature.append(np.nan)
                    self.pressure.append(np.nan)
                    self.humidity.append(np.nan)
                    self.wind_speed.append(np.nan)
                    self.dewpoint.append(np.nan)

        # MSLP átszámítása: https://en.wikipedia.org/wiki/Barometric_formula
        M = 0.0289644
        L = 0.0065
        T0 = 288.15
        g = 9.80665
        R = 8.31432
        
        if elevation is None:
            self.mslp = [np.nan] * len(self.pressure)
        else:
            h = float(elevation)
            for p, t in zip(self.pressure, self.temperature):
                if p is not None and not np.isnan(p) and t is not None and not np.isnan(t):
                    if (T0 - L * h) == 0:
                        mslp = np.nan
                    else:
                        t_kelvin = t + 273.15 
                        factor = 1 - (L * h / t_kelvin) 

                        if factor <= 0 or (t + 273.15) <= 0:
                            mslp = np.nan
                        else:
                            exponent = g * M / (R * L)
                            mslp = p / ((1 - L * h / T0) ** exponent)
                            exponent_original = -1*exponent
                            mslp = p * (factor ** exponent_original) 
                else:
                    mslp = np.nan
                self.mslp.append(mslp)

# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

def plotCombinedDataOnMap(model_data, all_measured_data, plot_time_index, param_to_plot, model_name="AIFS"):
    """Plot gridded model forecasts and point-based observations on a map.

    Displays a selected meteorological parameter at a given forecast lead time using
    model data (as a color grid) and station measurements (as labeled points).

    Parameters
    ----------
    model_data : object
        Contains 3D arrays (time, lat, lon) for each parameter (e.g., temperatures).
    all_measured_data : dict
        Maps station names to objects with time-indexed parameter arrays.
    plot_time_index : int
        Index of the forecast hour to visualize (e.g., 0 = initial, 1 = +6h).
    param_to_plot : str
        Parameter name to plot (e.g., "temperature", "precipitation").
    model_name : str, optional
        Name of the model to display in the title (default is "AIFS").

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    m : mpl_toolkits.basemap.Basemap
    pcolormesh_obj : matplotlib.collections.QuadMesh
    scatters : matplotlib.collections.PathCollection
    texts : list of matplotlib.text.Text

    Notes
    -----
    - Station values below -900 or missing (NaN/None) are skipped.
    - Relies on global variables like `grid_lats`, `grid_lons`, `param_units`, etc.
    - Model arrays must be 3D; otherwise, an error is raised.
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))

    m = Basemap(llcrnrlon=min_lon, llcrnrlat=min_lat,
                urcrnrlon=max_lon, urcrnrlat=max_lat,
                resolution='i', projection='lcc', # Lambert's conformal conic vetület (Lambert-Gauss-féle szögtartó kúpvetület)
                lat_0=(min_lat + max_lat) / 2, lon_0=(min_lon + max_lon) / 2,
                ax=ax)

    m.drawcoastlines()
    m.drawcountries()
    m.drawrivers(color='blue')
    #m.drawmapboundary(fill_color='aqua')
    m.drawmapboundary()
    m.fillcontinents(color='white', lake_color='aqua')
    m.drawparallels(np.arange(46., 49., 0.5), labels=[1, 0, 0, 0], fontsize=10) #szélességi körök megrajzolása
    m.drawmeridians(np.arange(16., 23., 0.5), labels=[0, 0, 0, 1], fontsize=10) #hosszúsági körök megrajzolása

    # cím beállításai
    current_time = inittime + timedelta(hours=forecast_hours[plot_time_index])
    param_title_display = param_titles_hu.get(param_to_plot, param_to_plot)
    ax.set_title(f"Mért és {model_name} {param_title_display} - {current_time.strftime('%Y-%m-%d %H:%M')} UTC")

    # összes modell adatok
    model_param_data_all_times = getattr(model_data, f"{param_to_plot}s") # modell attribútumok lekérése
    
    if model_param_data_all_times.ndim != 3: # ellenőrzi, hogy 3 dimenziós-e az adat
        raise ValueError(f"{param_to_plot} model parameters are not 3 dimensional (time, lat, lon).")

    model_data_at_time = model_param_data_all_times[plot_time_index] # kiválasztja az adott időhöz tartozó 2D adatot
    
    masked_model_data = ma.masked_invalid(model_data_at_time) #érvénytelen adatok maszkolása

    # a cellahatárok létrehozása pcolormeshnek (pcolormesh cellák sarkaira - nem közepére - vár koordinátákat)
    lon_edges_for_plot = np.linspace(min_lon, max_lon + 0.25, len(grid_lons) + 1)
    lat_edges_for_plot = np.linspace(min_lat, max_lat + 0.25, len(grid_lats) + 1)

    lon_mesh_for_plot, lat_mesh_for_plot = np.meshgrid(lon_edges_for_plot, lat_edges_for_plot) # meshgriddé alakítás
    x_map, y_map = m(lon_mesh_for_plot, lat_mesh_for_plot) #basemap vetületre váltás

    cmap = cmap_dict.get(param_to_plot, 'viridis') #alapértelmezettet állít be, ha nem szerepel színskála
    vmin = np.nanmin(model_param_data_all_times) # a színskála min/max értéke, hogy egységes legyen az összes térképen
    vmax = np.nanmax(model_param_data_all_times)

    # pcolormesh létrehozása
    pcolormesh_obj = m.pcolormesh(x_map, y_map, masked_model_data, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.7)
    
    # Colorbar a modell adatokhoz
    cb = plt.colorbar(pcolormesh_obj, ax=ax, label=f"{param_title_display} ({param_units[param_to_plot]})")

    # --- MÉRT PONTOK ADATAI ---
    measured_lons = []
    measured_lats = []
    measured_values = []
    station_display_names = []

    for synop_station_name, coords in synop_stations.items(): # végigiterál a synop dicten
        lat, lon = coords
        if synop_station_name in all_measured_data:
            measured_data_obj = all_measured_data[synop_station_name]
            # Ellenőrizzük, hogy az index valid-e a mért adatokra is
            if plot_time_index < len(getattr(measured_data_obj, param_to_plot)): #Van-e egyáltalán annyi adat, hogy a plot_time_index-edik elemhez hozzá tudjunk férni?
                value = getattr(measured_data_obj, param_to_plot)[plot_time_index]
                if value is not None and not np.isnan(value) and value > -900: # Szűrjük: None, nan, vagy hiányzó adat (-999)
                    measured_lons.append(lon)
                    measured_lats.append(lat)
                    measured_values.append(value)
                    station_display_names.append(synop_station_name.split(' ')[0]) #állomás nevének első része

    x_measured, y_measured = m(measured_lons, measured_lats)

    # Scatter plot a mért adatokhoz (színek megegyeznek a rács színskálájával, ha lehetséges)
    scatters = m.scatter(x_measured, y_measured, c=measured_values, cmap=cmap, s=70, edgecolors='black', linewidth=1.5, zorder=10, vmin=vmin, vmax=vmax)
    
    # Szöveges címkék (állomás neve és érték)
    texts = []
    for i, (lon, lat) in enumerate(zip(measured_lons, measured_lats)):
        x_text, y_text = m(lon + 0.05, lat + 0.05) # Kis eltolás, hogy ne fedje a pontot
        text = ax.text(x_text, y_text, f"{station_display_names[i]}\n{measured_values[i]:.1f}{param_units[param_to_plot]}",
                        fontsize=7, ha='left', va='bottom', zorder=11, color='black', weight='bold')
        texts.append(text)

    plt.tight_layout()
    return fig, ax, m, pcolormesh_obj, scatters, texts

#---------------------------------- ITT TARTOK -----------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
def update_combined_map(frame, model_data, all_measured_data, param_to_plot, model_name, m, pcolormesh_obj, scatters, texts, ax):
    """ Updates the combined map for each frame of the animation.

    This function is intended to be used as the update function in a matplotlib animation.
    It updates both the model grid data (raster) and measured station data (points + labels)
    at a given forecast time, corresponding to the animation frame index.

    Parameters
    ----------
    frame : int
        Index of the current frame (forecast hour index).
    model_data : object
        Object containing model forecast data arrays as attributes (e.g., model_data.temperatures).
    all_measured_data : dict
        Dictionary of measured data objects keyed by station name.
        Each object is expected to have attributes like temperature, pressure, etc., which are lists or arrays.
    param_to_plot : str
        Parameter name to visualize (e.g., "temperature", "precipitation", etc.).
    model_name : str
        Name of the model (e.g., "IFS", "AIFS") for use in the plot title.
    m : Basemap
        Basemap object used for coordinate transformation and plotting.
    pcolormesh_obj : QuadMesh
        The pcolormesh object for the model data (used to display grid data).
    scatters : PathCollection
        The scatter plot object representing measured station values.
    texts : list
        List of matplotlib Text objects for station labels. These are removed and redrawn each frame.
    ax : matplotlib.axes.Axes
        Axes object for updating the plot title and text labels.

    Returns
    -------
    list
        List of updated matplotlib artist objects for use with FuncAnimation.
    """
    # --- cím beállítások ---
    current_time = inittime + timedelta(hours=forecast_hours[frame])
    param_title_display = param_titles_hu.get(param_to_plot, param_to_plot.capitalize())
    ax.set_title(f"Mért és {model_name} {param_title_display} - {current_time.strftime('%Y-%m-%d %H:%M')} UTC")

    # --- modell rács adatok frissítése ---
    model_param_data_all_times = getattr(model_data, f"{param_to_plot}s")
    model_data_at_time = model_param_data_all_times[frame] # megfelelő időpillanathoz tartozó 2D-s rács
    masked_model_data = ma.masked_invalid(model_data_at_time) #nan maszkolás
    pcolormesh_obj.set_array(masked_model_data.ravel()) # lecseréli az eddigi térképadatokat az újra

    # --- mért pontok adatainak frissítése ---
    measured_lons = []
    measured_lats = []
    measured_values = []
    station_display_names = []

    for synop_station_name, coords in synop_stations.items(): # iteráció a synop állomásokon. teljesen hasonló a plotCombinedDataOnMap metódushoz
        lat, lon = coords
        if synop_station_name in all_measured_data:
            measured_data_obj = all_measured_data[synop_station_name]
            if frame < len(getattr(measured_data_obj, param_to_plot)): # nehogy kilógjon az index
                value = getattr(measured_data_obj, param_to_plot)[frame] # lekérdezi az adott időpontra az értéket a mért adat objektumból
                if value is not None and not np.isnan(value):
                    measured_lons.append(lon)
                    measured_lats.append(lat)
                    measured_values.append(value)
                    station_display_names.append(synop_station_name.split(' ')[0])

    x_measured, y_measured = m(measured_lons, measured_lats)

    scatters.set_offsets(np.c_[x_measured, y_measured]) # frissíti a körök helyét
    scatters.set_array(np.array(measured_values)) # és színét

    for text in texts:
        text.remove() # eltávolítja az előző képkocka szövegeit
    texts[:] = [] # üríti a listát

    for i, (lon, lat) in enumerate(zip(measured_lons, measured_lats)): # iteráció a mérési pontokon
        x_text, y_text = m(lon + 0.05, lat + 0.05) # kis eltolás
        text = ax.text(x_text, y_text, f"{station_display_names[i]}\n{measured_values[i]:.1f}{param_units[param_to_plot]}",
                        fontsize=7, ha='left', va='bottom', zorder=11, color='black', weight='bold') # újra legenerálja
        texts.append(text)
    
    return [pcolormesh_obj, scatters] + texts # ez kell a FuncAnimation-nek, hogy tudja, mit frissítsen

# ------------------------------------------------------------------------------------------------------------------------------------
def create_combined_map_animation(model_data, all_measured_data, param_to_plot, model_name="AIFS", output_filename="combined_animation.gif", fps=0.5):
    """
    Létrehoz egy animációt a kombinált térképes adatokból (modell rács + mért pontok).

    Args:
        model_data (ECMWF osztály objektum): A modell adatok, pl. ifs_data vagy aifs_data.
        all_measured_data (dict): A mért adatokat tartalmazó dictionary.
        param_to_plot (str): A megjelenítendő paraméter (pl. 'temperature').
        model_name (str): A megjelenített modell neve (pl. "IFS").
        output_filename (str): A kimeneti GIF fájl neve.
        fps (int): Képkockák másodpercenként (sebesség).
    """
    print(f"Creating combined animation for {param_to_plot} ({model_name})...")
    
    # Inicializálja az első képkockát
    fig, ax, m, pcolormesh_obj, scatters, texts = plotCombinedDataOnMap(model_data, all_measured_data, 0, param_to_plot, model_name)

    # Animáció létrehozása
    anim = animation.FuncAnimation(fig, update_combined_map, frames=len(forecast_hours),
                                   fargs=(model_data, all_measured_data, param_to_plot, model_name, m, pcolormesh_obj, scatters, texts, ax),
                                   interval=1000 // fps, blit=False, repeat=True) # blit=False ajánlott, ha a szövegobjektumokat újra kell rajzolni

    writer = animation.PillowWriter(fps=fps)
    animation_output_path = os.path.join(output_dir_base, output_filename)
    anim.save(animation_output_path, writer=writer)
    plt.close(fig)
    print(f"Combined animation saved to {animation_output_path}")


# --- Fő futtatási logika ---
if __name__ == "__main__":
    # A fő kimeneti mappa beállítása
    output_dir_base = r"C:\Users\opago\Documents\Projektek\TDK\Z_animations_diagrams\anims\combined" # Új mappa a kombinált animációknak
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    # 1. SYNOP metaadatok letöltése és betöltése
    master_measured_data_obj = MeasuredData(0, 0) 
    master_measured_data_obj.download_metadata()
    master_measured_data_obj.load_metadata()

    # 2. Modell adatok előkészítése (IFS-t használjuk az animációhoz)
    # A processForecast most a Basemap régiójának rácspontjait kapja meg.
    print("\n--- Modell adatok feldolgozása (AIFS) ---")
    aifs = AIFS(inittime)
    # Fontos: a Basemap által lefedett régió kiterjedését használjuk a processForecast-hoz
    # Mivel a Basemap min_lon, max_lon, min_lat, max_lat-ból indul ki
    # és a grid_lats/grid_lons a 0.25°-os lépésekkel van definiálva,
    # ezeket kell átadni a processForecast-nak.
    aifs_mslps, aifs_wind_speeds, aifs_temperatures, aifs_dewpoints, aifs_precipitations, aifs_times = \
        aifs.processForecast(grid_lats, grid_lons)
    
    # Tároljuk az összes modellt egy dictionary-ben, a könnyebb hozzáférésért
    # Fontos: Ezeket a processForecast hívásokat futtatni kell, mielőtt az animációkat hívjuk!
    model_data_store = {
        'aifs': {
            'mslps': aifs_mslps,
            'wind_speeds': aifs_wind_speeds,
            'temperatures': aifs_temperatures,
            'dewpoints': aifs_dewpoints,
            'precipitations': aifs_precipitations,
            'times': aifs_times
        },
        # 'aifs': { ... hasonlóan az aifs adatokhoz, ha használnánk }
    }
    # Ideiglenesen átalakítjuk az IFS objektumot, hogy az adatok közvetlenül elérhetőek legyenek.
    # Ez nem ideális objektumorientáltan, de most az egyszerűség kedvéért.
    # Később érdemes lehet egy külön osztályt létrehozni a feldolgozott modell adatok tárolására.
    aifs.mslps = aifs_mslps
    aifs.wind_speeds = aifs_wind_speeds
    aifs.temperatures = aifs_temperatures
    aifs.dewpoints = aifs_dewpoints
    aifs.precipitations = aifs_precipitations
    aifs.times = aifs_times


    # 3. Mért adatok gyűjtése minden SYNOP állomásra
    all_measured_data_for_map = {}
    print("\n--- SYNOP adatok feldolgozása minden állomásra ---")
    for synop_station_name_in_dict, coords in tqdm(synop_stations.items(), desc="Processing SYNOP stations"):
        lat, lon = coords
        
        current_synop_measured_data_obj = MeasuredData(lat, lon)
        current_synop_measured_data_obj.metadata = master_measured_data_obj.metadata
        
        omsz_station_code_for_zip, nearest_station_name_from_omsz = current_synop_measured_data_obj.find_nearest_station()
        
        if omsz_station_code_for_zip:
            current_synop_measured_data_obj.process_csv(omsz_station_code_for_zip)
            
            if current_synop_measured_data_obj.timestamps:
                all_measured_data_for_map[synop_station_name_in_dict] = current_synop_measured_data_obj

    # 4. Kombinált animációk generálása minden paraméterre
    print("\n--- Kombinált térképes animációk generálása ---")
    if not all_measured_data_for_map:
        print("Nincsenek feldolgozott mért adatok, animációk nem generálhatók.")
    else:
        for param_name in param_units.keys():
            create_combined_map_animation(aifs, all_measured_data_for_map, param_name, model_name="AIFS",
                                            output_filename=f"combined_AIFS_{param_name}_animation_{inittime.strftime('%Y%m%d%H')}.gif",
                                            fps=0.5)

    print("\n--- Animációk kész! ---")