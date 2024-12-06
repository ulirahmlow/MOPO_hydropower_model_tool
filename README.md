# **MOPO_hydro_tool**

## **Overview**

This tool processes geographical data and uses aggregated surface runoff and sub-surface runoff climate data to predict the hydropower according to Entso-e and eSett electricity data.

Equivalent model overview: to be added...

## **Requirements**

To be added...

## **Inflow Calculation**

### **Input Data**

**Geo Data**:
Download below database:

• [HydroBasins Europe](https://www.hydrosheds.org/products/hydrobasins)  
• [HydroRivers](https://www.hydrosheds.org/products/hydrorivers)  
• [JRC hydropower database](https://data.jrc.ec.europa.eu/dataset/52b00441-d3e0-44e0-8281-fda86a63546d)  
• [Processed era5 points shapefile](https://vttgroup.sharepoint.com/sites/EUESIMopo/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FEUESIMopo%2FShared%20Documents%2FGeneral%2FWP2%20Components%20CONFIDENTIAL%2FMopo%20WP2%20T3%2FToolVersion1%5Fextra%20data&viewid=6dc4b785%2D5340%2D4610%2Dbb2e%2D31ebc2ba5817)

**Other Data**:

• Apply Entsoe Transparency Platform API key to fetch data by contacting [Entsoe](https://transparency.entsoe.eu/)

• DTU CorRES webservice subscription by contacting [CorRES](https://corres.windenergy.dtu.dk/using-corres)

### **Setup**

1. **Update Configuration**:
   - Setup the correct folder path and API token in `.env.json` to include your downloaded geo data.
   - Setup the targeted zone and the method/level in `config.json`.

2. **Execute the Script**:
   - Run `01_main_generate_corres_points.py` to get the geographical points for fetching weather data from CorRES.

3. **Download Data**:
   - Download the surface runoff and subsurface runoff data from CorRES from 1991 to 2021 using the generated Excel file.

4. **Run Main Script**:
   - Execute `02_main_data.py` to get the predicted inflow or run-of-river generation data.

### **Solution**

Predicted inflow stored in `solutions/method/predicted_inflow`.

### **Method**

To be improved and updated…
