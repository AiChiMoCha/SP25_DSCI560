from flask import Flask, jsonify, render_template_string
from sqlalchemy import create_engine, text

# Database connection configuration
db_url = "mysql+pymysql://DSCI560:560560@172.16.161.128/oil_wells_db"
engine = create_engine(db_url, echo=False)

app = Flask(__name__)

@app.route("/")
def index():
    # HTML template containing a Leaflet map and frontend script
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Oil Well Map</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!-- Include Leaflet styles and scripts -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet/dist/leaflet.js"></script>
        <style>
            #map { width: 100%; height: 600px; }
        </style>
    </head>
    <body>
        <h1>Oil Well Map</h1>
        <div id="map"></div>
        <script>
            // Initialize the map with a default center and zoom level
            var map = L.map('map').setView([0, 0], 2);
            
            // Add OpenStreetMap tile layer
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 18
            }).addTo(map);
            
            // Fetch oil well data from API
            fetch('/api/wells')
                .then(response => response.json())
                .then(data => {
                    console.log("Raw well data:", data);

                    data.forEach(function(well, index) {
                        console.log("Well", index, "Raw coordinates:", well.latitude, well.longitude);
                        if (!well.latitude || !well.longitude) {
                            return;
                        }

                        // Replace various dash characters with a standard hyphen
                        let latStr = well.latitude.toString().replace(/[−–—]/g, '-').trim();
                        let lonStr = well.longitude.toString().replace(/[−–—]/g, '-').trim();

                        let lat = parseFloat(latStr);
                        let lon = parseFloat(lonStr);
                        console.log("Well", index, "Parsed coordinates:", lat, lon);

                        if (!isNaN(lat) && !isNaN(lon)) {
                            let marker = L.marker([lat, lon]).addTo(map);
                            let popupContent = "<b>" + (well.well_name || "No Name") + "</b><br>" +
                                "API: " + (well.api || "N/A") + "<br>" +
                                "Address: " + (well.address || "No Address") + "<br>" +
                                "County: " + (well.county || "N/A") + "<br>" +
                                "Field: " + (well.field || "N/A") + "<br>" +
                                "Status: " + (well.well_status || "N/A") + "<br>" +
                                "Production Info: " + (well.production_info || "N/A");
                            marker.bindPopup(popupContent);
                        }
                    });
                });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route("/api/wells")
def api_wells():
    # Retrieve all records from the oil_wells table
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM oil_wells"))
        wells = result.mappings().all()
    
    # Convert result set to a list of dictionaries and return as JSON
    wells_list = [dict(well) for well in wells]
    return jsonify(wells_list)

if __name__ == "__main__":
    # Disable automatic reloader
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
