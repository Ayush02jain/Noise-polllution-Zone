const fs = require('fs');
const path = require('path');

const src = path.join(__dirname, 'data', 'geo', 'all_cities_locations_geo.json');
const dest = path.join(__dirname, 'frontend', 'all_cities_locations_geo.json');

fs.copyFileSync(src, dest);
console.log('Copied all_cities_locations_geo.json to frontend/');
