/**
 * Sentinel-2 export for the LULC pipeline.
 * Paste into https://code.earthengine.google.com and run.
 *
 * Band order matters: the pipeline assumes [B2, B3, B4, B8, B11, B12]
 * (Blue, Green, Red, NIR, SWIR1, SWIR2). Change it here and you must change
 * BAND_NAMES / RGB_INDICES in src/lulc/constants.py to match.
 *
 * The default rectangle is Aligarh, Uttar Pradesh. Replace with your AOI.
 */

var city = ee.Geometry.Rectangle([78.0, 27.8, 78.2, 28.0]);

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(city)
  .filterDate('2024-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
  .median()                                   // median composite suppresses cloud
  .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']);

Map.centerObject(city, 11);
Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'S2 true colour');

Export.image.toDrive({
  image: s2,
  description: 'city_s2_6band',
  scale: 10,                                  // 10 m/px native for B2/B3/B4/B8
  region: city,
  fileFormat: 'GeoTIFF',
  maxPixels: 1e10
});
