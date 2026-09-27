import ee
ee.Initialize(project='stable-matter-508305-h2')

point = ee.Geometry.Point([-59.8, -6.2])  # any test coordinate
image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(point).first()
print(image.getInfo())