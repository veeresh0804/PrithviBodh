# S1/S2 Availability (Hyderabad [2019, 2025])

AOI [78.2, 17.11, 78.76, 17.65]
Decision: primary 2019 (S1A+B) + 2025 (S1C). S1B failed 2021-12-23, S1C regular 2025-03-26.

```js
// GEE Code Editor: paste, run, record counts in docs/availability_report.md
var aoi = ee.Geometry.Rectangle([78.20,17.11,78.76,17.65]);
[2019,2025].forEach(function(y){
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi).filter(ee.Filter.calendarRange(y,y,'year')).size().getInfo();
  print('S2 '+y, s2);
  ['ASCENDING','DESCENDING'].forEach(function(o){
    var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(aoi).filter(ee.Filter.calendarRange(y,y,'year'))
      .filter(ee.Filter.eq('instrumentMode','IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
      .filter(ee.Filter.eq('orbitProperties_pass',o)).size().getInfo();
    print('S1 '+y+' '+o, s1);
  });
});
```

Record: S2 scenes/year, S1 ASC vs DESC counts/year, chosen orbit, monsoon clear-obs %.
