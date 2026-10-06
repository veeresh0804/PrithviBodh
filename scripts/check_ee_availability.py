"""M0-W2 S1/S2 availability check (Hyderabad, 2019/2025).
Needs `earthengine authenticate`. Without creds prints manual GEE Code Editor snippet.
Writes docs/availability_report.md
"""
from __future__ import annotations
from pathlib import Path
import yaml

AOI = [78.20, 17.11, 78.76, 17.65]
YEARS = [2019, 2025]

SNIPPET = """// GEE Code Editor: paste, run, record counts in docs/availability_report.md
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
});"""

def main():
    Path("docs").mkdir(exist_ok=True)
    try:
        import ee  # type: ignore
        ee.Initialize(project=None)
        print("EE authed — full automated check not yet wired; use snippet below for week-2 report.")
    except Exception as e:
        print(f"EE not authed ({e}). Manual snippet written.")
    rep = Path("docs/availability_report.md")
    if not rep.exists():
        rep.write_text(f"# S1/S2 Availability (Hyderabad {YEARS})\n\nAOI {AOI}\nDecision: primary 2019 (S1A+B) + 2025 (S1C). S1B failed 2021-12-23, S1C regular 2025-03-26.\n\n```js\n{SNIPPET}\n```\n\nRecord: S2 scenes/year, S1 ASC vs DESC counts/year, chosen orbit, monsoon clear-obs %.\n")
        print("wrote docs/availability_report.md")

if __name__ == "__main__":
    main()
