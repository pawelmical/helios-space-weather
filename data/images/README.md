# Coronagraph Images Directory

This directory is for storing coronagraph image files for CME detection analysis.

## Supported Formats

- FITS files (`.fits`, `.fts`) - Preferred for solar imagery
- PNG/JPEG - For preprocessed images

## Expected Structure

```
images/
├── SOHO_LASCO/
│   ├── 2000-07-14_C2/
│   │   ├── frame_0001.fits
│   │   ├── frame_0002.fits
│   │   └── ...
│   └── 2000-07-14_C3/
├── STEREO_A/
│   ├── COR1/
│   └── COR2/
└── STEREO_B/
    ├── COR1/
    └── COR2/
```

## Data Sources

1. **SOHO/LASCO**: 
   - https://lasco-www.nrl.navy.mil/
   - https://cdaw.gsfc.nasa.gov/

2. **STEREO/SECCHI**:
   - https://stereo.gsfc.nasa.gov/
   - https://stereo-ssc.nascom.nasa.gov/

## Notes

If actual coronagraph images are not available, the detection module can generate synthetic test images using `generate_synthetic_cme_images()` function.
