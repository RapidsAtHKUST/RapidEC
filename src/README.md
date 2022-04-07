# RapidSV: Accelerating Elliptic Curve Signature Verification

## Configuration

Configurable flags in `gsv_wrapper.h`:

- `GSV_TPI`: Threads per instance, a parameter of the CGBN library. Can be set to 4, 8, 16, or 32. Default value is 16.
- `GSV_256BIT`: Enable this flag to use 256-bit integers for calculation rather than 512-bit integers. Will save space but a bit slower.
- `GSV_KNOWN_PKEY`: When the public key of a batch of signatures are the same, use a precomputed table to speed up verification. Need to generate different `sm2_pkey_512.table` for different keys.

## Acknowledgement
This project used the [CGBN](https://github.com/NVlabs/CGBN) library.
