# RapidSV: Accelerating Elliptic Curve Signature Verification

## Build

```
make volta  # For V100 GPU
make test   # Run test on sample data
make debug  # Enable debug logging
```

## Usage

```
./bin/gsv [GPU device ID=0]
```

## Performance

Signature verification speed (verification per second) of the SM2 curve on a V100 GPU.

| \# Instances | TPI=4  | TPI=8  | TPI=16 | TPI=32 |
|-----------|--------|--------|--------|--------|
| 256       | 24797  | 29211  | 46961  | 16247  |
| 512       | 50143  | 58245  | 93806  | 31834  |
| 1024      | 100425 | 116121 | 175414 | 54013  |
| 2048      | 199952 | 210174 | 187385 | 61769  |
| 4096      | 339504 | 228033 | 243576 | 60714  |
| 8192      | 209716 | 157628 | 221000 | 52265  |
| 16384     | 227121 | 171301 | 203729 | 45149  |
| 32768     | 184437 | 150094 | 178685 | 40747  |
| 65536     | 159113 | 132883 | 157507 | 38251  |
| 131072    | 135282 | 120625 | 147475 | 37165  |
| 262144    | 129478 | 114446 | 142155 | 36535  |

## Acknowledgement
This project used the [CGBN](https://github.com/NVlabs/CGBN) library.
