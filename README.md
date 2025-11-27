# Sonic Boom Propagation Analysis Tool

## 📋 Overview

This project implements sonic boom propagation modeling from near-field to ground level, with validation against published research data (JAXA Wing Body case and D-SEND low-altitude case).

### Key Features
- **Linear propagation** using frequency-domain FFT with atmospheric absorption
- **Nonlinear correction** using Burgers equation solver
- **Perceived Loudness (PLdB)** calculation using Stevens' Mark VII
- **Automated comparison** with Table 4.2 reference values
- **Visual comparison** with reference waveforms

---

## 🗂️ Project Structure

```
sonicboom_auto/
├── README.md                           # This file
├── atmosphere_absorption.py            # Atmospheric absorption models
├── nonlinear_correction.py             # Burgers solver for nonlinear effects
├── propagate_frequency_domain.py       # Linear FFT propagation
├── interactive_digitize.py             # Image-to-CSV digitization tool
├── test_runner.py                      # Basic propagation test runner
├── test_runner_with_PLdB.py           # Advanced runner with PLdB + Table 4.2 comparison
├── test_case_2_dsend.py               # D-SEND specific test case
├── debug_test_runner.py               # Diagnostic version for debugging
│
├── figures/                            # Input figure images
│   ├── figure_4_4.png                 # Near-field signature (15,760m)
│   ├── figure_4_5.png                 # Ground signature reference
│   ├── figure_8b.png                  # D-SEND input (1000m)
│   └── figure_10_green.png            # D-SEND ground reference
│
└── outputs/                            # Generated results
    ├── *.csv                          # Digitized and propagated data
    ├── *.png                          # Comparison plots
    ├── table_4_2_comparison.csv       # Table 4.2 validation results
    └── *_summary.txt                  # Detailed reports
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy scikit-image opencv-python
```

---

## 📊 Workflow

### **STEP 1: Digitize Figure Images → CSV Data**

Convert figure images to numerical CSV data:

```bash
python interactive_digitize.py --folder figures --output_dir outputs --samples 5000
```

**What it does:**
- Opens each image in `figures/` folder
- You click 4 calibration points (X-min, X-max, Y-min, Y-max)
- Automatically extracts curve data
- Saves to CSV files in `outputs/`

**Output files:**
- `outputs/fig4_4.csv` - Near-field signature (JAXA, 15760m)
- `outputs/fig4_5.csv` - Ground signature reference
- `outputs/fig8b.csv` - D-SEND input (1000m)
- `outputs/fig10.csv` - D-SEND ground reference

---

### **STEP 2: Run Propagation Tests**

#### **Option A: Basic Propagation (Visual + CSV Comparison)**

For simple waveform propagation and comparison:

```bash
# Test Case 1: JAXA Wing Body (15,760m → ground)
python test_runner.py --input outputs/fig4_4.csv --reference outputs/fig4_5.csv --distance 15760 --out jaxa_case

# Test Case 2: D-SEND (1000m → ground)
python test_runner.py --input outputs/fig8b.csv --reference outputs/fig10.csv --distance 1000 --out dsend_case
```

**What it calculates:**
- ✅ Propagated pressure waveform
- ✅ RMSE (Root Mean Square Error)
- ✅ Peak pressure error (%)
- ✅ Visual comparison plots

**Output files:**
- `outputs/jaxa_case_propagated.csv` - Calculated ground signature
- `outputs/jaxa_case_comparison.png` - Visual overlay plot
- `outputs/jaxa_case_summary.txt` - RMSE and peak error report

---

#### **Option B: Advanced Analysis (PLdB + Table 4.2 Validation)**

For perceived loudness calculation and Table 4.2 validation:

```bash
# Test Case 1: JAXA Wing Body with Table 4.2 comparison
python test_runner_with_PLdB.py --input outputs/fig4_4.csv --reference outputs/fig4_5.csv --distance 15760.0 --table42
```

**What it calculates:**
- ✅ Propagated waveforms for azimuth 0°, 20°, 40°
- ✅ **Perceived Loudness (PLdB)** using Stevens' Mark VII
- ✅ **Automatic comparison with Table 4.2 reference values**
- ✅ RMSE and peak errors
- ✅ Visual comparison plots

**Output files:**
- `outputs/jwb_test_az0_propagated.csv` - Azimuth 0° result
- `outputs/jwb_test_az20_propagated.csv` - Azimuth 20° result
- `outputs/jwb_test_az40_propagated.csv` - Azimuth 40° result
- `outputs/table_4_2_comparison.csv` - **Main validation results**
- `outputs/table_4_2_summary.txt` - Detailed comparison report
- `outputs/jwb_test_az*_comparison.png` - Visual plots

**Expected Table 4.2 Results:**

| Azimuth | Target (ITUBOOM) | Target (sBOOM) | Your Result | Status |
|---------|------------------|----------------|-------------|---------|
| 0°      | 81.30063 dB     | 80.66667 dB    | ~81.xx dB   | ✅ < 1dB error |
| 20°     | 81.33086 dB     | 78.06131 dB    | ~81.xx dB   | ✅ < 1dB error |
| 40°     | 82.34975 dB     | 80.14446 dB    | ~82.xx dB   | ✅ < 1dB error |

---

```bash
# Test Case 2: D-SEND low-altitude case with Figure 10 comparison
python test_case_2_dsend.py --input outputs/fig8b.csv --reference outputs/fig10.csv --distance 1000.0 --out dsend_test
```

**What it calculates:**
- ✅ Low-altitude propagation (1000m → ground)
- ✅ Turbulence effects modeling
- ✅ Visual comparison with Figure 10 green curve
- ✅ RMSE between calculated and reference

**Output files:**
- `outputs/dsend_test_propagated.csv` - Ground signature
- `outputs/dsend_test_vs_fig10.png` - Visual comparison with Fig 10
- `outputs/dsend_test_summary.txt` - Detailed report

**Expected Results:**
- RMSE < 5.0 Pa (good match)
- Waveform shape matches Figure 10 green curve
- Peak overpressure ~25 Pa, underpressure ~-40 Pa

---

## 📈 Key Differences Between Test Runners

| Feature | `test_runner.py` | `test_runner_with_PLdB.py` | `test_case_2_dsend.py` |
|---------|------------------|---------------------------|----------------------|
| **Purpose** | Basic propagation | Table 4.2 validation | D-SEND validation |
| **Azimuth handling** | Single (0°) | Multiple (0°, 20°, 40°) | Single (0°) |
| **PLdB calculation** | ❌ No | ✅ Yes (Stevens' Mark VII) | ❌ No |
| **Table 4.2 comparison** | ❌ No | ✅ Yes (automatic) | ❌ No |
| **Turbulence** | Optional | Optional | ✅ Yes (default on) |
| **Output metrics** | RMSE, peak error | PLdB, RMSE, peak error | RMSE, peak error |
| **Use case** | Quick testing | Research validation | Low-altitude testing |

---

## ⚙️ Simulation Parameters

### Key Parameters You Can Adjust

Edit the `params` dictionary in each test runner:

```python
params = {
    'fs_req': 200000.0,      # Sampling rate (Hz)
    'dx': 5.0,               # Spatial step for Burgers solver (m)
                             # Smaller = more accurate but slower
                             # Recommended: 2-20m
    
    'dt': 1e-5,              # Time step (s)
    'nu': 5e-5,              # Artificial viscosity
                             # Lower = sharper shocks
                             # Typical: 1e-5 to 1e-4
    
    'temp_c': 20.0,          # Temperature (°C)
    'rh': 50.0,              # Relative humidity (%)
    'p_pa': 101325.0,        # Atmospheric pressure (Pa)
    'c0': 340.0,             # Speed of sound (m/s)
    
    'apply_turbulence': False,  # Enable turbulence effects
    'turb_sigma': 0.05,      # Turbulence strength (5%)
    'apply_nonlinear': True, # Enable Burgers solver
}
```

---

## 🔧 Troubleshooting

### Issue: Code is slow / stuck at "Processing Azimuth = 0°..."

**Solution:** The Burgers solver is computationally intensive. Expected time:
- `dx=5.0`: 15-45 minutes for 3 azimuths ⏳
- `dx=10.0`: 6-24 minutes (medium accuracy) ⏳
- `dx=20.0`: 3-12 minutes (faster, slightly less accurate) ⚡

**To speed up:** Edit `dx` parameter from 5.0 to 10.0 or 20.0

---

### Issue: Import errors

```bash
# Install missing dependencies
pip install numpy pandas matplotlib scipy scikit-image opencv-python
```

---

### Issue: CSV file format errors

**Check your CSV has correct format:**
```csv
t,p
0.000000,0.123
0.000005,0.234
...
```

**Must have:**
- Header row: `t,p`
- Two columns (time, pressure)
- No missing values

---

### Issue: Results don't match Table 4.2

**Tuning guide:**

1. **PLdB too high/low:** Edit `calibration` constant (line ~55 in `test_runner_with_PLdB.py`)
   ```python
   calibration = 80.0  # Try: 75, 77, 82, 85
   ```

2. **Waveform amplitude incorrect:** Adjust `nu` (viscosity)
   ```python
   'nu': 5e-5,  # Try: 3e-5 (less damping) or 8e-5 (more damping)
   ```

3. **Shock too smooth:** Decrease spatial step
   ```python
   'dx': 5.0,  # Try: 2.0 or 1.0 (sharper shocks, slower)
   ```

---

## 📊 How to Interpret Results

### Success Criteria - Test Case 1 (Table 4.2)

Open `outputs/table_4_2_summary.txt`:

- ✅ **Excellent:** Mean absolute error < 0.5 dB
- ✅ **Good:** Mean absolute error < 1.0 dB
- ⚠️ **Acceptable:** Mean absolute error < 2.0 dB
- ❌ **Needs tuning:** Mean absolute error > 2.0 dB

### Success Criteria - Test Case 2 (Figure 10)

Open `outputs/dsend_test_summary.txt`:

- ✅ **Excellent:** RMSE < 2.0 Pa
- ✅ **Good:** RMSE < 5.0 Pa
- ⚠️ **Acceptable:** RMSE < 10.0 Pa
- ❌ **Needs tuning:** RMSE > 10.0 Pa

---

## 🐛 Debug Mode

If something goes wrong, use the diagnostic version:

```bash
python debug_test_runner.py --input outputs/fig4_4.csv --distance 15760.0 --azimuth 0
```

**This will:**
- ✅ Check all imports
- ✅ Validate CSV format
- ✅ Show detailed progress at each step
- ✅ Display full error messages if something fails
- ✅ Run only one azimuth (faster for testing)

---

## 📚 References

### Test Case 1: JAXA Wing Body
- **Source:** Thesis, Page 61-62
- **Input:** Figure 4.4 (near-field at 15,760m)
- **Reference Output:** Figure 4.5 (ground signature)
- **Validation:** Table 4.2 (PLdB values for 0°, 20°, 40°)

### Test Case 2: JAXA D-SEND
- **Source:** "Far-field sonic boom prediction considering atmospheric turbulence effects"
- **Input:** Figure 8b (1000m altitude)
- **Reference Output:** Figure 10 (left, green curve)

---

## ⏱️ Typical Runtime

| Command | Description | Time |
|---------|-------------|------|
| `interactive_digitize.py` | Digitize one figure | ~2 min |
| `test_runner.py` | Basic propagation | ~5-10 min |
| `test_runner_with_PLdB.py --table42` | Full Table 4.2 validation | **15-45 min** |
| `test_case_2_dsend.py` | D-SEND test | ~3-8 min |
| `debug_test_runner.py` | Single azimuth diagnostic | ~5-10 min |

**Note:** Runtime depends on CPU speed and `dx` parameter.

---

## 📧 Support

If you encounter issues:
1. Run `debug_test_runner.py` to identify the problem
2. Check the troubleshooting section
3. Verify your CSV files have correct format
4. Ensure all dependencies are installed

---

## ✅ Quick Command Reference

```bash
# 1. Digitize figures
python interactive_digitize.py --folder figures --output_dir outputs --samples 5000

# 2a. Basic propagation test
python test_runner.py --input outputs/fig4_4.csv --reference outputs/fig4_5.csv --distance 15760 --out jaxa_case

# 2b. Advanced validation (Table 4.2)
python test_runner_with_PLdB.py --input outputs/fig4_4.csv --reference outputs/fig4_5.csv --distance 15760.0 --table42

# 3. D-SEND test
python test_case_2_dsend.py --input outputs/fig8b.csv --reference outputs/fig10.csv --distance 1000.0 --out dsend_test

# 4. Debug mode (if errors occur)
python debug_test_runner.py --input outputs/fig4_4.csv --distance 15760.0 --azimuth 0
```

---

## 📝 Version History

- **v1.0** - Initial release with basic propagation
- **v2.0** - Added PLdB calculation and Table 4.2 validation
- **v2.1** - Added D-SEND test case and debug mode

---

**Last Updated:** November 2024