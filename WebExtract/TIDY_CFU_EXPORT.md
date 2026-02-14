# Tidy CFU CSV export (WebExtract)

WebExtract exports a **tidy/long** CSV for CFU time-kill plots (one row per curve per timepoint) to match the Excel template sheet named `cfu`.

## Output schema (column order)

The exported CSV columns are **exactly**:

- `Time`
- `Log10cfu`
- `BLOD`
- `LOD`
- `Conc1`
- `Antibiotic1`
- `Conc2`
- `Antibiotic2`
- `Conc3`
- `Antibiotic3`
- `Strain`

This order is controlled by `TIDY_CFU_COLUMNS` in [WebExtract/templates/index.html](templates/index.html).

## Curve label parsing

Curve labels are split on `+` (combo treatments). Each segment is parsed as:

- `AntibioticN`: the drug/condition name
- `ConcN`: the numeric concentration (units are ignored for the Excel template)

Examples:

- `COL 0.125µg/mL` → `Antibiotic1=COL`, `Conc1=0.125`
- `COL 0.125µg/mL + MEM 4µg/mL` →
  - `Antibiotic1=COL`, `Conc1=0.125`
  - `Antibiotic2=MEM`, `Conc2=4`

Special cases:

- Labels containing `Blank` map to `AntibioticN=Blank`, `ConcN=`
- Labels containing `Control` map to `AntibioticN=Control`, `ConcN=`

Only the first **three** treatments are exported (`1..3`). Extra segments are ignored.

## BLOD / LOD behavior

- If `DEFAULT_LOD_LOG10` (in log10 CFU/mL) is set in [WebExtract/templates/index.html](templates/index.html), then:
  - `LOD` is set to that value on every row
  - `BLOD` is set to `1` when `Log10cfu < LOD`
- If `DEFAULT_LOD_LOG10` is empty, `LOD` and `BLOD` are left blank.

## Strain

- If `DEFAULT_STRAIN` is set, `Strain` is always that value.
- Otherwise WebExtract attempts to infer `Strain` from extracted facts (when available), falling back to blank.
