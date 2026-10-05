# DealerRisk

A runnable local Python application for independent dealership loan-portfolio
analytics. Upload a borrower-level CSV, compare loss models, stress your
assumptions, and export a printable report or borrower contribution table.

## Start on Windows

1. Install Python **3.10 or newer** from https://www.python.org/downloads/.
2. Extract the entire ZIP into a folder.
3. Double-click `run_windows.bat`. The first launch installs three dependencies.
4. Open **http://127.0.0.1:8765** in your browser. A synthetic portfolio runs automatically.
5. Leave the terminal open while using the app. Press Ctrl+C there to stop it.

On macOS/Linux: open a terminal in this folder and run `sh run_mac_linux.sh`.

Manual setup (Windows PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
```

Manual setup (macOS/Linux):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Installation requires internet access. The running application uses no external
API, CDN, analytics service, or database. The dashboard binds only to your local
machine. If port 8765 is occupied, run `python app.py --port 8766` with your
environment's Python and open the displayed address.

## Included features

- Three loss models: independent Bernoulli defaults, finite-portfolio Vasicek,
  and a multi-factor CreditRisk+ compound-Poisson/gamma simulation.
- Expected loss, simulated mean, Monte Carlo mean standard error, loss standard
  deviation, VaR, expected shortfall, and a VaR-minus-EL buffer.
- Side-by-side base and stress scenarios; probability of exceeding a chosen
  loss budget; adjustable dependence assumptions and random seed.
- Exposure and expected-loss concentration by sector and borrower.
- Current 30/60/90+ DPD rates and origination-quarter snapshots.
- Searchable borrower records, CSV export, and a self-contained HTML report
  you can open and print to PDF from a browser.
- Strict CSV validation, reproducible simulation streams, and numerical tests.

## Loan CSV contract

**One row per distinct borrower, one currency (USD), one 12-month PD horizon.**
The portfolio as-of date is chosen in the dashboard. All input estimates must
refer to that date. This is a snapshot of non-defaulted receivables, including
delinquent loans that have not yet met your consistent default definition.

| Column | Required | Meaning |
|---|---|---|
| `loan_id` | Yes | Unique borrower-level exposure identifier; text, maximum 120 characters |
| `ead` | Yes | Assumed dollars of exposure at default over the next 12 months; nonnegative |
| `pd_12m` | Yes | Supplied 12-month probability of default; decimal between 0 and 1 |
| `lgd` | Yes | Supplied loss fraction after recovery and costs; decimal between 0 and 1 |
| `sector` | No | Shared-risk grouping, such as branch or geography; maximum 120 characters |
| `days_past_due` | No | Nonnegative integer at the as-of date; blank means unknown |
| `origination_date` | No | `YYYY-MM-DD`, not later than the as-of date; blank means unknown |
| `borrower_id` | No | If supplied, must be nonempty and unique; repeated borrowers are rejected |

Example:

```csv
loan_id,ead,pd_12m,lgd,sector,days_past_due,origination_date
0001,12000,0.12,0.55,North,0,2025-10-01
0002,17500,0.25,0.65,South,35,2026-03-15
```

**12% is `0.12`, not `12`.** Missing PD or LGD values are rejected, rather than
filled with an invented scoring rule. The example CSV contains synthetic
assumptions, not an industry calibration or a recommended underwriting policy.

For several loans to one borrower, aggregate EAD, supply one borrower PD, and
use EAD-weighted LGD before upload. The application cannot discover hidden
borrower overlap if you supply only loan IDs. Delinquency rates will then count
borrower-level records rather than separate contracts. Already-defaulted
receivables need a separate recovery-risk treatment.

The local version accepts up to 2,000 borrowers, 2 MB CSV input, and 50,000
simulations per model/scenario. Large runs can take several seconds. Blank
optional DPD values are excluded from delinquency denominators. Absent sector
labels use one shared sector. Unknown origination dates are excluded from
vintage snapshots. Extra columns are ignored by the analytical engine.

## Model definitions

### Independent baseline

For each borrower, `D_i ~ Bernoulli(PD_i)` independently and
`L = sum(EAD_i * LGD_i * D_i)`. This is a transparent benchmark without
systematic default dependence.

### Vasicek credit portfolio model

One shared factor `Z ~ N(0,1)` and independent shocks `epsilon_i ~ N(0,1)`:

```text
A_i = sqrt(rho) * Z + sqrt(1-rho) * epsilon_i
D_i = 1[A_i < Phi_inverse(PD_i)]
P(default_i | Z) = Phi((Phi_inverse(PD_i) - sqrt(rho)*Z) / sqrt(1-rho))
```

The code samples conditional Bernoulli defaults. This preserves each marginal
PD and retains finite-portfolio concentration and idiosyncratic default risk.
`rho` is latent asset correlation, not default-indicator correlation. This is
the **credit model**, rather than the Vasicek mean-reverting short-rate process
discussed in interest-rate modelling texts.

### CreditRisk+ Monte Carlo model

Independent mean-one gamma macro and sector factors with variance `CV^2`:

```text
G_0, G_sector ~ Gamma(shape=1/CV^2, scale=CV^2)
lambda_i = PD_i * ((1-w) + w*(m*G_0 + (1-m)*G_sector(i)))
N_i | G ~ Poisson(lambda_i)
L = sum(EAD_i * LGD_i * N_i)
```

`w` is the systematic weight, `m` the macro share. Each borrower has a
deterministic idiosyncratic weight `1-w`, macro weight `w*m`, and its sector
weight `w*(1-m)`; these sum to one. CV=0 gives factors identically equal to one.
If all borrowers share one sector, its factor and the macro factor remain
independent; macro share therefore still affects the total factor variance.

This is the original compound-Poisson loss approximation simulated directly,
not the analytical generating-function/Panjer-recursion implementation. **Do
not interpret `N_i` as a bounded default indicator:** counts can exceed one,
and losses can exceed the sum of all loan severities. The PD input acts as a
mean count intensity here; `P(N_i >= 1)` is not exactly PD. Low PDs make the
Poisson approximation more plausible. With high-PD BHPH loans, favor the
bounded Vasicek model and treat CreditRisk+ as an approximation comparison.
The UI warns when PD exceeds 10%; this is an illustrative diagnostic
threshold, not an empirically established model acceptance criterion.

Let `c_i = EAD_i*LGD_i`, `EL_i = PD_i*c_i`, and `EL_s` be sector EL. Then:

```text
E[L] = sum(EL_i)
Var(L) = sum(PD_i*c_i^2)
       + CV^2*w^2 * (m^2*EL_total^2 + (1-m)^2*sum(EL_s^2))
```

The app compares this exact variance with simulation as a numerical diagnostic.

### Risk outputs and stress

- EL is exactly `sum(PD*LGD*EAD)` for all three implemented models; numerical
  simulated means vary around it.
- VaR is the empirical inverse CDF quantile (`inverted_cdf` convention).
- ES averages the worst `1-alpha` fraction of loss mass, including fractional
  mass at the quantile. This correctly handles discrete loss distributions.
- The analytical buffer is `max(VaR - EL, 0)`, not a regulatory capital amount.
- Monte Carlo SE measures only numerical uncertainty in the simulated mean.
  It does not describe model, parameter, or forecast uncertainty.
- Stress transforms PD to `min(multiplier*PD, 1)` and LGD to
  `min(LGD + increase, 1)`; EAD and dependence settings remain fixed.
- Base and stress use the same model-specific seed. Poisson draws consume
  random numbers differently when intensities change; CreditRisk+ scenarios
  are not pathwise paired. Sampling noise can affect comparisons.
- Histogram bins match between base and stress for a given model, but differ
  between models. Histograms show mass per bin, not a continuous density.
- Settings, input SHA-256 hash, as-of date and assumptions are included in
  report exports. A warning appears when fewer than 100 simulations support
  the selected tail probability.

## What this version does and does not establish

This is a functional **analytical MVP**, suitable for exploration and further
development. It is not yet validated for dealership lending or collections
decisions. Numerical correctness tests are not empirical model validation.

The app **does not fit PD, LGD, rho or gamma parameters**. A current loan tape
does not supply the outcome history needed to do that well. Next development
should add monthly performance snapshots, origination features known at the
prediction date, a documented default definition, recoveries and costs,
exposure evolution, and horizon-aware handling of censored observations. Then
fit and test PD/LGD on earlier cohorts and validate on later cohorts. Compare
calibration, discrimination, stability, and actual portfolio loss outcomes.

Current vintage tables are snapshots, not default curves at equal months on
book. No significance or causal claim is made about vintage differences or
collections efficacy. Expected-loss shares are not tail-risk allocations.
EAD and LGD are deterministic; amortization, prepayments, recovery delays,
stochastic LGD, profitability, and mark-to-market changes are outside this
version. No interest-rate model is added solely because an interest-rate book
was supplied; it is not needed for this default-loss engine.

Uploaded portfolios remain in memory and are not saved by the server. Browser
downloads and CLI outputs are files you explicitly create and should protect
as loan-portfolio data. The local server is intended for one user; it has no
authentication, multi-tenant isolation or hosted production deployment.

## Files and tests

| File | Purpose |
|---|---|
| `app.py` | Local HTTP server; run this to launch the dashboard |
| `dashboard.html` | Self-contained interface, charts and report export |
| `risk_engine.py` | Model implementations, validation, concentration and risk calculations |
| `analyze_csv.py` | Batch/CLI analysis using the same model engine |
| `example_loans.csv` | Reproducible synthetic demonstration portfolio |
| `test_models.py` | Model, validation, HTTP and export tests |
| `test_dashboard.js` | Optional dependency-free Node.js smoke test for interface logic and HTTP flow |
| `requirements.txt` | Python dependencies |
| `run_windows.bat` / `run_mac_linux.sh` | Convenience launchers |

Run tests using your environment's Python:

```bash
python -m unittest -v test_models
```

The tests compare simulation moments with exact Bernoulli and gamma-Poisson
moments, verify independence and correlation behavior, check boundary cases
and discrete ES, validate stress caps and CSV rejection, and exercise the HTTP
upload/analysis/export workflow.

`node test_dashboard.js` additionally checks dashboard logic while the server
is running. It uses a DOM stub; it is not a real-browser CSS/layout test. A
sample synthetic report is included as `demo_report.html` for inspection
without starting Python.

CLI example:

```bash
python analyze_csv.py example_loans.csv --output analysis_output
python analyze_csv.py example_loans.csv --settings example_settings.json --output analysis_output
```

`example_settings.json` contains all model settings. CLI defaults use a fixed
as-of date of 2026-10-02; set the date to your data's actual snapshot date. The
dashboard initializes the date to the local server's current date.

## References

These are implementation references, not reproduced software or book content:

1. Credit Suisse Financial Products (1997), *CreditRisk+: A Credit Risk Management
   Framework*, especially the compound-Poisson frequency model and gamma sector
   factors. Original document mirrored at
   https://globalriskguard.com/resources/credit/creditrisk.pdf
2. David Jamieson Bolder (2018), *Credit-Risk Modelling*, chapter 3,
   §§3.3.1–3.3.2 (pp. 131–147). Note: the book's illustrative Monte Carlo code
   caps Poisson counts to indicators; this implementation retains uncapped
   counts for the original compound-Poisson approximation and its exact moments.
3. Basel Committee (2005), *An Explanatory Note on the Basel II IRB Risk Weight
   Functions*, for the Gaussian single-factor framework:
   https://www.bis.org/publications/explanatory-note-basel-ii-irb-risk-weight-functions

CreditRisk+ is a name associated with Credit Suisse. DealerRisk is an independent
implementation with no affiliation or endorsement. Original code is licensed
under the included MIT license. The supplied reference books are not included.
