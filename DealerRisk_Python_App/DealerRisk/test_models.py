"""Numerical and input contract tests; no additional test dependency required."""
import json
from http.server import ThreadingHTTPServer
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from app import Handler
from risk_engine import (Settings, analyze, creditrisk_variance, demo_csv,
                         load_portfolio, safe_csv, simulate, summarize)


def portfolio(n=100, p=0.08, ead=1000, lgd=0.5):
    return pd.DataFrame({"loan_id": [str(i) for i in range(n)], "ead": ead,
                         "pd_12m": p, "lgd": lgd, "sector": [str(i%2) for i in range(n)]})


class ModelTests(unittest.TestCase):
    def test_zero_pd(self):
        frame = portfolio(p=0)
        for name in ["Independent", "Vasicek", "CreditRisk+"]:
            self.assertTrue(np.all(simulate(frame, Settings(trials=1000), name) == 0))

    def test_certain_default_is_bounded(self):
        frame = portfolio(n=6, p=1)
        for name in ["Independent", "Vasicek"]:
            self.assertTrue(np.all(simulate(frame, Settings(trials=1000), name) == 3000))

    def test_bernoulli_moments_and_vasicek_rho_zero(self):
        frame = portfolio()
        s = Settings(trials=50000, rho=0)
        expected_mean = 100*0.08*500
        expected_var = 100*0.08*0.92*500**2
        for name in ["Independent", "Vasicek"]:
            losses = simulate(frame, s, name)
            se = np.sqrt(expected_var/s.trials)
            self.assertLess(abs(losses.mean()-expected_mean), 5*se)
            self.assertLess(abs(losses.var()/expected_var-1), 0.04)

    def test_vasicek_preserves_pd_and_increases_variance(self):
        frame = portfolio(n=150)
        s = Settings(trials=50000, rho=0.25)
        losses = simulate(frame, s, "Vasicek")
        target = float((frame.ead*frame.lgd*frame.pd_12m).sum())
        self.assertLess(abs(losses.mean()-target), 5*losses.std()/np.sqrt(s.trials))
        independent = simulate(frame, s, "Independent")
        self.assertGreater(losses.var(), independent.var()*3)
        self.assertLessEqual(losses.max(), float((frame.ead*frame.lgd).sum()))

    def test_creditrisk_exact_moments(self):
        frame = portfolio(n=80)
        for cv, weight, macro in [(0, 0.75, 0.5), (0.75, 0, 0.5), (0.75, 0.75, 0.5), (1, 1, 1), (0.5, 1, 0)]:
            s = Settings(trials=50000, factor_cv=cv, systematic_weight=weight, macro_share=macro)
            losses = simulate(frame, s, "CreditRisk+")
            mean = float((frame.ead*frame.lgd*frame.pd_12m).sum())
            variance = creditrisk_variance(frame, s)
            self.assertLess(abs(losses.mean()-mean), 5*np.sqrt(variance/s.trials))
            self.assertLess(abs(losses.var()/variance-1), 0.07)

    def test_creditrisk_counts_are_not_capped(self):
        losses = simulate(portfolio(n=1, p=1), Settings(trials=1000, factor_cv=0), "CreditRisk+")
        self.assertGreater(losses.max(), 500)

    def test_discrete_es_fractional_quantile_mass(self):
        loss = np.array([0]*95+[100]*5, dtype=float)
        r = summarize(loss, 0.90, 5, 50)
        self.assertEqual(r["var"], 0)
        self.assertAlmostEqual(r["es"], 50)
        r = summarize(loss, 0.99, 5, 50)
        self.assertEqual(r["var"], 100)
        self.assertAlmostEqual(r["es"], 100)
        self.assertAlmostEqual(r["budget_exceedance"], .05)

    def test_reproducibility(self):
        f = portfolio()
        s = Settings(trials=1000)
        for name in ["Independent", "Vasicek", "CreditRisk+"]:
            np.testing.assert_array_equal(simulate(f, s, name), simulate(f, s, name))

    def test_demo_and_stress_accounting(self):
        csv_text = demo_csv(n=30)
        r = analyze(csv_text, Settings(trials=1000))
        self.assertEqual(r["overview"]["accounts"], 30)
        self.assertAlmostEqual(sum(a["expected_loss"] for a in r["accounts"]), r["overview"]["expected_loss"])
        self.assertAlmostEqual(sum(a["expected_loss"] for a in r["sectors"]), r["overview"]["expected_loss"])
        self.assertGreater(r["overview"]["stress_expected_loss"], r["overview"]["expected_loss"])
        self.assertTrue(any("Poisson" in w for w in r["warnings"]))
        self.assertTrue(any("tail" in w for w in r["warnings"]))
        json.dumps(r, allow_nan=False)

    def test_stress_caps(self):
        csv_text = "loan_id,ead,pd_12m,lgd\na,1000,0.9,0.9\n"
        r = analyze(csv_text, Settings(trials=1000, pd_multiplier=5, lgd_add=0.5))
        self.assertEqual(r["overview"]["stress_expected_loss"], 1000)


class InputTests(unittest.TestCase):
    def test_minimal_csv_and_leading_zero_id(self):
        f, w = load_portfolio("loan_id,ead,pd_12m,lgd\n001,100,0.12,0.55\n", "2026-10-02")
        self.assertEqual(f.loan_id.iloc[0], "001")
        self.assertEqual(f.sector.iloc[0], "Whole portfolio")

    def test_bad_inputs(self):
        header = "loan_id,ead,pd_12m,lgd\n"
        examples = [header, header+"a,-1,0.12,0.5", header+"a,100,12,0.5",
            header+"a,100,nan,0.5", header+"a,inf,0.12,0.5", header+"a,100,0.12,1.5",
            header+"a,100,0.12,0.5\na,100,0.12,0.5", "loan_id,ead,lgd\na,100,0.5",
            "loan_id,ead,pd_12m,lgd,ead\na,100,0.1,0.5,100", header+"a,100,0.1,0.5,EXTRA",
            header+",100,0.1,0.5", header+"a,0,0.1,0.5"]
        for text in examples:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_portfolio(text, "2026-10-02")

    def test_optional_field_validation(self):
        base = "loan_id,ead,pd_12m,lgd,days_past_due,origination_date\n"
        for row in ["a,100,.1,.5,-1,2025-01-01", "a,100,.1,.5,1.2,2025-01-01",
                    "a,100,.1,.5,foo,2025-01-01", "a,100,.1,.5,0,2027-01-01",
                    "a,100,.1,.5,0,not-a-date"]:
            with self.assertRaises(ValueError):
                load_portfolio(base+row, "2026-10-02")
        f, _ = load_portfolio(base+"a,100,.1,.5,,", "2026-10-02")
        self.assertTrue(f.days_past_due.isna().all())

    def test_shared_borrower_rejected(self):
        with self.assertRaisesRegex(ValueError, "aggregate"):
            load_portfolio("loan_id,ead,pd_12m,lgd,borrower_id\na,100,.1,.5,x\nb,100,.1,.5,x", "2026-10-02")

    def test_nonfinite_and_out_of_range_settings(self):
        for s in [Settings(rho=1), Settings(factor_cv=-1), Settings(seed=-1), Settings(trials=1),
                  Settings(confidence=1), Settings(rho=float("nan")), Settings(trials=1000.1),
                  Settings(as_of="bad"), Settings(lgd_add=True)]:
            with self.assertRaises(ValueError):
                s.validate()

    def test_formula_injection_export(self):
        out = safe_csv([{"loan_id": "=HYPERLINK(1)", "sector": " +cmd", "loss": 12.3}])
        self.assertIn("'=HYPERLINK", out)
        self.assertIn("' +cmd", out)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:"+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_demo_upload_analyze_and_export(self):
        with urlopen(self.base+"/api/demo") as response:
            self.assertIn(b"pd_12m", response.read())
        payload = json.dumps({"csv": demo_csv(20), "settings": {"trials": 1000}}).encode()
        request = Request(self.base+"/api/analyze", payload, {"Content-Type": "application/json", "Origin": self.base})
        with urlopen(request) as response:
            result = json.load(response)
            self.assertEqual(result["overview"]["accounts"], 20)
        request = Request(self.base+"/api/export", json.dumps({"rows": result["accounts"]}).encode(), {"Content-Type": "application/json"})
        with urlopen(request) as response:
            self.assertIn(b"stress_expected_loss", response.read())

    def test_bad_origin_and_bad_csv(self):
        for origin, payload, code in [("https://other.example", {}, 403),
            (self.base, {"csv": "bad", "settings": {"trials": 1000}}, 400),
            (self.base, {"csv": demo_csv(2), "settings": {"fake": 1}}, 400)]:
            request = Request(self.base+"/api/analyze", json.dumps(payload).encode(), {"Content-Type": "application/json", "Origin": origin})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request)
            self.assertEqual(caught.exception.code, code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
