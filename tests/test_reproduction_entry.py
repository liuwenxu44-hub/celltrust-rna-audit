"""Fail-closed checks for the summary reader and original fold sentinels."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
repro=load('repro',ROOT/'scripts/reproduce_summaries.py')
score=load('score_cli',ROOT/'scripts/score.py')
class ReproductionTests(unittest.TestCase):
    def test_missing_not_zero(self):
        self.assertFalse(repro.equal([np.nan],[0.]));self.assertTrue(repro.equal([np.nan],[np.nan]))
    def test_changes_rejected(self):
        self.assertFalse(repro.equal([.01],[.010001]))
    def test_cluster_summary(self):
        x=np.array([[1.,3.],[3.,5.]])
        p,l,u,n=repro.summarize(x,np.array([[0,1],[1,0]]),[0,1])
        self.assertEqual((p,l,u,n),(3.,3.,3.,2))
    def test_missing_stratum_prevents_overall(self):
        x=np.array([[1.,np.nan],[3.,np.nan]])
        p,l,u,n=repro.summarize(x,np.array([[0,1]]),[0,1])
        self.assertTrue(np.isnan(p));self.assertEqual(n,0)
    def test_both_original_query_sentinels(self):
        for sentinel in ('OUTER_QUERY','OUTER_HOLDOUT'):
            d=pd.DataFrame({'cell_id':['a','b','c'],'donor_id':['d1','d2','d3'],'outer_fold':['O1','O2','O3'],
              'inner_O1':[sentinel,'I1','I2'],'inner_O2':['I1',sentinel,'I2'],'inner_O3':['I1','I2',sentinel]})
            self.assertEqual(len(score.validate_design(d)),9)
            d.loc[0,'inner_O1']='I1'
            with self.assertRaises(ValueError):score.validate_design(d)
    def test_existing_destination_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(AssertionError):repro.run(ROOT/'source_data',Path(t))
if __name__=='__main__':unittest.main()
