import os
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.signal import argrelextrema


class Peak:
    target = "moBWHT"
    fileName = "TruePredDiff.xlsx"
    saveDir = "Result_peak"
    order = 30

    def __init__(self, name):
        self.name = str(name)
        self.dataDirs = None
        self.results_True_X = None
        self.results_Pred_X = None
        self.results_True_Y = None
        self.results_Pred_Y = None

    def load(self):
        self.dataDirs = natsorted(
            [
                os.path.join(path[0], path[2][0])
                for path in list(os.walk(self.name))
                if ((Peak.target in path[0]) and (Peak.fileName in path[2]))
            ]
        )

    def find_X(self, colData, results):
        ilocs_max = argrelextrema(colData.values, np.greater_equal, order=Peak.order)
        timing = ilocs_max[0]
        if len(timing) >= 1:
            total_peak = colData.iloc[timing]
            total_peak = total_peak.sort_values(ascending=False)
            results.append(
                [
                    total_peak.iloc[0],
                    total_peak.index[0],
                ]
            )
        else:
            results.append(
                [
                    0,
                    0,
                ]
            )
        return results

    def findPeak_X(self):
        df = pd.read_excel(self.dataDirs[0], sheet_name=["true_X", "pred_X"])
        results = []
        for (_, colData) in df[f"true_X"].iteritems():
            results = self.find_X(colData, results)
        self.results_True_X = pd.DataFrame(results, columns=["value", "timing"])
        results = []
        for (_, colData) in df[f"pred_X"].iteritems():
            results = self.find_X(colData, results)
        self.results_Pred_X = pd.DataFrame(results, columns=["value", "timing"])

    def find_Y(self, colData, results):
        ilocs_min = argrelextrema(colData.values, np.less_equal, order=Peak.order)
        timing = ilocs_min[0]
        if len(timing) >= 2:
            total_peak = colData.iloc[timing]
            total_peak = total_peak.sort_values(ascending=True)
            results.append(
                [
                    total_peak.iloc[0],
                    total_peak.index[0],
                    total_peak.iloc[1],
                    total_peak.index[1],
                ]
                if total_peak.index[0] < total_peak.index[1]
                else [
                    total_peak.iloc[1],
                    total_peak.index[1],
                    total_peak.iloc[0],
                    total_peak.index[0],
                ]
            )
        elif len(timing) == 1:
            total_peak = colData.iloc[timing]
            total_peak = total_peak.sort_values(ascending=True)
            results.append(
                [
                    total_peak.iloc[0],
                    total_peak.index[0],
                    0,
                    0,
                ]
            )
        else:
            results.append(
                [
                    0,
                    0,
                    0,
                    0,
                ]
            )
        return results

    def findPeak_Y(self):
        df = pd.read_excel(self.dataDirs[0], sheet_name=["true_Y", "pred_Y"])
        results = []
        for (_, colData) in df[f"true_Y"].iteritems():

            results = self.find_Y(colData, results)
        self.results_True_Y = pd.DataFrame(
            results, columns=["value_1", "timing_1", "value_2", "timing_2"]
        )
        results = []
        for (_, colData) in df[f"pred_Y"].iteritems():
            results = self.find_Y(colData, results)
        self.results_Pred_Y = pd.DataFrame(
            results, columns=["value_1", "timing_1", "value_2", "timing_2"]
        )

    def save(self):
        with pd.ExcelWriter(
            os.path.join(Peak.saveDir, f"peak_{self.name}.xlsx"), engine="xlsxwriter"
        ) as writer:
            self.results_True_X.to_excel(
                writer,
                sheet_name="true_X",
            )
            self.results_Pred_X.to_excel(
                writer,
                sheet_name="pred_X",
            )
            self.results_True_Y.to_excel(
                writer,
                sheet_name="true_Y",
            )
            self.results_Pred_Y.to_excel(
                writer,
                sheet_name="pred_Y",
            )
