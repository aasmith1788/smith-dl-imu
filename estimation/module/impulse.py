import os
import numpy as np
import pandas as pd
from natsort import natsorted

import module.moment as m


class Impulse(m.Moment):
    def __init__(self, name):
        super().__init__(name)
        self.saveDir = "Result_impulse"

    def cal_X(self, colData, results):
        colData[colData < 0] = 0  # 양수인 것만 적분
        results.append(np.trapz(colData))
        return results

    def calImpulse_X(self):
        df = pd.read_excel(self.dataDirs[0], sheet_name=["true_X", "pred_X"])
        results = []
        for (_, colData) in df[f"true_X"].iteritems():
            results = self.cal_X(colData, results)
        self.results_True_X = pd.DataFrame(
            results,
            columns=[
                "value",
            ],
        )
        results = []
        for (_, colData) in df[f"pred_X"].iteritems():
            results = self.cal_X(colData, results)
        self.results_Pred_X = pd.DataFrame(
            results,
            columns=[
                "value",
            ],
        )

    def cal_Y(self, colData, results):
        colData[colData > 0] = 0  # 음수인 것만 적분
        results.append(np.trapz(colData))
        return results

    def calImpulse_Y(self):
        df = pd.read_excel(self.dataDirs[0], sheet_name=["true_Y", "pred_Y"])
        results = []
        for (_, colData) in df[f"true_Y"].iteritems():
            results = self.cal_Y(colData, results)
        self.results_True_Y = pd.DataFrame(
            results,
            columns=[
                "value",
            ],
        )
        results = []
        for (_, colData) in df[f"pred_Y"].iteritems():
            results = self.cal_Y(colData, results)
        self.results_Pred_Y = pd.DataFrame(
            results,
            columns=[
                "value",
            ],
        )
