import os
import numpy as np
import pandas as pd
from natsort import natsorted


class Moment:
    target = "moBWHT"
    fileName = "TruePredDiff.xlsx"

    def __init__(self, name):
        self.name = str(name)
        self.dataDirs = None
        self.results_True_X = None
        self.results_Pred_X = None
        self.results_True_Y = None
        self.results_Pred_Y = None
        self.saveDir = None
        self.type = None

    def load(self):
        self.dataDirs = natsorted(
            [
                os.path.join(path[0], path[2][0])
                for path in list(os.walk(self.name))
                if ((Moment.target in path[0]) and (Moment.fileName in path[2]))
            ]
        )

    def save(self):
        with pd.ExcelWriter(
            os.path.join(
                self.saveDir, f"{str(self.saveDir).split('_')[-1]}_{self.name}.xlsx"
            ),
            engine="xlsxwriter",
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
