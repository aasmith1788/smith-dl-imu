# PDF Review and Data Classification Script
# Purpose: Open PDFs sequentially, allow manual review with keyboard input, 
# automatically save decisions, and quickly review data files

# Required libraries
import os
from re import A
import subprocess
from natsort import natsorted
import pandas as pd
import keyboard
import signal
import subprocess
import psutil
import math
import shutil
import argparse
from tqdm import tqdm

# Function definitions

def refresh_dir(file_path):
    """Remove directory if it exists, then create a new empty one"""
    if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(file_path)

def ensure_dir(file_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def kill(proc_pid):
    """Kill a process and all its child processes"""
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

# Program modes: 1. Review PDFs and create classification list 2. Move CSV files based on classification list
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    "-mo",
    required=False,
    default="CHK",
    type=str,
    help="mode: CHK (check/review) | PP (postponed items) | EXP (export/move files)",
)
args = parser.parse_args()

# File directory paths
# Look through RAW folder, open PDFs from FIG folder, save decisions to move files later
dataDir = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero"
rawDir = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\RAW_AXIS_corrected"
figDir = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\FIG"
includeDir = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked\RAW"
excludeDir = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\EXC"
postpondDir = r"R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\PP"
dataExt = r".csv"

# Get list of all CSV files for review
dataList = natsorted([_ for _ in os.listdir(rawDir) if _.endswith(dataExt)])
excluded_fig = []  # Track excluded files (1 = excluded)

# Load or create the classification tracking Excel file
# This file tracks which files have been reviewed and their classification status
# Shows progress so you can see where you left off and continue from there
listName = r"list_Excluded_byFig.xlsx"
if os.path.isfile(os.path.join(dataDir, listName)):
    # Load existing classification list
    list_Excluded_byFig = pd.read_excel(
        os.path.join(dataDir, listName), engine="openpyxl"
    )
else:
    # Create new classification list with all files marked as "To Be Determined"
    list_Excluded_byFig = pd.DataFrame(columns=["filename", "type", "note", "checker"])
    list_Excluded_byFig["filename"] = dataList
    list_Excluded_byFig["type"] = "TBD"  # TBD = To Be Determined

# Get reviewer's name for tracking who made decisions
print("\n= Type your Name = ")
chk_name = input()
print(f"{chk_name} is entering...")

# Show current classification status
print("\n== Status ==")
print(list_Excluded_byFig["type"].value_counts())
if "TBD" in list_Excluded_byFig["type"].value_counts():
    num_TBD = list_Excluded_byFig["type"].value_counts()["TBD"]
if "PP" in list_Excluded_byFig["type"].value_counts():
    num_PP = list_Excluded_byFig["type"].value_counts()["PP"]

# === EXPORT MODE: Move files based on classification decisions ===
if args.mode == "EXP":
    print("\n=== EXPORT MODE ===")
    print("now exporting files...")
    
    # Create destination directories
    ensure_dir(includeDir)
    ensure_dir(excludeDir)
    ensure_dir(postpondDir)

    # Move files based on their classification in the Excel list
    for file in tqdm(list_Excluded_byFig["filename"]):
        # Move files marked as "IN" (include) to include directory
        if (
            list_Excluded_byFig.loc[list_Excluded_byFig.filename == file, "type"]
            == "IN"
        ).bool():
            if not os.path.exists(os.path.join(includeDir, file)):
                shutil.copyfile(
                    os.path.join(rawDir, file), os.path.join(includeDir, file)
                )

        # Move files marked as "EXC" (exclude) to exclude directory
        elif (
            list_Excluded_byFig.loc[list_Excluded_byFig.filename == file, "type"]
            == "EXC"
        ).bool():
            if not os.path.exists(os.path.join(excludeDir, file)):
                shutil.copyfile(
                    os.path.join(rawDir, file), os.path.join(excludeDir, file)
                )
        
        # Move files marked as "PP" (postponed) to postponed directory
        elif (
            list_Excluded_byFig.loc[list_Excluded_byFig.filename == file, "type"]
            == "PP"
        ).bool():
            if not os.path.exists(os.path.join(postpondDir, file)):
                shutil.copyfile(
                    os.path.join(rawDir, file), os.path.join(postpondDir, file)
                )
    print("Done!")

# === CHECK MODE: Review PDFs and classify files ===
elif args.mode == "CHK":
    print("\n=== CHECK MODE ===")
    print("||   KEY - FUNCTION   ||")
    print("|| 'w' - go next file || 'p' - terminate program ||")
    print("|| 'q' - for include  || 'e' - for exclude       ||")
    print("|| 's' - for postpone ||                         ||")
    
    num_trial = 0
    firstScene = True
    
    # Go through each file that hasn't been classified yet
    for datum in dataList:
        if (
            list_Excluded_byFig.loc[
                list_Excluded_byFig.filename == datum, "type"
            ].values
            == "TBD"
        ):
            # Wait for user to press 'w' to continue or 'p' to quit
            while True:
                if keyboard.read_key(suppress=True) == "w":
                    print(f"\nNo:{num_trial} | remains: {num_TBD-num_trial}")
                    num_trial = num_trial + 1
                    break
                if keyboard.read_key(suppress=True) == "p":
                    print("\nwait for saving the list...")
                    list_Excluded_byFig.to_excel(
                        os.path.join(dataDir, listName), index=False
                    )
                    print("See you!")
                    quit()
            
            print(f"Target: {datum}")
            # Open the corresponding PDF file for visual inspection
            plot = subprocess.Popen(
                [os.path.join(figDir, datum.split(".")[-2] + ".pdf")], shell=True
            )
            
            # Wait for user decision on the displayed PDF
            while True:
                if keyboard.read_key(suppress=True) == "q":  # Include file
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "type"
                    ] = "IN"
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "checker"
                    ] = chk_name
                    print(f"Included!: {datum}")
                    break
                    
                if keyboard.read_key(suppress=True) == "e":  # Exclude file
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "type"
                    ] = "EXC"
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "checker"
                    ] = chk_name
                    print(f"Excluded!: {datum}")
                    break
                    
                if keyboard.read_key(suppress=True) == "s":  # Postpone file (needs more review)
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "type"
                    ] = "PP"
                    print("Enter your note:")
                    reason = input()
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "note"
                    ] = reason
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "checker"
                    ] = chk_name
                    print(f"Postponed!: {datum}")
                    break

            # Close the PDF viewer process
            kill(plot.pid)
            firstScene = False
            
    print("\nThere is no data!")
    print("wait for saving the list...")
    list_Excluded_byFig.to_excel(os.path.join(dataDir, listName), index=False)
    print("See you!")
    quit()

# === PP MODE: Review previously postponed files ===
elif args.mode == "PP":
    print("\n=== POSTPONED REVIEW MODE ===")
    print("||   KEY - FUNCTION   ||")
    print("|| 'w' - go next file || 'p' - terminate program ||")
    print("|| 'q' - for include  || 'e' - for exclude       ||")
    print("|| 's' - keep postponed ||                       ||")
    
    num_trial = 0
    firstScene = True
    
    # Go through files that were previously marked as postponed
    for datum in dataList:
        if (
            list_Excluded_byFig.loc[list_Excluded_byFig.filename == datum, "type"]
            == "PP"
        ).bool():
            # Similar workflow as CHK mode but only for postponed files
            while True:
                if keyboard.read_key(suppress=True) == "w":
                    print(f"\nNo:{num_trial} | remains: {num_PP-num_trial}")
                    num_trial = num_trial + 1
                    break
                if keyboard.read_key(suppress=True) == "p":
                    print("\nwait for saving the list...")
                    list_Excluded_byFig.to_excel(
                        os.path.join(dataDir, listName), index=False
                    )
                    print("See you!")
                    quit()
                    
            print(f"Target: {datum}")
            plot = subprocess.Popen(
                [os.path.join(figDir, datum.split(".")[-2] + ".pdf")], shell=True
            )
            
            # Same decision options as CHK mode
            while True:
                if keyboard.read_key(suppress=True) == "q":  # Include file
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "type"
                    ] = "IN"
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "checker"
                    ] = chk_name
                    print(f"Included!: {datum}")
                    break
                    
                if keyboard.read_key(suppress=True) == "e":  # Exclude file
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "type"
                    ] = "EXC"
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "checker"
                    ] = chk_name
                    print(f"Excluded!: {datum}")
                    break
                    
                if keyboard.read_key(suppress=True) == "s":  # Keep postponed
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "type"
                    ] = "PP"
                    print("Enter your note:")
                    reason = input()
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "note"
                    ] = reason
                    list_Excluded_byFig.loc[
                        list_Excluded_byFig.filename == datum, "checker"
                    ] = chk_name
                    print(f"Postponed!: {datum}")
                    break

            kill(plot.pid)
            firstScene = False
            
    print("\nThere is no data!")
    print("wait for saving the list...")
    list_Excluded_byFig.to_excel(os.path.join(dataDir, listName), index=False)
    print("See you!")