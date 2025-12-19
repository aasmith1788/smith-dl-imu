"""
Dataset Creation Script with Demographics
==========================================
Creates 5-fold cross-validation datasets for IMU knee prediction models.
Includes demographic information (age, height, weight, sex) for each trial.

Demographics:
- age: participant age
- height: participant height (cm)
- weight_bl: participant baseline weight (kg)
- sex: participant sex (0=female, 1=male)

Output:
- Dataset name: IWALQQ_1st_correction_withDemographics
- Location: SAVE_dataSet/IWALQQ_1st_correction_withDemographics/
- Files per fold: train.npz, test.npz, and 4 scaler .pkl files
"""

import random
import pandas as pd
import numpy as np
from numpy import savez_compressed
from numpy import load
from natsort import natsorted
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
from os.path import join
from pickle import dump

# Configuration
SEED_RAND = 41
DATASET_NAME = "IWALQQ_1st_correction_withDemographics"

# File paths
DATA_DIR = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\Data\allnew_20220325_raw_byDeepak_csv\INC_ByStep\INC_ByZero\Included_checked'
NORMALIZED_DIR = join(DATA_DIR, r'NORM')
DEMOGRAPHICS_PATH = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\IMUforKnee\preperation\demographics.xlsx'
LIST_FILE = r'list_dataset_correction.xlsx'

def makeColumnsWOMAG():
    """
    Creates column names for biomechanical data excluding magnetometer (MAG) sensors.
    Returns standardized column names for IMU and biomech without MAG columns.
    """
    SIDEIDX = ['non','oa']
    PARTIDX = ['shank','shoe','thigh']
    TYPEIDX = ['ACC','GYRO','MAG']
    AXISIDX = ['X', 'Y', 'Z']

    LEGCOLUMNSLENGTH = 54
    COl_imu_legs = [f'{SIDEIDX[int(i//(LEGCOLUMNSLENGTH/2))]}_{PARTIDX[(i//(len(TYPEIDX)*len(AXISIDX)))%len(PARTIDX)]}_{TYPEIDX[(i//(len(AXISIDX)))%len(TYPEIDX)]}_{AXISIDX[i%len(AXISIDX)]}' for i in range(0,LEGCOLUMNSLENGTH)]

    TRKCOLUMNSLENGTH = 9
    Col_imu_trunk = [f'trunk_{TYPEIDX[(i//(len(AXISIDX)))%len(TYPEIDX)]}_{AXISIDX[i%len(AXISIDX)]}' for i in range(0,TRKCOLUMNSLENGTH)]

    FPCOLUMNSLENGTH = 12
    FPTYPEIDX = ['GRF','ANGLE','MONM','MOBWHT']
    Col_FP = [f'{FPTYPEIDX[(i//(len(AXISIDX)))%len(FPTYPEIDX)]}_{AXISIDX[i%len(AXISIDX)]}' for i in range(0,FPCOLUMNSLENGTH)]

    newColumns = COl_imu_legs+Col_imu_trunk+Col_FP
    newColumnswithoutMAG = [col for col in newColumns if not 'MAG' in col]
    return newColumnswithoutMAG

def kfold2subfold(arrName, listData, train, test):
    """
    Converts subject-based train/test splits to data index-based splits.
    """
    arrTrain = []
    arrTest = []

    for pID in arrName[train]:
        idxofID = listData.index[listData['patientID']==pID].copy()
        arrTrain.extend(idxofID.to_list())

    for pID in arrName[test]:
        idxofID = listData.index[listData['patientID']==pID].copy()
        arrTest.extend(idxofID.to_list())

    return arrTrain, arrTest

def make_dir(file_path):
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        print(f"Created directory: {file_path}")

def sex_to_binary(sex_value):
    """
    Convert sex to binary encoding.
    0 = female, 1 = male
    Handles various formats: 'F'/'M', 'Female'/'Male', 'f'/'m', etc.
    """
    if pd.isna(sex_value):
        raise ValueError("Sex value is missing (NaN)")

    sex_str = str(sex_value).strip().lower()

    if sex_str in ['f', 'female', 'woman', '0']:
        return 0
    elif sex_str in ['m', 'male', 'man', '1']:
        return 1
    else:
        raise ValueError(f"Unknown sex value: {sex_value}. Expected 'F'/'Female' or 'M'/'Male'.")

def load_data():
    """Load all necessary data files and return them."""
    print("="*80)
    print("LOADING DATA FILES")
    print("="*80)

    # Get CSV file list
    dataExt = r".csv"
    listFromFolder = natsorted([_ for _ in os.listdir(NORMALIZED_DIR) if _.endswith(dataExt)])
    print(f"✓ Found {len(listFromFolder)} CSV files in NORM directory")

    # Load file list metadata
    listFromxlsx = pd.read_excel(join(DATA_DIR, LIST_FILE))
    print(f"✓ Loaded metadata: {len(listFromxlsx)} trials")

    # Load demographics
    dg_Fromxlsx = pd.read_excel(DEMOGRAPHICS_PATH)
    print(f"✓ Loaded demographics: {len(dg_Fromxlsx)} subjects")

    # Extract unique subjects
    arrName = listFromxlsx.patientID.unique()
    print(f"✓ Total unique subjects: {len(arrName)}")

    # Verify data consistency
    if len(listFromxlsx) != len(listFromFolder):
        raise ValueError(f"Mismatch: {len(listFromxlsx)} metadata rows vs {len(listFromFolder)} files")

    print("✓ All data loaded successfully!\n")
    return listFromFolder, listFromxlsx, dg_Fromxlsx, arrName

def process_fold(countfold, idx4train, idx4test, listFromFolder, listFromxlsx, dg_Fromxlsx):
    """Process a single fold: load data, scale, reshape, and save."""

    print(f"\n{'='*80}")
    print(f"PROCESSING FOLD {countfold}")
    print(f"{'='*80}")
    print(f"Training samples: {len(idx4train)}")
    print(f"Testing samples:  {len(idx4test)}")

    # Initialize data structures
    columnsWOMAG = makeColumnsWOMAG()
    X_columns = [str(i) for i in range(0,42)]
    Y_columns = [str(i) for i in range(0,3)]

    df_trainData_X = pd.DataFrame(columns=X_columns)
    df_trainData_Y_angle = pd.DataFrame(columns=Y_columns)
    df_trainData_Y_moBWHT = pd.DataFrame(columns=Y_columns)
    arr_dg_train = []

    # ===========================
    # PROCESS TRAINING DATA
    # ===========================
    print("\n[1/4] Loading training data...")
    for idx, datum in enumerate(tqdm([listFromFolder[i] for i in idx4train], desc="Train files")):
        df = pd.read_csv(join(NORMALIZED_DIR, datum))
        dfWOMAG = df.loc[:,columnsWOMAG]

        # Standardize sensor arrangement
        if listFromxlsx.loc[idx4train[idx],'side'] == "oaleg":
            targetLegArr = dfWOMAG.loc[:,'oa_shank_ACC_X':'oa_thigh_GYRO_Z']
            nonTargetLegArr = dfWOMAG.loc[:,'non_shank_ACC_X':'non_thigh_GYRO_Z']
            otherArr = dfWOMAG.loc[:,'trunk_ACC_X':'trunk_GYRO_Z']
        else:
            targetLegArr = dfWOMAG.loc[:,'non_shank_ACC_X':'non_thigh_GYRO_Z']
            nonTargetLegArr = dfWOMAG.loc[:,'oa_shank_ACC_X':'oa_thigh_GYRO_Z']
            otherArr = dfWOMAG.loc[:,'trunk_ACC_X':'trunk_GYRO_Z']

        concated = pd.concat([targetLegArr, nonTargetLegArr, otherArr], axis=1)
        concated.columns = X_columns
        df_trainData_X = pd.concat([df_trainData_X, concated], axis=0, ignore_index=True)

        # Output data
        angle = dfWOMAG.loc[:,'ANGLE_X':'ANGLE_Z']
        angle.columns = Y_columns
        df_trainData_Y_angle = pd.concat([df_trainData_Y_angle, angle], axis=0, ignore_index=True)

        moBWHT = dfWOMAG.loc[:,'MOBWHT_X':'MOBWHT_Z']
        moBWHT.columns = Y_columns
        df_trainData_Y_moBWHT = pd.concat([df_trainData_Y_moBWHT, moBWHT], axis=0, ignore_index=True)

        # Extract demographics (age, height, weight, sex)
        patientID = datum.split('_')[2]
        dg_infos = dg_Fromxlsx.loc[dg_Fromxlsx['ID']==patientID][['age','height','weight_bl','sex']].reset_index(drop=True)

        # Convert sex to binary (0=female, 1=male)
        age = dg_infos.loc[0, 'age']
        height = dg_infos.loc[0, 'height']
        weight = dg_infos.loc[0, 'weight_bl']
        sex_binary = sex_to_binary(dg_infos.loc[0, 'sex'])

        dg_list = [age, height, weight, sex_binary]
        arr_dg_train.append(dg_list)

    # ===========================
    # FIT SCALERS (TRAINING ONLY)
    # ===========================
    print("\n[2/4] Fitting scalers on training data...")
    scaler4X = MinMaxScaler()
    scaler4Y_angle = MinMaxScaler()
    scaler4Y_moBWHT = MinMaxScaler()
    scaler4Demographic = StandardScaler()

    scaler4X.fit(df_trainData_X)
    scaler4Y_angle.fit(df_trainData_Y_angle)
    scaler4Y_moBWHT.fit(df_trainData_Y_moBWHT)

    narr_dg_train = np.array(arr_dg_train)
    scaler4Demographic.fit(narr_dg_train)
    print("✓ Scalers fitted")

    # Transform training data
    scaled_X_train = scaler4X.transform(df_trainData_X)
    scaled_Y_angle_train = scaler4Y_angle.transform(df_trainData_Y_angle)
    scaled_Y_moBWHT_train = scaler4Y_moBWHT.transform(df_trainData_Y_moBWHT)
    scaled_DG_train = scaler4Demographic.transform(narr_dg_train)

    # Save scalers
    scalerDir = join(DATA_DIR, r'SAVE_dataSet', DATASET_NAME)
    make_dir(scalerDir)

    dump(scaler4X, open(join(scalerDir, f"{countfold}_fold_scaler4X.pkl"), 'wb'))
    dump(scaler4Y_angle, open(join(scalerDir, f'{countfold}_fold_scaler4Y_angle.pkl'), 'wb'))
    dump(scaler4Y_moBWHT, open(join(scalerDir, f'{countfold}_fold_scaler4Y_moBWHT.pkl'), 'wb'))
    dump(scaler4Demographic, open(join(scalerDir, f'{countfold}_fold_scaler4Demographic.pkl'), 'wb'))
    print("✓ Scalers saved")

    # Reshape training data
    X_train = []
    Y_angle_train = []
    Y_moBWHT_train = []
    for i in range(0, len(idx4train)):
        chopped_X_train = scaled_X_train[i*101:101+i*101,:]
        X_train.append(chopped_X_train.flatten('F').reshape(-1,1))

        chopped_Y_angle_train = scaled_Y_angle_train[i*101:101+i*101,:]
        Y_angle_train.append(chopped_Y_angle_train.flatten('F').reshape(-1,1))

        chopped_Y_moBWHT_train = scaled_Y_moBWHT_train[i*101:101+i*101,:]
        Y_moBWHT_train.append(chopped_Y_moBWHT_train.flatten('F').reshape(-1,1))

    final_X_train = np.array(X_train)
    final_Y_angle_train = np.array(Y_angle_train)
    final_Y_moBWHT_train = np.array(Y_moBWHT_train)
    final_DG_train = scaled_DG_train

    print(f"✓ Training data reshaped: X={final_X_train.shape}, Y_angle={final_Y_angle_train.shape}, Y_moBWHT={final_Y_moBWHT_train.shape}, DG={final_DG_train.shape}")

    # Save training data
    setDir = join(DATA_DIR, r'SAVE_dataSet', DATASET_NAME)
    savez_compressed(join(setDir, f"{countfold}_fold_final_train.npz"),
                     final_X_train=final_X_train,
                     final_Y_angle_train=final_Y_angle_train,
                     final_Y_moBWHT_train=final_Y_moBWHT_train,
                     final_DG_train=final_DG_train)
    print("✓ Training data saved")

    # ===========================
    # PROCESS TEST DATA
    # ===========================
    print("\n[3/4] Loading test data...")
    df_testData_X = pd.DataFrame(columns=X_columns)
    df_testData_Y_angle = pd.DataFrame(columns=Y_columns)
    df_testData_Y_moBWHT = pd.DataFrame(columns=Y_columns)
    arr_dg_test = []

    for idx, datum in enumerate(tqdm([listFromFolder[i] for i in idx4test], desc="Test files")):
        df = pd.read_csv(join(NORMALIZED_DIR, datum))
        dfWOMAG = df.loc[:,columnsWOMAG]

        if listFromxlsx.loc[idx4test[idx],'side'] == "oaleg":
            targetLegArr = dfWOMAG.loc[:,'oa_shank_ACC_X':'oa_thigh_GYRO_Z']
            nonTargetLegArr = dfWOMAG.loc[:,'non_shank_ACC_X':'non_thigh_GYRO_Z']
            otherArr = dfWOMAG.loc[:,'trunk_ACC_X':'trunk_GYRO_Z']
        else:
            targetLegArr = dfWOMAG.loc[:,'non_shank_ACC_X':'non_thigh_GYRO_Z']
            nonTargetLegArr = dfWOMAG.loc[:,'oa_shank_ACC_X':'oa_thigh_GYRO_Z']
            otherArr = dfWOMAG.loc[:,'trunk_ACC_X':'trunk_GYRO_Z']

        concated = pd.concat([targetLegArr, nonTargetLegArr, otherArr], axis=1)
        concated.columns = X_columns
        df_testData_X = pd.concat([df_testData_X, concated], axis=0, ignore_index=True)

        angle = dfWOMAG.loc[:,'ANGLE_X':'ANGLE_Z']
        angle.columns = Y_columns
        df_testData_Y_angle = pd.concat([df_testData_Y_angle, angle], axis=0, ignore_index=True)

        moBWHT = dfWOMAG.loc[:,'MOBWHT_X':'MOBWHT_Z']
        moBWHT.columns = Y_columns
        df_testData_Y_moBWHT = pd.concat([df_testData_Y_moBWHT, moBWHT], axis=0, ignore_index=True)

        # Extract demographics (age, height, weight, sex)
        patientID = datum.split('_')[2]
        dg_infos = dg_Fromxlsx.loc[dg_Fromxlsx['ID']==patientID][['age','height','weight_bl','sex']].reset_index(drop=True)

        # Convert sex to binary (0=female, 1=male)
        age = dg_infos.loc[0, 'age']
        height = dg_infos.loc[0, 'height']
        weight = dg_infos.loc[0, 'weight_bl']
        sex_binary = sex_to_binary(dg_infos.loc[0, 'sex'])

        dg_list = [age, height, weight, sex_binary]
        arr_dg_test.append(dg_list)

    # Transform test data using training scalers
    print("\n[4/4] Applying scalers to test data...")
    scaled_X_test = scaler4X.transform(df_testData_X)
    scaled_Y_angle_test = scaler4Y_angle.transform(df_testData_Y_angle)
    scaled_Y_moBWHT_test = scaler4Y_moBWHT.transform(df_testData_Y_moBWHT)

    narr_dg_test = np.array(arr_dg_test)
    scaled_DG_test = scaler4Demographic.transform(narr_dg_test)

    # Reshape test data
    X_test = []
    Y_angle_test = []
    Y_moBWHT_test = []
    for i in range(0, len(idx4test)):
        chopped_X_test = scaled_X_test[i*101:101+i*101,:]
        X_test.append(chopped_X_test.flatten('F').reshape(-1,1))

        chopped_Y_angle_test = scaled_Y_angle_test[i*101:101+i*101,:]
        Y_angle_test.append(chopped_Y_angle_test.flatten('F').reshape(-1,1))

        chopped_Y_moBWHT_test = scaled_Y_moBWHT_test[i*101:101+i*101,:]
        Y_moBWHT_test.append(chopped_Y_moBWHT_test.flatten('F').reshape(-1,1))

    final_X_test = np.array(X_test)
    final_Y_angle_test = np.array(Y_angle_test)
    final_Y_moBWHT_test = np.array(Y_moBWHT_test)
    final_DG_test = scaled_DG_test

    print(f"✓ Test data reshaped: X={final_X_test.shape}, Y_angle={final_Y_angle_test.shape}, Y_moBWHT={final_Y_moBWHT_test.shape}, DG={final_DG_test.shape}")

    # Save test data
    savez_compressed(join(setDir, f"{countfold}_fold_final_test.npz"),
                     final_X_test=final_X_test,
                     final_Y_angle_test=final_Y_angle_test,
                     final_Y_moBWHT_test=final_Y_moBWHT_test,
                     final_DG_test=final_DG_test)
    print("✓ Test data saved")
    print(f"{'='*80}\n")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("DATASET CREATION WITH DEMOGRAPHICS")
    print("="*80)
    print(f"Dataset name: {DATASET_NAME}")
    print(f"Random seed: {SEED_RAND}")
    print(f"K-folds: 5")
    print("="*80 + "\n")

    # Load all data
    listFromFolder, listFromxlsx, dg_Fromxlsx, arrName = load_data()

    # Setup K-Fold cross-validation
    kfold = KFold(n_splits=5, random_state=SEED_RAND, shuffle=True)

    # Process each fold
    for countfold, (train, test) in enumerate(kfold.split(arrName)):
        print(f"\nFold {countfold}: {len(arrName[train])} train subjects, {len(arrName[test])} test subjects")

        # Convert subject indices to data indices
        idx4train, idx4test = kfold2subfold(arrName, listFromxlsx, train, test)

        # Process this fold
        process_fold(countfold, idx4train, idx4test, listFromFolder, listFromxlsx, dg_Fromxlsx)

    # Final summary
    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE!")
    print("="*80)
    output_dir = join(DATA_DIR, r'SAVE_dataSet', DATASET_NAME)
    print(f"\nOutput location:")
    print(f"  {output_dir}")
    print(f"\nFiles created per fold:")
    print(f"  - {{fold}}_fold_final_train.npz")
    print(f"  - {{fold}}_fold_final_test.npz")
    print(f"  - {{fold}}_fold_scaler4X.pkl")
    print(f"  - {{fold}}_fold_scaler4Y_angle.pkl")
    print(f"  - {{fold}}_fold_scaler4Y_moBWHT.pkl")
    print(f"  - {{fold}}_fold_scaler4Demographic.pkl")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
