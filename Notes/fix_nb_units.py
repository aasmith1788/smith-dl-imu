path = 'C:/Users/aasmi/smith-dl-imu/estimation/notsensor/makeEstimationWithPDF_PyramidAttnCNN_RESIDUAL_v3.ipynb'
with open(path, 'rb') as f:
    content = f.read()

deg = b'\xc2\xb0'
mid = b'\xc2\xb7'

replacements = [
    # naive baseline - training range print (quotes escaped as \" in JSON)
    (
        b'print(f\\"  Training range: {train_range:.2f}' + deg + b' (min {train_min:.2f}' + deg + b', max {train_max:.2f}' + deg + b')\\")',
        b'print(f\\"  Training range: {train_range:.2f} Nm/BW' + mid + b'Ht (min {train_min:.2f}, max {train_max:.2f})\\")'
    ),
    # naive baseline - global rmse print
    (
        b'print(f\\"  Global RMSE: {global_rmse:.3f}' + deg + b'  |  Global nRMSE: {global_nrmse_pct:.2f}% (of training range)\\")',
        b'print(f\\"  Global RMSE: {global_rmse:.3f} Nm/BW' + mid + b'Ht  |  Global nRMSE: {global_nrmse_pct:.2f}% (of training range)\\")'
    ),
    # metrics function - range print per axis
    (
        b'print(f\\"  {ax}: range={rng:.2f}' + deg + b' (min {gmin:.2f}' + deg + b', max {gmax:.2f}' + deg + b')\\")',
        b'print(f\\"  {ax}: range={rng:.2f} Nm/BW' + mid + b'Ht (min {gmin:.2f}, max {gmax:.2f})\\")'
    ),
    # metrics table header
    (
        b'GlobRMSE' + deg + b'  Glob nRMSE%',
        b'GlobRMSE(Nm/BW' + mid + b'Ht)  Glob nRMSE%'
    ),
    # AVG line
    (
        b'global_rmse={macro_rmse:.3f}' + deg,
        b'global_rmse={macro_rmse:.3f} Nm/BW' + mid + b'Ht'
    ),
]

count = 0
for old, new in replacements:
    n = content.count(old)
    content = content.replace(old, new)
    print(f'Replaced {n}x: {old[:70]}')
    count += n

with open(path, 'wb') as f:
    f.write(content)
print(f'Done. {count} total replacements.')
