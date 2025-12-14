import os
from glob import glob

import numpy as np
from tqdm.auto import tqdm


def csc_to_dense(data, indices, indptr, shape):
    """
    CSC (Compressed Sparse Column) → dense (T, 128)
    shape: (time_steps, 128)
    """
    n_rows, n_cols = int(shape[0]), int(shape[1])
    dense = np.zeros((n_rows, n_cols), dtype=np.uint8)

    for col in range(n_cols):
        start = indptr[col]
        end = indptr[col + 1]
        rows = indices[start:end]
        # velocity > 0이면 1로 두자 (0/1만 필요하니까)
        dense[rows, col] = 1

    return dense


def load_lpd5_pianoroll_csc(path):
    """
    LPD-5 Cleansed npz 1개 → (T, 128, C_tracks) uint8 (0/1)
    """
    d = np.load(path, allow_pickle=True)

    # 어떤 pianoroll_X 들이 있는지 찾기
    track_ids = []
    for k in d.files:
        if k.startswith("pianoroll_") and k.endswith("_csc_shape"):
            mid = k.split("_")[1]        # "pianoroll_0_csc_shape" → "0"
            track_ids.append(int(mid))

    if len(track_ids) == 0:
        raise ValueError("No pianoroll_X_csc_* keys found.")

    track_ids = sorted(set(track_ids))

    # 전체 T 길이 추정
    if "downbeat" in d.files:
        T = d["downbeat"].shape[0]
    elif "tempo" in d.files:
        T = d["tempo"].shape[0]
    else:
        shape0 = d[f"pianoroll_{track_ids[0]}_csc_shape"]
        T = int(shape0[0])

    P = 128
    track_arrays = []

    for tid in track_ids:
        shape = d[f"pianoroll_{tid}_csc_shape"]  # (T_track, 128)
        T_track, P_track = int(shape[0]), int(shape[1])

        if P_track != 128:
            raise ValueError(f"Unexpected pitch dim {P_track} in {os.path.basename(path)}")

        data_arr = d[f"pianoroll_{tid}_csc_data"]
        indices_arr = d[f"pianoroll_{tid}_csc_indices"]
        indptr_arr = d[f"pianoroll_{tid}_csc_indptr"]

        if T_track == 0 or data_arr.size == 0:
            dense = np.zeros((T, 128), dtype=np.uint8)
        else:
            dense = csc_to_dense(data_arr, indices_arr, indptr_arr, shape)  # (T_track,128)

            # 전체 T에 맞춰 패딩/자르기
            if T_track < T:
                pad = np.zeros((T, 128), dtype=np.uint8)
                pad[:T_track] = dense
                dense = pad
            elif T_track > T:
                dense = dense[:T]

        track_arrays.append(dense)

    pr = np.stack(track_arrays, axis=-1)  # (T,128,C_tracks), uint8 (0/1)
    return pr


def preprocess_lpd5_to_dense(raw_dir, out_dir, max_files=None):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob(os.path.join(raw_dir, "*.npz")))
    if max_files is not None:
        files = files[:max_files]

    print(f"[Preprocess] Found {len(files)} raw npz files.")
    print(f"[Preprocess] Output dir: {out_dir}")

    skipped = 0
    for path in tqdm(files, desc="Preprocess LPD-5 CSC → dense"):
        try:
            pr_bin = load_lpd5_pianoroll_csc(path)  # (T,128,C)
            base = os.path.basename(path)
            out_path = os.path.join(out_dir, base)

            # uint8 0/1 상태로 저장 (압축 npz)
            np.savez_compressed(out_path, pr_bin=pr_bin)
        except Exception as e:
            print(f"[WARN] Skip {os.path.basename(path)} due to error: {e}")
            skipped += 1
            continue

    print(f"[Preprocess] Done. Skipped {skipped} files.")


if __name__ == "__main__":
    raw_dir = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\lpd_5\npz"        # 원본 LPD-5 Cleansed (CSC)
    out_dir = r"C:\Users\hyi8402\Desktop\Sound to Sheet\Dataset\lpd_5\dense_npz"  # 캐싱할 폴더

    preprocess_lpd5_to_dense(raw_dir, out_dir, max_files=None)
