"""
Extract windows from preprocessed databases.
"""
import numpy as np
import os


def get_dbs(db_path):
    return [os.path.join(db_path, f) for f in sorted(list(os.listdir(db_path)))
            if os.path.isfile(os.path.join(db_path, f))
            and f.endswith('.npz')]


def extract_windows(clip, window_size, window_step):
    # this is the same as Holden does it
    all_windows = []
    for j in range(0, len(clip) - window_size // 8, window_step):

        # If slice too small pad out by repeating start and end poses
        window = clip[j:j + window_size]

        if len(window) < window_size:
            left = window[:1].repeat((window_size - len(window)) // 2 + (window_size - len(window)) % 2, axis=0)
            left[:, -7:-4] = 0.0
            right = window[-1:].repeat((window_size - len(window)) // 2, axis=0)
            right[:, -7:-4] = 0.0
            window = np.concatenate([left, window, right], axis=0)

        assert len(window) == window_size, 'something went wrong when extracting window of size {}'.format(window_size)
        all_windows.append(window)

    return all_windows


def process_db(db, train_path, valid_path, valid_prob=0.1):
    print('\nprocessing database {} ...'.format(db.split('/')[-1]))

    # load the database
    data = np.load(db)['clips']
    all_windows_train = []
    all_windows_valid = []
    for i, one_clip in enumerate(data):
        # extract windows from the clip
        windows = extract_windows(one_clip, window_size=240, window_step=120)

        # decide if this is validation data at random
        is_validation = bool(np.random.binomial(1, valid_prob))

        # make sure there is at least one validation sample per database
        if i == len(data) - 1 and len(all_windows_valid) == 0:
            # this is the last clip in this db and none was selected previously for validation so select this one for sure
            is_validation = True

        if is_validation:
            all_windows_valid += windows
        else:
            all_windows_train += windows

        print('\r\tprocessed {}/{} {}'.format(i+1, len(data), '*' if is_validation else ''), end='')

    data_train = np.array(all_windows_train)
    data_valid = np.array(all_windows_valid)

    np.savez_compressed(train_path, clips=data_train)
    np.savez_compressed(valid_path, clips=data_valid)

    real_valid = float(data_valid.shape[0]) / float(data_valid.shape[0] + data_train.shape[0]) * 100.00
    print('\n\tfinished, extracted {:.2f}% validation data'.format(real_valid))


if __name__ == '__main__':
    data_path = '../data_preprocessed/raw/'
    train_path = '../data_preprocessed/train/'
    valid_path = '../data_preprocessed/valid/'

    dbs = get_dbs(data_path)

    for db in dbs:
        db_name = db.split('/')[-1]
        process_db(db=db,
                   train_path=os.path.join(train_path, db_name),
                   valid_path=os.path.join(valid_path, db_name))


