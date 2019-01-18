import numpy as np
import os

def prepare_data_2d_from_detectron(detectron_data, output_file_2d, output_file_3d):
    keypoints_2d = import_detectron_poses(detectron_data)
    dataset_2d, dataset_3d = keypoints_2d_to_datasets(keypoints_2d)
    np.savez(output_file_2d, **dataset_2d)
    np.savez(output_file_3d, **dataset_3d)






def keypoints_2d_to_datasets(poses, subject='S1', action='Default'):
    """
    given a sequence of 2d keypoints for a single subject and action, 
    create the datasets necessary to predict 3d keypoints with Video2Pose run.py.
    """
    
    dataset_2d = dict({
        'positions_2d': dict({
            subject: dict({
                action: [poses] # array of cameras but we have only one camera
            })
        }),
        'metadata': dict({
            'layout_name': 'h36m', 
            'num_joints': 17, 
            'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
        })
    })
    
    # to use run.py to generate 3d predictions, 
    # it requires a 3d dataset that matches 
    # all the subjects and actions in the 2d dataset
    dataset_3d_fake = dict({
        'positions_3d': dict({
            subject: dict({
                action: np.ones((poses.shape[0], 32, 3), dtype=np.float32)
            })
        })
    })
    
    return dataset_2d, dataset_3d_fake












def import_detectron_poses(path):
    """
    given an npz archive of detectron keypoints and boxes,
    extract the keypoints for the highest-confidence person

    Args:
        path - file path to an npz file generated from Detectron
            and containing keypoints and boxes elements,
            the value of each being stacked video-frame results
            from detectron--cls_keyps, and cls_boxes respectively.
            Since the Detectron results for each frane
            include results for all detecron classes, 
            the import process here pulls just the 'person' class (1).
            Similarly, since the Detectron box results include
            multiple ROI proposals (regions of interest for possible persons),
            the import process takes only the highest-confidence person
            for each frame. NOTE: this is not a robust approach, since
            there genuinely could be multiple persons in video frames,
            and we should really figure out a way to identify and then
            track a specific person.
    """
    # Latin1 encoding because Detectron runs on Python 2.7
    data = np.load(path, encoding='latin1')
    kp = data['keypoints']
    bb = data['boxes']
    results = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0:
            assert i > 0
            # Use last pose in case of detection failure
            results.append(results[-1])
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        keypoints = kp[i][1][best_match].T.copy()
        results.append(keypoints)
    results = np.array(results)
    # return results[:, :, 4:6] # Soft-argmax
    return results[:, :, [0, 1, 3]]


def extract_data(npz_file):
    """
    given a data archive or (PathLike to a data archive)
    ensure the archive is loaded and then returns 
    the two top-level components.
    
    This function exists mainly to document
    the structure of .npz data archives included with VideoPose3D in the data folder
    and simplify loading/access.
    
    Args:
        npz_file (PathLike|numpy.lib.npyio.NpzFile): NpzFile or path to NpzFile

    Returns:
        positions_2d Dictionary of Subject => Action => [camera, frame, keypoint, dim]
        metadata Dictionary of metadata about positions_2d
    """
    
    if npz_file is not np.lib.npyio.NpzFile:
        npz_file = np.load(npz_file)
        
    
    positions_2d = npz_file['positions_2d'].item()
    metadata = npz_file['metadata'].item()
    
    return positions_2d, metadata



    
def get_poses(positions_2d, subject='S1', action='Default', camera=0):
    """
    given the positions_2d dictionary from a VideoPose3D data archive @see extract_data
    and a subject, action and camera index, 
    returns the numpy ndarray of poses with shape
    [n_frames, n_key_points, n_dimensions (e.g. 2 for x, y)]
    
    Args:
        positions_2d Dictionary of Subject => Action => [camera, frame, keypoint, dim]
        subject The subject to use, default is 'S1'
        action The action to use, default is 'Default' but common real example is 'Walking 1'
        camera The camera to use, default is 0

    Returns:
        numpy.ndarray of poses with shape [n_frames, n_keypoints_per_frame, n_dims_per_keypoint]
    
    Example:
        data_file = np.load('data_2d_h36m_detectron_ft_h36m.npz')
        positions_2d, = extract_data(data_file)
        poses = get_poses(positions_2d, action='Walking 1')
        print(poses.shape) # [3000, 17, 2] 3000 frames, 17 keypoints/frame, 2 dims/keypoint (x,y)
    """
    
    
    if not isinstance(positions_2d, dict):
        print(f'expected positions_2d dict, encountered {type(positions_2d)}')
        return None

    subject_data = positions_2d[subject]
    
    if not isinstance(subject_data, dict):
        print(f'no subject {subject} found in data')
        return None
    
    action_data = subject_data[action]
    
    if not isinstance(action_data, list):
        print(f'no action {action} found for {subject} in data')
        return None
    
    if len(action_data) < camera + 1:
        print(f'no camera {camera} found for action {action} and {subject} in data')
        return None
    
    poses = action_data[camera]
    
    return poses
    

def poses_2_archive(poses, subject='S1', action='Default'):
    """
    given a sequence of poses, create an archive
    in the format required by VideoPose3D.
    """
    
    s = dict()
    s[action] = [poses] # array of cameras but we have only one camera
    positions_2d = dict()
    positions_2d[subject] = s
    
    metadata = dict({
        'layout_name': 'h36m', 
        'num_joints': 17, 
        'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]]
    })
    
    dataset_2d = dict({
        'positions_2d': positions_2d,
        'metadata': metadata
    })
    
    return dataset_2d