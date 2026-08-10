import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import re
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import Compose
from model.util.transform import Resize, NormalizeImage, PrepareForNet
from dataset.vkitti_split import scene_is_selected

def safe_collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.default_collate(batch)

class StatefulRandomCrop(A.RandomCrop):
    def __init__(self, height, width, **kwargs):
        super().__init__(height, width, **kwargs)
        self.last_crop_coords = (0, 0)

    def get_params_dependent_on_data(self, params, data):
        params_dict = super().get_params_dependent_on_data(params, data)
        x_min, y_min, x_max, y_max = params_dict["crop_coords"]
        self.last_crop_coords = (x_min, y_min)
        return params_dict
    
class RandomScale:
    def __init__(self, scale_limit, last_ch):
        self.scale_limit = scale_limit
        self.last_ch = last_ch

    def __call__(self, x):
        scale = np.random.uniform(*self.scale_limit)
        x = cv2.resize(x, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        x[..., -self.last_ch:] = x[..., -self.last_ch:] * scale
        return x
    
class RandomHorizontalFlip:
    def __init__(self, last_ch):
        self.last_ch = last_ch
    def __call__(self, x):
        if np.random.rand() > 0.5:
            x = cv2.flip(x, 1)
            seq_len = self.last_ch // 4
            x[..., -self.last_ch:-self.last_ch+seq_len] = -x[..., -self.last_ch:-self.last_ch+seq_len]
            x[..., -self.last_ch+2*seq_len:-self.last_ch+3*seq_len] = -x[..., -self.last_ch+2*seq_len:-self.last_ch+3*seq_len]
        return x
    
class RandomCropWithInfo(A.DualTransform):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super(RandomCropWithInfo, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.last_crop_coords = None

    def apply(self, img, x_min=0, y_min=0, **params):
        self.last_crop_coords = (x_min, y_min)
        return img[y_min:y_min + self.height, x_min:x_min + self.width]

class DepthVideoDataset(Dataset):
    def __init__(self, mode, data_dirs=[''], crop_size=518, seq_len=4):
        if data_dirs is None:
            data_dirs = ['']
        elif isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.mode = mode
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.tartanair_ratio=1 #30.5W
        self.vkitti_ratio=1
        self.max_depth_outer=200
        self.max_depth_inner = 80
        self.data_paths = []
        self.vkitti_data_paths=[]
        self.tartanair_data_paths=[]
        print(data_dirs)
        for data_dir in data_dirs:
            if 'vkitti' in data_dir:
                print("vkitti (2.0.3)")
                # VKITTI 2.0.3 layout (wrapper-agnostic: works whether tars were
                # extracted with or without a top-level vkitti_2.0.3_* folder):
                #   <...>/<Scene>/<variation>/frames/rgb/Camera_0/rgb_XXXXX.jpg
                #   <...>/<Scene>/<variation>/frames/depth/Camera_0/depth_XXXXX.png
                #   <...>/<Scene>/<variation>/extrinsic.txt
                rgb_suffix = os.path.join('frames', 'rgb', 'Camera_0')
                rgb_cam_dirs = sorted(glob.glob(
                    os.path.join(data_dir, '**', rgb_suffix), recursive=True))
                for rgb_dir in rgb_cam_dirs:
                    if not rgb_dir.endswith(rgb_suffix):
                        continue
                    base = rgb_dir[:-len(rgb_suffix)]            # <...>/<Scene>/<variation>/
                    trimmed = base.rstrip(os.sep)
                    variation = os.path.basename(trimmed)
                    scene = os.path.basename(os.path.dirname(trimmed))
                    if not scene_is_selected(scene, variation, mode):
                        continue

                    depth_dir = os.path.join(base, 'frames', 'depth', 'Camera_0')
                    if not os.path.isdir(depth_dir):
                        cand = glob.glob(os.path.join(data_dir, '**', scene, variation,
                                                      'frames', 'depth', 'Camera_0'),
                                         recursive=True)
                        depth_dir = cand[0] if cand else None
                    extr_file = os.path.join(base, 'extrinsic.txt')
                    if not os.path.isfile(extr_file):
                        cand = glob.glob(os.path.join(data_dir, '**', scene, variation,
                                                      'extrinsic.txt'), recursive=True)
                        extr_file = cand[0] if cand else None
                    if depth_dir is None or extr_file is None:
                        continue

                    pose_by_frame = self._load_vkitti2_extrinsics(extr_file)
                    rgb_by_frame = {}
                    for fn in os.listdir(rgb_dir):
                        if fn.endswith('.jpg') or fn.endswith('.png'):
                            fr = int(os.path.splitext(fn)[0].split('_')[-1])
                            rgb_by_frame[fr] = os.path.join(rgb_dir, fn)
                    depth_by_frame = {}
                    for fn in os.listdir(depth_dir):
                        if fn.endswith('.png'):
                            fr = int(os.path.splitext(fn)[0].split('_')[-1])
                            depth_by_frame[fr] = os.path.join(depth_dir, fn)

                    frames = sorted(set(rgb_by_frame) & set(depth_by_frame) & set(pose_by_frame))
                    if len(frames) < seq_len:
                        continue
                    seq_num = len(frames) - seq_len + 1
                    for i in range(seq_num):
                        set_paths = []
                        for j in range(seq_len):
                            fr = frames[i + j]
                            set_paths.append([rgb_by_frame[fr], depth_by_frame[fr],
                                              pose_by_frame[fr]])
                        self.vkitti_data_paths.append(['vkitti', set_paths])

            if 'tartanair' in data_dir:
                print("tartanair_true")
                scene_paths = sorted(glob.glob(data_dir + '/*/*/*'))
                for scene_path in scene_paths:
                    image_names = sorted([f for f in os.listdir(os.path.join(scene_path, 'image_left')) if f.endswith('.png')])
                    depth_names = sorted([f for f in os.listdir(os.path.join(scene_path, 'depth_left')) if f.endswith('.npy')])
                    pose_path = os.path.join(scene_path, 'pose_left.txt')
                    assert len(image_names) == len(depth_names)
                    image_num = len(image_names)
                    seq_num = image_num - seq_len + 1
                    poses = np.loadtxt(pose_path, delimiter=' ')
                    poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
                    if mode == 'train':
                        start_idx = 0
                        end_idx = round(seq_num )
                    else:
                        start_idx = round(seq_num * 0.9)+1
                        end_idx = seq_num
                    for i in range(start_idx, end_idx):
                        set_paths = []
                        for j in range(seq_len):
                            image_path = os.path.join(scene_path, 'image_left', image_names[i  + j])
                            depth_path = os.path.join(scene_path, 'depth_left', depth_names[i  + j])
                            pose_path=poses[i+j]
                            set_paths.append([image_path, depth_path,pose_path])
                        self.tartanair_data_paths.append(['TartanAir', set_paths])

        self.data_paths = self.vkitti_data_paths * self.vkitti_ratio + self.tartanair_data_paths*self.tartanair_ratio  

        self.scale = {
            'vkitti': RandomScale(scale_limit=(0.8, 0.85), last_ch=4*(seq_len-1)),
            'TartanAir': RandomScale(scale_limit=(0.8, 0.85), last_ch=4*(seq_len-1)),
        }

        self.flip = {
            'vkitti': RandomHorizontalFlip(last_ch=4*(seq_len-1)),
            'TartanAir': RandomHorizontalFlip(last_ch=4*(seq_len-1)),
        }
        
        self.transform = {
        'vkitti': A.Compose([  
        StatefulRandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.ToFloat()
        ]),
        'TartanAir': A.Compose([ 
        StatefulRandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.ToFloat()
        ])
                        }
        
        
        self.transform_infer = Compose([
            Resize(
                width=crop_size,
                height=crop_size,
                resize_target=True ,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] )

    def __getitem__(self, item):
        while True:
            label, set_paths = self.data_paths[item]
            images = []
            images_ori=[]
            depths = []
            masks =[]
            poses = []
            path=[]
            if label in ['vkitti','TartanAir']:
                for image_path, depth_path,pose in set_paths:
                    image = cv2.imread(image_path).astype(np.float32) / 255
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_ori = (image - 0.5) * 2
                    if label == 'vkitti':
                        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/100
                        depth = np.expand_dims(depth, axis=-1)
                        depth = depth.astype(np.float32)
                        depth[depth >= 655] = 0.0  # VKITTI2 sky/invalid sentinel 65535 -> mark invalid
                        mask = depth > 0
                        depth[depth > self.max_depth_outer] = self.max_depth_outer
                        pose = np.asarray(pose, dtype=np.float32)  # 4x4 world->camera (camera 0)
                    if label == 'TartanAir':
                        depth = np.load(depth_path).astype(np.float32)[..., None]
                        mask=depth >0
                        depth[depth > self.max_depth_outer]=self.max_depth_outer
                        q = pose[3:]
                        t = pose[:3]
                        rotation = R.from_quat(q)
                        R_matrix = rotation.as_matrix()
                        T = np.eye(4)
                        T[:3, :3] = R_matrix
                        T[:3, 3] = t 
                        pose = T.astype(np.float32)  
                        pose=np.linalg.inv(pose)                        
                    path.append(depth_path)
                    poses.append(pose) 
                    sample = self.transform_infer({'image': image, 'depth': depth,'mask':mask,'image_ori':image_ori})
                    sample["image"]=np.transpose(sample["image"], (1, 2, 0))
                    images.append(sample["image"])
                    depths.append(np.expand_dims(sample['depth'], axis=-1))
                    masks.append(np.expand_dims(sample['mask'], axis=-1))
                    sample["image_ori"]=np.transpose(sample["image_ori"], (1, 2, 0))
                    images_ori.append(image)

            images = np.concatenate(images, axis=-1)  # H, W, 3T 
            images_ori = np.concatenate(images_ori, axis=-1)
            depths = np.concatenate(depths, axis=-1)  # H, W, T 
            masks = np.concatenate(masks, axis=-1)
            H,W,_=images_ori.shape
            h_new,w_new,_=images.shape
            factor=h_new/H
            all = np.concatenate((images,depths,masks), axis=-1)
            all = self.transform[label](image=all)['image']
            crop_transform = self.transform[label].transforms[0]
            left_margin, top_margin = crop_transform.last_crop_coords
            start = 0; end = 3 * self.seq_len; images = all[..., start:end]  # H, W, 3T
            start = end; end = start+self.seq_len; depths = all[..., start:end]  # H, W, T
            start = end; end = start + self.seq_len; masks = all[..., start:end]
            images = np.stack(np.split(images, self.seq_len, axis=-1), axis=0)#T H W 3
            depths = np.stack(np.split(depths, self.seq_len, axis=-1), axis=0)
            masks = np.stack(np.split(masks, self.seq_len, axis=-1), axis=0)
            images = torch.from_numpy(images).permute(0, 3, 1, 2)#T 3 H W
            depths = torch.from_numpy(depths).permute(0, 3, 1, 2)
            masks = torch.from_numpy(masks).permute(0, 3, 1, 2)
            if  label in ['vkitti','TartanAir']:
                inputs={}
                if label == 'TartanAir':
                    fx=320
                    fy=320
                    cx=320
                    cy=240
                if label == 'vkitti':
                    fx=725
                    fy=725
                    cx=620.5
                    cy=187        
                IntM = np.zeros((3, 3))
                IntM[2, 2] = 1.
                IntM[0, 0] = fx*factor
                IntM[1, 1] = fy*factor
                IntM[0, 2] = cx*factor-left_margin
                IntM[1, 2] = cy*factor-top_margin
                IntM = IntM.astype(np.float32)
                inputs = self.get_K(IntM, inputs)
                inv_K=inputs[('inv_K_pool', 0)]    
            sample = {
                'image': images,
                'depth': depths,
                'mask':masks,
                'label': label,
                'inv_K':inv_K,
                'poses':poses,
                'IntM':IntM,
                'path':path
            }
            return sample

    def __len__(self):
        return len(self.data_paths) // 4 * 4

    def _load_vkitti2_extrinsics(self, path):
        # VKITTI 2.0.3 extrinsic.txt:
        #   line 0 = header, then rows: frame cameraID <matrix values ...>
        # Returns {frame: 4x4 world->camera float32} for camera 0 only.
        pose_by_frame = {}
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            frame = int(float(parts[0]))
            cam = int(float(parts[1]))
            if cam != 0:
                continue
            vals = list(map(float, parts[2:]))
            if len(vals) >= 16:
                M = np.array(vals[:16], dtype=np.float32).reshape(4, 4)
            else:
                M = np.eye(4, dtype=np.float32)
                M[:3, :4] = np.array(vals[:12], dtype=np.float32).reshape(3, 4)
            pose_by_frame[frame] = M.astype(np.float32)
        return pose_by_frame

    def get_K(self, K, inputs):
        inv_K = np.linalg.inv(K)
        K_pool = {}
        ho, wo = self.crop_size, self.crop_size
        for i in range(6):
            K_pool[(ho // 2**i, wo // 2**i)] = K.copy().astype('float32')
            K_pool[(ho // 2**i, wo // 2**i)][:2, :] /= 2**i

        inputs['K_pool'] = K_pool

        inputs[("inv_K_pool", 0)] = {}
        for k, v in K_pool.items():
            K44 = np.eye(4)
            K44[:3, :3] = v
            inputs[("inv_K_pool", 0)][k] = np.linalg.inv(K44).astype('float32')

        inputs[("inv_K", 0)] = torch.from_numpy(inv_K.astype('float32'))

        inputs[("K", 0)] = torch.from_numpy(K.astype('float32'))
    
        return inputs
    
    
if __name__ == '__main__':
    dataset = DepthVideoDataset('train',
                                data_dirs=[""],
                                seq_len=32)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4,shuffle=True,pin_memory=True)
    with torch.no_grad():
        for i, sample_batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            path=sample_batch['path']
            images = sample_batch['image'].cuda()
            depths = sample_batch['depth'].cuda()
            masks = sample_batch['mask'].cuda()
            inv_K = sample_batch['inv_K']
            poses = sample_batch['poses']
        
        