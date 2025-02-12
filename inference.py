from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import argparse
import copy
import math


parser = argparse.ArgumentParser()
parser.add_argument('--no_ddim', help='use DDPM', action = 'store_true')
parser.add_argument('--ddim_use_original_steps', help='use DDIM original steps', action = 'store_true')
parser.add_argument('--ddim_step', help='DDIM step', default = 200, type = int)
parser.add_argument('--external', help='External', action = 'store_true')
parser.add_argument('--eta', help='eta of ddim', default = 0., type = float)
args = parser.parse_args()
eta, ddim_step, ddim, ddim_use_original_steps = args.eta, args.ddim_step, (not args.no_ddim), args.ddim_use_original_steps

external = args.external

print(ddim_use_original_steps)


# config_name = training_path.split('_')[0]
# config_name = ''

# config = OmegaConf.load(f'logs/{training_path}/configs/-project.yaml')
# ckpt = glob.glob(f'logs/{training_path}/checkpoints/last.ckpt')
# ckpt.sort()
config = OmegaConf.load('configs/inference.yaml')
# ckpt = 'weights/stage2.ckpt'
# config['model']['params']['ckpt_path'] = ckpt
# config['data']['params']['batch_size'] = 1


condition = config['model']['params']['cond_stage_config']['params']['args']['conditioning']
data_loader = instantiate_from_config(config['data'])
data_loader.prepare_data()
data_loader.setup()
device = torch.device('cuda:0')
valid_dataset = data_loader.datasets['test'] if external else data_loader.datasets['validation']
print('dataset', len(valid_dataset))
print('model', config['model']['params']['ckpt_path'])
model = instantiate_from_config(config['model']).to(device)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
model.eval()

first_model = 'vq' if 'VQ' in config['model']['params']['first_stage_config']['target'] else 'kl'
attn_resolution = ''.join([str(attn_resol) + '_' for attn_resol in config['model']['params']['first_stage_config']['params']['ddconfig']['attn_resolutions']])
conditioning = ''.join([str(attn_resol) + '_' for attn_resol in config['model']['params']['cond_stage_config']['params']['args']['conditioning']])
downsampling = str(2 ** config['model']['params']['cond_stage_config']['params']['args']['n_stages'])


sampling_config = f'ddimstep_{ddim_step}_eta_{eta}' if ddim else 'ddpm'
save_path = f'experiment_interpolation/{sampling_config}'
os.makedirs(save_path, exist_ok = True)
condition.append('movement')

g = 2.2

with torch.no_grad():
    model.eval()
    for i, data in enumerate(valid_dataloader):
        data['image'] = data['image'].to(device)
        
        for cond in condition:
            data['landmarkwithpre'][cond] = data['landmarkwithpre'][cond].to(device)
        
        path = data['landmarkwithpre']['path'][0][0].split('/')[-2]
        if 'SNU' in path:
            data['image'] = ((data['image']+1)/2 ** (1 / g)) * 255
            data['image'] = data['image']
            data['image'] = data['image'] / 255
            data['image'] = data['image'] * 2 - 1
            
            data['landmarkwithpre']['pre'] = ((data['landmarkwithpre']['pre']+1)/2 ** (1 / g)) * 255
            data['landmarkwithpre']['pre'] = data['landmarkwithpre']['pre']
            data['landmarkwithpre']['pre'] = data['landmarkwithpre']['pre'] / 255
            data['landmarkwithpre']['pre'] = data['landmarkwithpre']['pre'] * 2 - 1            

        surgical_movment_prediction = data['landmarkwithpre']['movement']
        
        unconditional_conditioning = copy.deepcopy(data)
        unconditional_conditioning['landmarkwithpre']['movement'] = torch.zeros_like(unconditional_conditioning['landmarkwithpre']['movement'])
        _, unconditional_conditioning, _, _, _ = model.get_input(unconditional_conditioning, model.first_stage_key,return_first_stage_outputs=True,force_c_encode=True,return_original_cond=True,bs=1)
        
        print(path)
        for scale in [1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6]:

            z, c, x, xrec, xc = model.get_input(data, model.first_stage_key,return_first_stage_outputs=True,force_c_encode=True,return_original_cond=True,bs=1)
            with model.ema_scope("Plotting"):
                samples, z_denoise_row = model.sample_log(cond=c,batch_size=1,ddim=True,ddim_steps=args.ddim_step,eta=0.,quantize_denoised=False, unconditional_conditioning = unconditional_conditioning, unconditional_guidance_scale = scale)

            x_samples = model.decode_first_stage(samples)

            x = x[0][0].detach().cpu().numpy()
            x_samples = x_samples[0][0].detach().cpu().numpy()
            pre = data['landmarkwithpre']['pre'][0][:,:,0].detach().cpu().numpy()
            os.makedirs(f'{save_path}/FAKE', exist_ok=True)
            os.makedirs(f'{save_path}/FAKE/{path}', exist_ok=True)
            os.makedirs(f'{save_path}/FAKE/{path}/scale', exist_ok=True)


            array = np.concatenate([x, x_samples, pre], 1)
            array = np.stack([array, array, array], 2)
            array = np.clip(array, -1, 1)
            array = ((array + 1)/2)* 255
            array = array.astype(np.uint8)
            
            font_size = 80
            font = ImageFont.truetype("arial.ttf", size=font_size)
            
            img = Image.fromarray(array)
            draw = ImageDraw.Draw(img)
            draw.text((1024 * 0 + 10, 10), 'Post-Ceph', (255,0,0), font = font) 
            draw.text((1024 * 1 + 10, 10), 'Syn Post-Ceph', (0,255,0), font = font) 
            draw.text((1024 * 2 + 10, 10), 'Pre-Ceph', (0,0,255), font = font) 
            if scale == 1.0:
                img.save(f'{save_path}/{path}.png')
            Image.fromarray(((np.clip(x_samples, -1, 1) + 1)/2 * 255).astype(np.uint8)).save(f'{save_path}/FAKE/{path}/scale/scale_{path}_FAKE_{scale}.png')
