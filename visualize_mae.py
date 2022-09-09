import argparse
import torch
import models_mae
import webdataset as wds
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor, Normalize
from torchvision.utils import save_image

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_images(img, model):

    # run MAE
    _, y, mask = model(img, mask_ratio=0.75)
    y = model.unpatchify(y)
    # y = torch.einsum('nchw->nhwc', y)

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nchw->nhwc', mask).detach()
    
    # img = torch.einsum('nchw->nhwc', img)

    # masked image
    im_masked = img * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = img * (1 - mask) + y * mask

    print(img.shape, y.shape, im_masked.shape, im_paste.shape)

    return torch.cat([img, im_masked, y, im_paste])

def preprocess(sample):
    return sample[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Visualize MAE')
    parser.add_argument('--model_path', default='/scratch/eo41/mae/models_vitl/say_5fps_vitl16_checkpoint.pth', type=str, help='Model path')
    parser.add_argument('--arch', default='mae_vit_large_patch16', type=str, help='Architecture')
    parser.add_argument('--data_path', default='/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar', type=str, help='data path')
    parser.add_argument('--n_imgs', default=5, type=int, help='number of images')

    args = parser.parse_args()
    print(args)

    # model
    model_mae = prepare_model(args.model_path, args.arch)
    model_mae.eval()
    print('Model loaded.')

    # data
    transform = Compose([Resize(256), RandomCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = (wds.WebDataset(args.data_path, resampled=True).shuffle(10000, initial=10000).decode("pil").to_tuple("jpg").map(preprocess).map(transform))
    data_loader = wds.WebLoader(dataset, shuffle=False, batch_size=args.n_imgs, num_workers=4)

    for it, images in enumerate(data_loader):
        imgs = run_images(images.cuda(), model_mae.cuda())

        if it == 0:
            break

    save_image(imgs, "reconstructions.pdf", nrow=5, padding=1, normalize=True)