# -*- coding: utf-8 -*-
import os
from copy import deepcopy
from typing import Dict, Union, List, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as pt
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, Trainer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from .noisy_dataset import SimpleNoisy
from .pytorch_ssim import ssim as eval_ssim


def arr_dict_mean(dicts: List[Dict], suffix=""):
    """
    Recursively computes the mean across elements of a list of (optionally nested) dictionaries.

    :param dicts: List[Dict] to operate n
    :param suffix: optional suffix to append to the computed values names
    :return: (nested) dictionary containing mean
    """
    means = {}
    d0 = dicts[0]
    keys = d0.keys()
    for k in keys:
        if type(d0[k]) == dict:
            means[k] = arr_dict_mean([d[k] for d in dicts], suffix)
        else:
            km = f"{k}{suffix}"
            means[km] = torch.stack([d[k] for d in dicts]).mean(0)
    return means
import abc

def mean_psnr(im, ref):
    """Computes and returns the mean psnr on a batch

    :param im: (tensor, shape (B x 1 x) H x W): estimated image
    :param ref: (tensor, shape (B x 1 x) H x W): reference image
    :return: Average PSNR of the batch
    """
    psnr_tot = 0
    assert im.shape == ref.shape

    if len(im.shape) == 2:
        im = im.unsqueeze(0)
        ref = ref.unsqueeze(0)
    max_list = ref.view(im.shape[0], -1).max(1)[0]
    for idx, (i, r) in enumerate(zip(im, ref)):
        psnr_tot += 10 * torch.log10(max_list[idx] ** 2 / F.mse_loss(i, r))
    return psnr_tot / im.shape[0]


def grad_penalty_gp(D, x_true, x_generated, observed, lp=False, device="cuda"):
    """
    Gradient penalty as prescribed in  I. Gulrajani et al.  "Improved training of wasserstein GANs."
    NIPS 2017.
    :param D: Model of the discriminator
    :param x_true: concatenation of [G(z1,y),x] or [x,G(z2,y)] w.p. 0.5 if 'ao' else x
    :param x_generated: concatenation [G(z1,y), G(z2,y)] if 'ao' else G(z,y)
    :param observed: observation y
    :param device: 'cuda' or 'cpu'
    :return: the gradient penalty
    """
    epsilon = torch.rand(x_true.shape[0], 1, 1, 1).to(device)

    x_hat = epsilon * x_true + (1 - epsilon) * x_generated

    x_hat = x_hat.to(device)
    x_hat.requires_grad_()
    d_hat = D(torch.cat((x_hat, observed), 1))
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones(d_hat.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    img_N = x_true.shape[-1]
    gradients = gradients.reshape(-1, img_N, img_N)
    eps = torch.finfo(gradients.dtype).eps
    gradients = gradients + eps
    ddx = torch.sqrt(torch.sum(gradients ** 2, dim=(1, 2)))
    if lp:
        # WGAN-LP, as prescripted in Petzka, Fischer and Lukovnikov. "On the regularization of WGANs."
        gradient_penalty = torch.mean(torch.max((ddx - 1.0), torch.ones_like(ddx)) ** 2)
    else:
        # WGAN-GP, as prescribed in  I. Gulrajani et al.  "Improved training of wasserstein GANs."
        gradient_penalty = torch.mean((ddx - 1.0) ** 2)
    return gradient_penalty

#  the noise model for zeiss

def constant_lr(epoch):
    return 1



class cWGAN(LightningModule):
    """
    Basic cWGAN framework using  the Adler-Oktem loss

    https://arxiv.org/abs/1811.05910

    TTUR

    https://arxiv.org/abs/1706.08500

    and optionally the LP loss

    https://arxiv.org/abs/1709.08894
    """

    def __init__(self, hparams: Dict=None):
        if hparams is None:
           hparams=type(self).default_hparams()
        super().__init__()
        self.hparams = hparams
        assert "discriminator_args" in hparams
        assert "generator_args" in hparams
        for mode in ["train","val","test"]:
            assert f"{mode}_loader_args" in hparams
        self.discriminator = self.make_discriminator(hparams.get("discriminator_args"))
        self.generator = self.make_generator(hparams.get("generator_args"))

    @abc.abstractmethod
    def make_discriminator(self,discriminator_args:Dict):
        pass
    @abc.abstractmethod
    def make_generator(self,generator_args:Dict):
        pass
    @classmethod
    def default_hparams(cls):
        hparams=dict(
            l_drift=0.001, l_grad=10, lp=False,
            G_lr=1e-4,D_freq=1,#TTUR by default
            D_lr=3e-4,G_freq=1,
            opt_sched=dict(step_size=50,gamma=0.5),
            N_eval_samples=5# samples used during validation
        )
        return hparams


    def save_checkpoint(self, path):
        """
        Helper to save a checkpoint directly from the model.

        :param path:
        :return:
        """
        save_model = type(self)(self.hparams)
        save_model.load_state_dict(self.state_dict(destination="cpu"))

        trainer = Trainer(
            max_epochs=0, max_steps=0, num_sanity_val_steps=0, weights_summary=None
        )
        trainer.fit(save_model)
        trainer.save_checkpoint(path)


    def discriminator_train_forward(self, obs, ground_truth ):
        """
        Return the discriminator loss, including penalites
        :param real:
        :param rec:
        :param mask:
        :return: loss:pt.Tensor, log:Dict,progress_bar:Dict
        """
        x_posterior_1 = self.forward(obs)
        x_posterior_2 = self.forward(obs)
        x_generated = torch.cat([x_posterior_1, x_posterior_2], 1)  # DO NOT DETACH HERE
        # Channel shuffle
        if np.random.random() > 0.5:
            x_true = torch.cat([x_posterior_1, ground_truth], 1)
        else:
            x_true = torch.cat([ground_truth, x_posterior_2], 1)
        # Evaluate the discriminator
        x_generated = x_generated.detach()
        x_true = x_true.detach()
        disc_in_true = torch.cat([x_true, obs], 1)
        d_true = self.discriminator(disc_in_true)
        disc_in_gen = torch.cat([x_generated, obs], 1)
        d_generated = self.discriminator(disc_in_gen)
        real_score = torch.mean(d_true)
        fake_score = torch.mean(d_generated)
        W1 = real_score - fake_score
        drift = self.hparams["l_drift"] * torch.mean(d_true ** 2)
        grad_pen = self.hparams["l_grad"] * grad_penalty_gp(
            self.discriminator,
            x_true,
            x_generated,
            obs,
            lp=self.hparams["lp"],
            device=d_generated.device,
        )
        d_total_loss = -W1 + drift + grad_pen
        d_list = [W1, drift, grad_pen]
        d_list = [d.detach().item() for d in d_list]
        log = dict(
            W1=W1,
            penalty=grad_pen,
            disc_loss=d_total_loss,
            fake_score=fake_score,
            real_score=real_score,
            drift_pen=drift,
        )
        progress_bar = dict(disc_loss=d_total_loss, W1=W1, penalty=grad_pen)
        result = d_total_loss, log, progress_bar
        return result

    def generator_train_forward(self, obs,ground_truth):
        """
        Return the generator loss, including penalites
        :param ground_truth:
        :param rec:
        :param mask:
        :return: loss:pt.Tensor, log:Dict,progress_bar:Dict
        """
        x_posterior_1 = self.forward(obs)
        x_posterior_2 = self.forward(obs)
        #
        x_generated = torch.cat([x_posterior_1, x_posterior_2], 1)
        # Channel shuffle
        disc_in_gen = torch.cat([x_generated, obs], 1)
        d_generated = self.discriminator(disc_in_gen)
        gen_loss = -d_generated.mean()
        log = dict(gen_loss=gen_loss)
        progress_bar = dict()
        result = gen_loss, log, progress_bar
        return result

    def generator_val_forward(self, ground_truth, obs,n_eval_samples=1, prefix="val"):
        """
        Validation step to correct metrics, used inside validation and test step
        :param ground_truth:
        :param obs:
        :param mask:
        :return:
        """
        recs = []
        for i in range(n_eval_samples):
            rec = self.forward(obs)
            recs.append(rec)
        # return to what we were doing before

        recs = pt.stack(recs)
        rec_mu = recs.mean(dim=0)
        rec_std = recs.std(dim=0)

        val_loss = pt.tensor(0.0)  # pt.zeros((), device=self.device)
        log = dict()  # log: gets logged to tensorboard/other loggers
        progress_bar = dict()  # gets displayed in progrss bar during validation
        mse_us = (obs - ground_truth).pow(2).mean()
        psnr_us = mean_psnr(obs, ground_truth)
        ssim_us = eval_ssim(obs, ground_truth)

        metrics = dict(
            mse=(rec_mu - ground_truth).pow(2).mean(),
            psnr=mean_psnr(rec_mu, ground_truth),
            ssim=eval_ssim(rec_mu, ground_truth),
            rec_std=rec_std.mean(),
            mse_us=mse_us,
            psnr_us=psnr_us,
            psnr_diff=mean_psnr(rec_mu, ground_truth) - psnr_us,
            ssim_us=ssim_us,
        )
        log.update(**{f"{prefix}_{k}": v for k, v in metrics.items()})
        progress_bar.update(**{f"{prefix}_{k}": v for k, v in metrics.items()})

        return val_loss, log, progress_bar

    def forward(self, X, noise=None):
        """
        Expect X to be in NCHW format, if n_noise_channels_added > 0, we add some guassian noise along the channel dim
        :param X:
        :return:
        """
        N_NOISE_CHANNELS=1
        if noise is None:
                N,_C,H,W=X.shape
                Z = pt.rand(
                    N,
                    N_NOISE_CHANNELS,
                    H,
                    W,
                    device=X.device,
                    dtype=X.dtype,
                )
                X = pt.cat([X, Z], 1)
        else:
            X = pt.cat([X, noise], 1)
        return self.generator(X)

    def training_step(self, batch, batch_idx, optimizer_idx):
        obs, ground_truth = batch

        if optimizer_idx == 0:  # Discriminator
            loss, log, progress_bar = self.discriminator_train_forward(obs,ground_truth)
        elif optimizer_idx == 1:  # Generator
            loss, log, progress_bar = self.generator_train_forward(obs,ground_truth)
        else:
            NotImplemented("Expected 1 Disc and 1 Gen")

        return dict(loss=loss, log=log, progress_bar=progress_bar)

    def training_epoch_end(self, output):
        return output[0] if hasattr(output,"__len__") and  len(output)>0  else output

    def validation_step(self, batch, batch_idx):
        obs, ground_truth = batch
        N_SAMPLES=self.hparams.get("N_eval_samples")
        loss, log, progress_bar = self.generator_val_forward(obs, ground_truth,n_eval_samples=N_SAMPLES, prefix="val")
        return dict(loss=loss, log=log, progress_bar=progress_bar)

    def validation_epoch_end(self, outputs):
        return arr_dict_mean(outputs, "_ep_mu")

    def test_epoch_end(self, outputs):
        return arr_dict_mean(outputs, "_ep_mu")

    def test_step(self, batch):
        obs, ground_truth = batch
        loss, log, progress_bar = self.generator_val_forward(
            obs, ground_truth, prefix="test"
        )
        return dict(loss=loss, log=log, progress_bar=progress_bar)

    def configure_optimizers(self):
        gen_opt = pt.optim.Adam(self.generator.parameters() )

        disc_opt = pt.optim.Adam(self.discriminator.parameters())

        opts = (
            {"optimizer": disc_opt, "frequency": 1},
            {"optimizer": gen_opt, "frequency": 5},
        )

        # -1 because we want to have the generator updated every nth time, so we update the discriminator n-1 times.
        dis_scheduler = StepLR(disc_opt,**self.hparams["opt_sched"])
        opts[0].update(lr_scheduler=dis_scheduler)

        gen_scheduler = StepLR(gen_opt,**self.hparams["opt_sched"] )
        opts[1].update(lr_scheduler=gen_scheduler)

        return opts

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, **self.hparams["train_loader_args"])

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_set, **self.hparams["test_loader_args"])

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_set, **self.hparams["val_loader_args"])


from unet import  UNet


class ScoreWrapper(pt.nn.Module):
    """
    Turns a segementation or reconstruction network like unet into
    a WGAN discriminator
    """

    def __init__(self,model) -> None:
        super().__init__()
        self.inner=model

    def forward(self, *input: Any, **kwargs: Any):
        out=self.inner(*input,**kwargs)
        out=out.flatten().logsumexp(-1) #interpolation between max and mean pool
        return out


class Unet_cWGAN(cWGAN):
    """
    Simple cWGAN stub using https://github.com/fepegar/unet
    as reconstructor and discriminator
    """

    def __init__(self, hparams: Dict = None):
        super().__init__(hparams)

    def make_discriminator(self, discriminator_args: Dict):
        return ScoreWrapper(UNet(**discriminator_args))

    def make_generator(self, generator_args: Dict):
        return UNet(**generator_args)

    @classmethod
    def default_hparams(cls):
        hparams= super().default_hparams()
        return hparams

class TwinUnetMIST(cWGAN):
    """
    Simple cWGAN stub using https://github.com/fepegar/unet
    as reconstructor and discriminator
    """

    def __init__(self, hparams: Dict = None):
        super().__init__(hparams)

    def make_discriminator(self, discriminator_args: Dict):
        return ScoreWrapper(UNet(**discriminator_args))

    def make_generator(self, generator_args: Dict):
        return UNet(**generator_args)

    @classmethod
    def default_hparams(cls):
        hparams= super().default_hparams()
        hparams.update(generator_args=dict(
            in_channels= 2,
            out_classes= 1,
            dimensions= 2,
            num_encoding_blocks= 2,# each downscales by factor 2
            out_channels_first_layer= 64,
            normalization= "instance",
            pooling_type= 'max',
            upsampling_type= 'conv',
            preactivation= True,
            residual= True,
            activation="ReLU",
            padding= 0,
            padding_mode= 'zeros',
            dropout= 0,

        ))
        hparams.update(discriminator_args=dict(
            in_channels=3,
            out_classes=1,
            dimensions=2,
            num_encoding_blocks=2,  # each downscales by factor 2
            out_channels_first_layer=64,
            normalization="instance",
            pooling_type='max',
            upsampling_type='conv',
            preactivation=True,
            residual=True,
            activation="ReLU",
            padding=0,
            padding_mode='zeros',
            dropout=0,
        ))
        for mode in ["train","val","test"]:
            print("setting dummy loader args for debugging, replace with good defaults")
            hparams.update(**{f"{mode}_loader_args":dict(
                shuffle=mode=="train",
                batch_size=16,
                num_workers=4
            )})
        return hparams

    def prepare_data(self) -> None:
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x:x.unsqueeze(0) if x.dim()==2 else x) # since grayscale removes the channels
        ])
        self._train_full=SimpleNoisy(MNIST(os.path.expanduser("~/.mldata/mnist"),
                               train=True,
                               download=True,
                                           transform=transform))

        self.train_set=Subset(self._train_full,range(0,int(5e4)))
        self.val_set=Subset(self._train_full,range(int(5e4),int(6e4)))
        self.test_set = SimpleNoisy(MNIST(os.path.expanduser("~/.mldata/mnist"),
                             train=False,
                             download=True,
                                     transform=transform ))

    def setup(self, stage: str):
        self.prepare_data()
