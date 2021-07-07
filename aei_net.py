import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.models import resnet101
import pytorch_lightning as pl

from model.AEINet import ADDGenerator, MultilevelAttributesEncoder
from model.MultiScaleDiscriminator import MultiscaleDiscriminator

from model.loss import GANLoss, AEI_Loss

from dataset import *


class AEINet(pl.LightningModule):
    def __init__(self, hp):
        super(AEINet, self).__init__()
        self.hp = hp

        self.G = ADDGenerator(hp.arcface.vector_size)
        self.E = MultilevelAttributesEncoder()
        self.D = MultiscaleDiscriminator(3)

        self.Z = resnet101(num_classes=256)
        self.Z.load_state_dict(torch.load(hp.arcface.pth, map_location='cpu'))

        self.Loss_GAN = GANLoss()
        self.Loss_E_G = AEI_Loss()


    def forward(self, target_img, source_img):
        z_id = self.Z(F.interpolate(source_img, size=112, mode='bilinear'))
        z_id = F.normalize(z_id)
        z_id = z_id.detach()

        feature_map = self.E(target_img)

        output = self.G(z_id, feature_map)

        output_z_id = self.Z(F.interpolate(output, size=112, mode='bilinear'))
        output_z_id = F.normalize(output_z_id)
        output_feature_map = self.E(output)
        return output, z_id, output_z_id, feature_map, output_feature_map


    def training_step(self, batch, batch_idx, optimizer_idx):
        target_img, source_img, same = batch

        if optimizer_idx == 0:
            output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

            self.generated_img = output

            if batch_idx % self.hp.trainer.d_per_g_train_ratio:
                output_multi_scale_val = self.D(output)
                loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
                loss_E_G, loss_attr, loss_id, loss_rec = self.Loss_E_G(target_img, 
                    output, feature_map, output_feature_map, z_id, output_z_id, same)

                loss_G = loss_E_G + loss_GAN

                self.log("gan_loss", loss_GAN.item(),prog_bar=True)
                self.log("g_loss", loss_G.item(),prog_bar=True)
                self.log("attr_loss", loss_attr.item(),prog_bar=True)
                self.log("id_loss", loss_id.item(), prog_bar=True)
                self.log("rec_loss", loss_rec.item(),prog_bar=True)

                return loss_G

        else:
            multi_scale_val = self.D(target_img)
            output_multi_scale_val = self.D(self.generated_img.detach())

            loss_D_fake = self.Loss_GAN(multi_scale_val, True)
            loss_D_real = self.Loss_GAN(output_multi_scale_val, False)

            loss_D = loss_D_fake + loss_D_real

            self.log("d_loss", loss_D.item(), prog_bar=True)
            return loss_D

    def validation_step(self, batch, batch_idx):
        target_img, source_img, same = batch

        output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

        self.generated_img = output

        output_multi_scale_val = self.D(output)
        loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
        loss_E_G, loss_attr, loss_id, loss_rec = self.Loss_E_G(target_img, output, 
            feature_map, output_feature_map, z_id, output_z_id, same)
        loss_G = loss_E_G + loss_GAN

        return {"loss": loss_G, "target": target_img[0].cpu(), 
            "source": source_img[0].cpu(),  "output": output[0].cpu()}

    def validation_epoch_end(self, outputs):
        """The number of validation_step call depends on the batch_size in train.yaml, and thus
        the method is necessary to accumulate the results.
        
        Args:
            outputs (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        val_img = []
        for x in outputs:
            val_img = val_img + [x['target'], x['source'], x["output"]]
 
        val_img = torchvision.utils.make_grid(val_img, nrow=3)

        self.log("val_loss", loss.item(), prog_bar=True)
        self.logger.experiment.add_image("val_img", val_img, self.global_step)

        return {"loss": loss, "image": val_img}

    def configure_optimizers(self):
        lr_g = self.hp.model.learning_rate_E_G
        lr_d = self.hp.model.learning_rate_D
        b1 = self.hp.model.beta1
        b2 = self.hp.model.beta2

        opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.E.parameters()), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            ])
        dataset = AEI_Dataset(self.hp.data.trainset_dir, transform=transform)
        return DataLoader(dataset, batch_size=self.hp.trainer.batch_size, num_workers=self.hp.trainer.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
        ])
        dataset = AEI_Val_Dataset(self.hp.data.valset_dir, transform=transform)
        return DataLoader(dataset, batch_size=1, shuffle=False)
