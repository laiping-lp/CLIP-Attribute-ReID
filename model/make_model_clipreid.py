import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.dataset_name = cfg.DATASETS.NAMES
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.fuse_proj = nn.Linear(1024, 512)
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        if dataset_name == "uavhuman_attr":
            self.prompt_learner = PromptLearnerAttr(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        else:
            self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x = None, label=None, get_image = False, get_text = False, get_expansion=False, get_i2t=False,cam_label= None, view_label=None,gender_label=None,hat_label=None,backpack_label=None,ucc_label=None,ucs_label=None,lcc_label=None,lcs_label=None):
        if get_expansion == True:
            if self.dataset_name == "uavhuman_attr":
                prompts = self.prompt_learner(self.training,label,gender_label,ucc_label,ucs_label,lcc_label,lcs_label,hat_label,backpack_label)
            else:
                prompts = self.prompt_learner(self.training,label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                image_features = image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                image_features = image_features_proj[:,0]
            # 做法一：拼接 concat（信息保留最多，但维度变 1024）先归一化再拼
            t = torch.nn.functional.normalize(text_features, dim=-1)
            i = torch.nn.functional.normalize(image_features, dim=-1)
            fused = self.fuse_proj(torch.cat([i, t], dim=-1))  # [B,512]
            fused = torch.nn.functional.normalize(fused, dim=-1)
            return image_features         

        if get_text == True:
            if self.dataset_name == "uavhuman_attr":
                prompts = self.prompt_learner(self.training,label,gender_label,ucc_label,ucs_label,lcc_label,lcs_label,hat_label,backpack_label)
            else:
                prompts = self.prompt_learner(self.training,label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            if get_i2t == True:
                if self.dataset_name == "uavhuman_attr":
                    prompts = self.prompt_learner(self.training,label,gender_label,ucc_label,ucs_label,lcc_label,lcs_label,hat_label,backpack_label)
                else:
                    prompts = self.prompt_learner(self.training,label)
                t_feats = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
                i_feats = img_feature_proj
                image_logits = self.classifier_proj(i_feats)
                text_logits = self.classifier_proj(t_feats)
                cls_score = self.classifier(feat)
                cls_score_proj = self.classifier_proj(feat_proj)
                return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj],t_feats,i_feats,text_logits,image_logits
            else:
                cls_score = self.classifier(feat)
                cls_score_proj = self.classifier_proj(feat_proj)
                return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

        

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, get_train,label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 

# A photo of a <Gender> person wearing <UCC> <UCS> upper clothes and <LCC> <LCS> lower clothes, with <Hat> hat and <Backpack> backpack.

class PromptLearnerAttr(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding,
                    num_gender=2,num_ucc=12,num_ucs=4,num_lcc=12,num_lcs=4,num_hat=5,num_backpack=5,
        ):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a <Gender> wearing <UCC> <UCS> and <LCC> <LCS>, with <Hat> hat and <Backpack> backpack."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        # 4. 为每个属性建立 soft prompt 表
        #    这里每个属性只用 1 个 token；如果想更强表示力，可以设成 >1
        n_ctx_gender   = 4
        n_ctx_ucc      = 4
        n_ctx_ucs      = 4
        n_ctx_lcc      = 4
        n_ctx_lcs      = 4
        n_ctx_hat      = 4
        n_ctx_backpack = 4
        n_ctx_cls = 4

        n_cls_ctx = 32

        self.gender_ctx = nn.Parameter(
            torch.empty(num_gender, n_ctx_gender, ctx_dim, dtype=dtype)
        )
        self.ucc_ctx = nn.Parameter(
            torch.empty(num_ucc, n_ctx_ucc, ctx_dim, dtype=dtype)
        )
        self.ucs_ctx = nn.Parameter(
            torch.empty(num_ucs, n_ctx_ucs, ctx_dim, dtype=dtype)
        )
        self.lcc_ctx = nn.Parameter(
            torch.empty(num_lcc, n_ctx_lcc, ctx_dim, dtype=dtype)
        )
        self.lcs_ctx = nn.Parameter(
            torch.empty(num_lcs, n_ctx_lcs, ctx_dim, dtype=dtype)
        )
        self.hat_ctx = nn.Parameter(
            torch.empty(num_hat, n_ctx_hat, ctx_dim, dtype=dtype)
        )
        self.backpack_ctx = nn.Parameter(
            torch.empty(num_backpack, n_ctx_backpack, ctx_dim, dtype=dtype)
        )
        # 有标签的 cls prompt 表
        self.cls_ctx = nn.Parameter(torch.empty(num_class, n_ctx_cls, ctx_dim, dtype=dtype))
        nn.init.normal_(self.cls_ctx, std=0.02)

        # 无标签时用的“uncond cls prompt”
        self.cls_ctx_uncond = nn.Parameter(torch.empty(1, n_ctx_cls, ctx_dim, dtype=dtype))
        nn.init.normal_(self.cls_ctx_uncond, std=0.02)
        
        # cls_vectors = torch.empty(num_class, n_ctx_cls, ctx_dim, dtype=dtype) 
        # nn.init.normal_(cls_vectors, std=0.02)
        # self.cls_ctx = nn.Parameter(cls_vectors) 

         

        for param in [self.gender_ctx, self.ucc_ctx, self.ucs_ctx,
                      self.lcc_ctx, self.lcs_ctx, self.hat_ctx, self.backpack_ctx]:
            nn.init.normal_(param, std=0.02)
        
        self.ctx_dim = ctx_dim

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, 
                get_train,
                label,
                gender_idx: torch.Tensor,    # [B]
                ucc_idx: torch.Tensor,       # [B]
                ucs_idx: torch.Tensor,       # [B]
                lcc_idx: torch.Tensor,       # [B]
                lcs_idx: torch.Tensor,       # [B]
                hat_idx: torch.Tensor,       # [B]
                backpack_idx: torch.Tensor,  # [B]):
    ):
        """
        输入：每张图的属性标签索引
        输出：对应 batch 的 prompt embedding，形状 [B, L_text, ctx_dim]
        """

        B = gender_idx.shape[0]
        
        # 1. prefix / suffix broadcast 到 batch
        prefix = self.token_prefix.expand(B, -1, -1)  # [B, P, dim]
        suffix = self.token_suffix.expand(B, -1, -1)  # [B, S, dim]
        
        gender_ctx   = self.gender_ctx[gender_idx]      # [B, n_ctx_gender, dim]
        ucc_ctx      = self.ucc_ctx[ucc_idx]            # [B, n_ctx_ucc, dim]
        ucs_ctx      = self.ucs_ctx[ucs_idx]            # [B, n_ctx_ucs, dim]
        lcc_ctx      = self.lcc_ctx[lcc_idx]            # [B, n_ctx_lcc, dim]
        lcs_ctx      = self.lcs_ctx[lcs_idx]            # [B, n_ctx_lcs, dim]
        hat_ctx      = self.hat_ctx[hat_idx]            # [B, n_ctx_hat, dim]
        backpack_ctx = self.backpack_ctx[backpack_idx]  # [B, n_ctx_backpack, dim]
        
        if label is None:
            cls_part = self.cls_ctx_uncond.expand(B, -1, -1)  # [B,n_ctx_cls,D]
        else:
            label = label.view(-1).long()                     # 确保 [B]
            cls_part = self.cls_ctx[label]                    # [B,n_ctx_cls,D]
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_part,
                gender_ctx,
                ucc_ctx,
                ucs_ctx,
                lcc_ctx,
                lcs_ctx,
                hat_ctx,
                backpack_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 
        
        return prompts 


        