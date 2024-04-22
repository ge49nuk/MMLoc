import torch
import torch.nn as nn
from modules.ReCon.get_pt_feats import get_features
from torchvision.models import resnet18, resnet50
from utils.utils import calculate_iou


class CNN(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
    
        # Setup resnet-feature extractor
        resnet = resnet50(pretrained=True)
        self.children_list = []
        for n,c in resnet.named_children():
            self.children_list.append(c)
            if n == "layer4":
                break
        # self.children_list[0] = nn.Conv2d(4, 64, 7, 2, 3) # change input dim
        self.resnet = nn.Sequential(*self.children_list).cuda()
        
        self.mesh_to_rgb = nn.Sequential(
            nn.Linear(768, 224*4)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1), # 7x7x1024
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1), # 7x7x256
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 32, 3, 1, 1), # 7x7x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.lin = nn.Sequential(
            nn.Linear(1568 + 768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            # nn.Linear(1568, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 4),
        )
    
        self.loss = nn.MSELoss()

    def forward(self, input_dict):
        bs = len(input_dict["obj_id"])
        rgb_id_to_rgb = input_dict["rgb_id_to_rgb"]
        obj_id_to_pts = input_dict["obj_id_to_pts"]

        obj_ids = input_dict["obj_id"]
        rgb_ids = input_dict["rgb_id"]
        bboxes = input_dict["bbox"]
        vertices = input_dict["vertices"]
        rgbs = input_dict["rgb"]
        
        # Altern. approach: concat geom. feats before MLP
        vert_feats = get_features(vertices).view(vertices.shape[0], -1)
        res_out = self.resnet(rgbs)
        conv_out = self.conv(res_out).view(rgbs.shape[0], 7*7*32)
        inputs = []
        for i, rgb_id in enumerate(rgb_ids): # iterate through batch samples
            vert_feat = vert_feats[obj_id_to_pts[obj_ids[i]]]
            conv_feat = conv_out[rgb_id_to_rgb[rgb_id]]
            input_feat = torch.cat((vert_feat, conv_feat), dim=0)
            inputs.append(input_feat)
        input_feats = torch.stack(inputs)
        preds = self.lin(input_feats)

        # Embedd 3d and rgb data
        # vert_feats = get_features(vertices).view(vertices.shape[0], -1)
        # vert_feats = vertices[:,::4,:].reshape(-1,768)
        # for obj_id, v_id in obj_id_to_pts.items():
        #     vert_feats[v_id, :] = (float(obj_id)- 8280) / 4780
        # vert_feats = self.mesh_to_rgb(vert_feats).view(vertices.shape[0], 1, 4, 224)
        # assert vert_feats.shape == (vertices.shape[0], 1, 4, 224)
        
        
        
        # Fuse embeddings together
        # inputs = []
        # for i, rgb_id in enumerate(rgb_ids): # iterate through batch samples
        #     vert_feat = vert_feats[obj_id_to_pts[obj_ids[i]]]
        #     rgb = rgbs[rgb_id_to_rgb[rgb_id]]
        #     input_feat = torch.cat((vert_feat.repeat(1,56,1),rgb), dim=0)
        #     assert input_feat.shape == (4, 224, 224)
        #     inputs.append(input_feat)
        # input_feats = torch.stack(inputs)
        # assert input_feats.shape == (bs, 4, 224, 224)
        
        # Predict bbox
        # res_out = self.resnet(input_feats)
        # conv_out = self.conv(res_out).view(bs, 7*7*32)
        # preds = self.lin(conv_out)
        

        gts = (bboxes).cuda()        
        loss = self.loss(preds, gts)
        ious = []
        for gt, pred in zip(gts, preds):
            ious.append(calculate_iou(pred, gt))
        out_dict = {"loss":loss, "ious":torch.tensor(ious), "preds":preds}
        return out_dict
    
    def inference(self, vertices, rgb):
        # Embed 3d and rgb data
        vert_feats = get_features(vertices).view(vertices.shape[0], -1)
        vert_feats = self.mesh_to_rgb(vert_feats).view(vertices.shape[0], 1, 4, 224)
        assert vert_feats.shape == (vertices.shape[0], 1, 4, 224)
        
        # Fuse embeddings together
        input_feats = torch.cat((vert_feats[0].repeat(1,56,1), rgb[0]), dim=0).unsqueeze(0)
        assert input_feats.shape == (1, 4, 224, 224)
        
        input_feats = self.c1(input_feats)
        # Predict bbox
        res_out = self.resnet(input_feats)
        conv_out = self.conv(res_out).view(1, 7*7*32)
        preds = self.lin(conv_out)
        
        return preds[0]