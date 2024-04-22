from model import Model
import torch


def main():
    torch.set_float32_matmul_precision('medium')
    model = Model()
    checkpoint = torch.load("MA-Thesis/h0k6pucq/checkpoints/epoch=139-step=11060.ckpt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.inference(...)#TODO

if __name__=="__main__":
    main()
