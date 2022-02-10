import torch
import src.Python.exportsd as exportsd

class BasicConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
       
        self.stack = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, bias=False, **kwargs),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.stack(x)

if __name__ == '__main__':
    # Create model
    model = BasicConv1d(1, 32)
    
    #Export model to .dat file for ingestion into TorchSharp
    f = open("bug510.dat", "wb")
    exportsd.save_state_dict(model.to("cpu").state_dict(), f)
    f.close()