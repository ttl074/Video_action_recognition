from models import LRCN
import torch

def test_batch_independence():
    torch.manual_seed(2026)
    model = LRCN(hidden_size=32,n_layers=1,dropout_rate=0.9,n_classes=50,pretrained=False,cnn_model='resnet18')
    model.eval()
    x = torch.randn(3,4,3,64,64)
    mismatches = []
    with torch.no_grad():
        a = model(x)
        for i in range(len(a)):
            b = model(x[i:i+1])
            if not torch.allclose(a[i],b[0], atol=1e-5):
                mismatches.append((i,(a[i]-b[0]).abs().max()))
    assert not mismatches, f"Mismatches: {mismatches[:5]}"
