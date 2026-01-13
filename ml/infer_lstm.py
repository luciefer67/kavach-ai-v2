import torch

def predict_sequence(model, seq):
    model.eval()
    with torch.no_grad():
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        out = model(seq)
        prob = torch.softmax(out, dim=1)
        cls = torch.argmax(prob).item()
        return cls, float(prob[0][cls])
    return None, None
    return None, None
    return None, None
    return None, None
    
