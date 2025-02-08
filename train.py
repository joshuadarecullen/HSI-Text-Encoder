import torch.optim as optim
from hsitextmodel import HSITextModel


def train_model(model, dataloader, optimizer, epochs=10, device='cuda'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for hsi, text, mask in dataloader:
            hsi, text, mask = hsi.to(device), text.to(device), mask.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(hsi, text, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


# Initialize model and optimizer
model = HSITextModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
