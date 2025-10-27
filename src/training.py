import os, argparse, math
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.data.dataset import AgeDataset
from src.models.custom_cnn import CustomAgeCNN
from src.utils.metrics import mae, mse
from torch.amp import autocast, GradScaler  # neu

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--num_classes", type=int, default=10)
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--plot_samples", action="store_true")
    # neu:
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--num_workers", type=int, default=max(2, (os.cpu_count() or 2)))
    ap.add_argument("--no_amp", action="store_true", help="Mixed Precision deaktivieren")
    ap.add_argument("--model_width", type=int, default=32)
    return ap.parse_args()

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # schnellere Convs

    # Leichtere Augmentation + kleinere Auflösung
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    age_bins = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,120)]

    full_ds = AgeDataset(args.data_dir, transform=train_tf, regression=args.regression, age_bins=age_bins)
    val_len = int(len(full_ds)*args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    # val mit eval_tf (ersetzen Transform)
    val_ds.dataset.transform = eval_tf

    pin = device == "cuda"
    nw = int(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pin, persistent_workers=(nw>0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, pin_memory=pin, persistent_workers=(nw>0))

    if args.plot_samples:
        batch = next(iter(train_loader))
        imgs, targets = batch
        grid = utils.make_grid(imgs[:16], nrow=8, normalize=True, value_range=(-1,1))
        plt.figure(figsize=(12,4))
        plt.imshow(grid.permute(1,2,0).cpu())
        plt.title("Train Samples")
        plt.axis("off")
        plt.savefig(os.path.join(args.save_dir, "samples.png"))
        plt.close()

    model = CustomAgeCNN(regression=args.regression, num_classes=args.num_classes, width=args.model_width).to(device)
    criterion = nn.L1Loss() if args.regression else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    amp_enabled = (device == "cuda" and not args.no_amp)
    device_type = "cuda" if device == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=amp_enabled)

    best_metric = math.inf
    history = {"train_loss": [], "val_loss": []}
    if args.regression:
        history["val_mae"] = []
    else:
        history["val_acc"] = []

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, enabled=amp_enabled):
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()*xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast(device_type=device_type, enabled=amp_enabled):
                    out = model(xb)
                    loss = criterion(out, yb)
                val_loss += loss.item()*xb.size(0)
                if args.regression:
                    val_mae += mae(out, yb).item()*xb.size(0)
                    val_mse += mse(out, yb).item()*xb.size(0)
                else:
                    preds = out.argmax(1)
                    correct += (preds==yb).sum().item()
                    total += yb.size(0)

        val_loss /= len(val_loader.dataset)
        if args.regression:
            val_mae /= len(val_loader.dataset)
            val_mse /= len(val_loader.dataset)
            metric = val_mae
            print(f"[Epoch {epoch}] TrainLoss={train_loss:.3f} ValLoss={val_loss:.3f} MAE={val_mae:.2f} MSE={val_mse:.2f}")
        else:
            acc = correct/total if total>0 else 0
            metric = 1-acc
            print(f"[Epoch {epoch}] TrainLoss={train_loss:.3f} ValLoss={val_loss:.3f} Acc={acc:.3f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if args.regression:
            history["val_mae"].append(val_mae)
        else:
            history["val_acc"].append(acc)

        if metric < best_metric:
            best_metric = metric
            path = os.path.join(args.save_dir, f"best_epoch_{epoch}.pt")
            torch.save({"model": model.state_dict(),"args": vars(args)}, path)
            print("Gespeichert:", path)

    # Plotten erst nach dem Training
    epochs = range(1, args.epochs+1)
    if args.regression:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.plot(epochs, history["val_loss"], label="Val Loss")
        ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(epochs, history["val_mae"], label="Val MAE", color="orange")
        ax2.set_title("Validation MAE"); ax2.set_xlabel("Epoch"); ax2.legend()
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6,4))
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.plot(epochs, history["val_loss"], label="Val Loss")
        ax1.plot(epochs, history["val_acc"], label="Val Acc")
        ax1.set_title("Training Curves"); ax1.set_xlabel("Epoch"); ax1.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curves.png"))
    plt.close()

if __name__ == "__main__":
    main()