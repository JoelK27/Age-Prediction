import torch, argparse
from PIL import Image
from torchvision import transforms
from src.models.custom_cnn import CustomAgeCNN

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--regression", action="store_true")
    ap.add_argument("--num_classes", type=int, default=10)
    return ap.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    model = CustomAgeCNN(regression=args.regression, num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        out = model(x)
        if args.regression:
            print(f"Predicted Age: {out.item():.2f}")
        else:
            print("Predicted Age Bucket Index:", out.argmax(1).item())

if __name__ == "__main__":
    main()