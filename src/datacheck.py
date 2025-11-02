from src.data.dataset import AgeDataset
ds = AgeDataset(r"src\data\UTKface_inthewild\part3", transform=None, regression=True)
for i, (p,a) in enumerate(ds.samples[:10]):
    print(i, a, p)