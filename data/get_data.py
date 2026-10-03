import wget
path = wget.download(
    "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip",
    out=""
)

print(f"Downloaded pusht.zip to {path}. Please unzip it to the data/ directory.")