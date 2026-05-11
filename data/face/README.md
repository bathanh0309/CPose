# Face Gallery

Use one directory per identity:

```text
data/face/<person_name>/
├── embeddings.npy
└── meta.json
```

Legacy `emb_00.npy`, `emb_01.npy`, ... files are supported. Run `FaceGallery.load()` to stack them into `embeddings.npy`.
