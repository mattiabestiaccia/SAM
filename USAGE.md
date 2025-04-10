# Documentazione: Segment Anything Model (SAM)

## Introduzione

Il **Segment Anything Model (SAM)** è un modello di segmentazione di immagini sviluppato da Meta AI Research (FAIR). SAM è in grado di produrre maschere di alta qualità per oggetti in immagini a partire da prompt di input come punti o box, e può essere utilizzato per generare maschere per tutti gli oggetti presenti in un'immagine. È stato addestrato su un dataset di 11 milioni di immagini e 1,1 miliardi di maschere, e dimostra ottime prestazioni in zero-shot su una varietà di attività di segmentazione.

*Nota: Meta ha rilasciato SAM 2, che estende la funzionalità anche ai video. È disponibile su [github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2).*

## Architettura del Modello

SAM utilizza un'architettura basata su Vision Transformer (ViT) e comprende tre componenti principali:
1. **Image Encoder**: Codifica l'intera immagine in una rappresentazione di embedding di immagine
2. **Prompt Encoder**: Codifica vari tipi di prompt (punti, box, ecc.)
3. **Mask Decoder**: Combina gli embedding dell'immagine e del prompt per predire maschere di segmentazione

![Architettura SAM](assets/model_diagram.png)

## Requisiti di Sistema

- Python >= 3.8
- PyTorch >= 1.7
- TorchVision >= 0.8
- Si raccomanda fortemente l'installazione di PyTorch e TorchVision con supporto CUDA

## Installazione

```bash
# Installazione diretta da GitHub
pip install git+https://github.com/facebookresearch/segment-anything.git

# Oppure clone locale e installazione
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

Dipendenze opzionali (necessarie per il post-processing delle maschere, salvataggio in formato COCO, notebook di esempio, esportazione ONNX):
```bash
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

## Utilizzo Base

### Predizione con Prompt
```python
from segment_anything import SamPredictor, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
predictor = SamPredictor(sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)
```

### Generazione Automatica di Maschere
```python
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)
```

### Utilizzo da Linea di Comando
```bash
python scripts/amg.py --checkpoint <path/to/checkpoint> --model-type <model_type> --input <image_or_folder> --output <path/to/output>
```

## Checkpoint del Modello

Sono disponibili tre versioni del modello con backbone di dimensioni diverse:

- **`default` o `vit_h`**: [Modello SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- `vit_l`: [Modello SAM ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [Modello SAM ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

## Esportazione ONNX

Il decoder di maschere di SAM può essere esportato in formato ONNX per essere eseguito in qualsiasi ambiente che supporti ONNX runtime, come un browser web:

```bash
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

## Demo Web

Nella cartella `demo/` è disponibile una semplice applicazione React che mostra come eseguire la predizione di maschere con il modello ONNX esportato in un browser web con multithreading. Vedere [`demo/README.md`](https://github.com/facebookresearch/segment-anything/blob/main/demo/README.md) per maggiori dettagli.

## Dataset SA-1B

Il dataset può essere scaricato da [qui](https://ai.facebook.com/datasets/segment-anything-downloads/). Scaricando il dataset si accettano i termini della Licenza di Ricerca SA-1B Dataset.

Le maschere per immagine sono salvate come file JSON nel seguente formato:

```python
{
    "image": image_info,
    "annotations": [annotation],
}

image_info {
    "image_id": int,              # ID immagine
    "width": int,                 # Larghezza immagine
    "height": int,                # Altezza immagine
    "file_name": str,             # Nome file immagine
}

annotation {
    "id": int,                    # ID annotazione
    "segmentation": dict,         # Maschera salvata in formato COCO RLE
    "bbox": [x, y, w, h],         # Box intorno alla maschera, in formato XYWH
    "area": int,                  # Area in pixel della maschera
    "predicted_iou": float,       # Predizione della qualità della maschera
    "stability_score": float,     # Misura della qualità della maschera
    "crop_box": [x, y, w, h],     # Ritaglio dell'immagine usato per generare la maschera
    "point_coords": [[x, y]],     # Coordinate del punto in input al modello
}
```

Per decodificare una maschera in formato COCO RLE in binario:

```python
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])
```

## Notebook di Esempio

Il repository include notebook di esempio che mostrano:
- Utilizzo di SAM con prompt (`notebooks/predictor_example.ipynb`)
- Generazione automatica di maschere (`notebooks/automatic_mask_generator_example.ipynb`)
- Utilizzo del modello ONNX esportato (`notebooks/onnx_model_example.ipynb`)

## Licenza

Il modello è distribuito sotto la [licenza Apache 2.0](LICENSE).

## Citazione

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```