# PrevenIA

## Introduccion

En este informe exploraremos los datos utilizados para entrenar el modelo **Evo 2**.

Con este fin, descargaremos archivos del [dataset publicado en el paper](https://huggingface.co/datasets/arcinstitute/opengenome2/tree/main), este repositorio se maneja con git lfs (large file storage) y por lo tanto se descargarán los datasets con esta extensión.

```bash
git lfs pull --include="json/pretraining_or_both_phases/eukaryotic_genic_windows/windows_5kb_test_chunk1.jsonl.gz"
git lfs pull --include="fasta/ncbi_eukaryotic_genomes/batch1.tar"
```

