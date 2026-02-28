## Which Corpus should I choose?
You can select a Corpus in your desired language. 

For example, in Turkish: https://huggingface.co/datasets/uonlp/CulturaX This is one of the Corpuses used by the TanAI team. The Turkish version is 280GB, but this is too much for a model with 42M parameters. You can use utils/corpus_slicer.py to allocate a 10GB section or download only the 10-15GB portion. Use it as a txt file. Train the Base Model with this dataset.

For SFT, you can use https://huggingface.co/datasets/TFLai/Turkish-Alpaca as an example. More options are available on HuggingFace.
