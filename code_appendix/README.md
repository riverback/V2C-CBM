We provide the codes for building V2C tokenizer in four steps, the saved codebook features with the vocabulary list can be used as V2C tokenizer.

`v2c.sh` is the script which combines the four steps together.

Once the `class2concepts.json` is generated, it can be used for building the V2C-CBM. We construct the V2C-CBM based on the codes provided in https://github.com/YueYANG1996/LaBo, the most improtant modification part in the code is that we don't use the concept selection part in the original repository since our class-specific concepts are directly generated from the V2C tokenizer, and we also modify the initialization part to implement our initialization with concept priors.