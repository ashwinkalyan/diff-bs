## Inference Package containing classes for decoding

### Standard Decoding
Class `Decoder` implemented in `decoder.py` supports standard inference methods like:
- argmax sampling (supports reinforce during training)
- beam search
- diverse beam search with hamming diversity (Vijayakumar et al., 2018)

Helpers for BS and DBS are implemented in `beam_utils.py`.

### Trainable Decoding (this paper)
Class `SetDecoder` is similar to the previous `Decoder` class but implements Differentiable Beam Search. 

`sub_utils` and `dsf.py` implement the selector module and Deep Submodular Function (DSF) respectively.

