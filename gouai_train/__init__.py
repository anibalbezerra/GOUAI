"""
gouai_train: train, validate and compare the GOUAI neural networks on new hybrid-model events.

The modules are documented one by one in docs/06_code_reference.md. In short:

    config      experiment configuration (YAML) and the run directory
    rawdata     reading the hybrid events (density grids + labels)
    eos         energy density -> entropy density (lattice QCD equation of state)
    centrality  minimum-bias centrality from the total initial entropy S
    splits      held-out test set and cross-validation folds
    targets     the trend g(S) and the residual targets z the network learns
    features    image channels and scalar inputs (edit here to add inputs)
    store       read access to the prepared events of a run
    pipeline    tf.data input pipeline and data augmentation          (TensorFlow)
    attention   spatial-attention block                                (TensorFlow)
    model       the network, loss and metrics                          (TensorFlow)
    training    training one network, predictions                      (TensorFlow)
    ensemble    the 5-network ensemble, and the published v6 networks  (TensorFlow)
    metrics     accuracy measures, per centrality class, bootstrap
    plots       figures; style: the figure style

Modules marked TensorFlow import it on first use; the others work without it.
"""
__version__ = '1.0.0'
