using Electra.Model.Activations;
using Tensorflow.Keras;

namespace Electra.Model;

public readonly struct BertConfig
{
    public BertConfig(int vocabSize) => VocabSize = vocabSize;

    public int VocabSize { get; }

    public int HiddenSize { get; init; } = 768;

    public int NumHiddenLayers { get; init; } = 12;

    public int NumAttentionHeads { get; init; } = 12;

    public int IntermediateSize { get; init; } = 3072;

    public Activation? HiddenAct { get; init; } = GeluWrapper.Gelu;

    public float HiddenDropoutProb { get; init; } = 0.1f;

    public float AttentionProbsDropoutProb { get; init; } = 0.1f;

    public int MaxPositionEmbeddings { get; init; } = 512;

    public int TypeVocabSize { get; init; } = 2;

    public float InitializerRange { get; init; } = 0.02f;
}
