using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Electra.Pretraining;

public readonly struct PretrainingConfig
{
    private const int MaxSeqLength = 128;

    public PretrainingConfig()
    {

    }

    public bool Debug { get; init; } = false;

    public bool DoTrain { get; init; } = true;

    public bool DoEval { get; init; } = false;

    public int MaxPredictionsPerSeq => (int)(MaskProbability + 0.005f) * MaxSeqLength;

    public float MaskProbability { get; init; } = 0.15f;
}
